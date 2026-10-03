#pragma once

#include <cuda_fp16.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "microinfer/device_buffer.h"
#include "microinfer/paged_kv_cache.h"
#include "microinfer/rope_table.h"
#include "microinfer/shadow_store.h"

// P, the tokens per page, is a build-time parameter (ADR-0004): 32 by
// default, and set with -DMICROINFER_PAGE_TOKENS for Milestone 3's ablation
// over 16, 32 and 64. It is defined by CMake and nowhere else, so no source
// file can quietly assume a value.
#ifndef MICROINFER_PAGE_TOKENS
#error "MICROINFER_PAGE_TOKENS must be defined by the build (CMakeLists.txt)"
#endif

namespace microinfer
{

  constexpr int kPageTokens = MICROINFER_PAGE_TOKENS;

  // The page_index of a layer's two open pages at a quantised tier: never a
  // position's page, so they cannot collide with one.
  constexpr std::array<int, 2> kOpenPages{-1, -2};

  // The page_index a downgrade's or an upgrade's new page is allocated
  // under while the page it replaces is still read (#93): never a
  // position's page, nor an open page.
  constexpr int kStagingPage = -3;

  // The page_index of the FP16 page a move from a shadow to a quantised
  // tier uploads the shadow to, to quantise from (#95, #96), freed before
  // the move returns.
  constexpr int kShadowUploadPage = -4;

  // Which halves of a page a quantised tier quantises. Both is the tier; the
  // other two are a diagnostic (#18): the page is stored at FP16, and when it
  // is sealed only the named half is replaced by what the tier's quantiser
  // would return for it, so that the cost of a tier can be split between
  // keys and values. The engine never runs them unless asked.
  enum class Halves : int
  {
    Both = 0,
    Keys = 1,
    Values = 2,
  };

  // A tier map (#90): row l names the tier each page of positions of layer l
  // is born at, page i at [l][i].
  using TierMap = std::vector<std::vector<Tier>>;

  // The engine's KV cache, on pages (#14): one layer's keys and values for
  // page_tokens consecutive positions per page, allocated from a
  // PagedKVCache as the sequence grows.
  //
  // An FP16 page holds page_tokens rows of keys, then page_tokens rows of
  // values, each row (kv_heads, head_dim) in fp16. Page `i` of layer `l` is
  // the page table entry (l, i), and holds positions [i * page_tokens,
  // (i + 1) * page_tokens).
  //
  // At a quantised tier (#18, ADR-0011) the tier is static. A page (l, i) is
  // allocated at the tier once all its positions are reserved, and *sealed*
  // once, when its last position is stored: quantised into place, in
  // quant.h's layout, and never written again. Until then its positions
  // are in one of the layer's two open pages, (l, kOpenPages[k]), FP16 and
  // allocated once. A key channel's scale spans a whole page, so a page
  // being filled cannot be quantised (ADR-0005). A page changes tier only
  // by downgrade() or upgrade() (#93, #96), each of which puts a new page,
  // at the target tier, in its place: a sealed page is still never written
  // again.
  //
  // Attention at a quantised tier is causal in the sense decode is: a query
  // reads the pages before its own as sealed, and its own page at FP16,
  // whether it arrived alone or in a chunk of hundreds. A chunk that begins
  // mid-page needs that page's earlier positions from the open page they are
  // in, while its own last, partial page needs an open page to go into; the
  // two open pages are for those two, so the second never overwrites the
  // first before attention has read it.
  //
  // A page's tier is the page table's: the allocator records it for every
  // (layer, page). This class reads it there to address a page, at its own
  // tier's size, to seal a page of positions, at its own tier, and to tell
  // attention the tier each sealed page is read at (#91). The open pages
  // are always FP16. The cache's own tier is the tier a page of positions
  // is born at where the map names none; the diagnostic Halves' round trip
  // goes by it too, its tier being the cache's by definition.
  //
  // The allocator may move any page whenever it frees one (ADR-0007), so this
  // class never keeps a page's address. store() and attention() each resolve
  // what they need immediately before their launches.
  class KVPages
  {
  public:
    // The bytes one FP16 page needs: page_tokens rows of keys and as many of
    // values, kv_width fp16 elements each. The one place this layout's size
    // is written down.
    static std::size_t page_bytes_for(int page_tokens, std::size_t kv_width);

    // Whether a cache at `tier`, built from `tier_map`, seals (ADR-0011,
    // amended by #90 and #93): whenever any page of positions is born at a
    // quantised tier, that is, at a quantised `tier` or when the map names a
    // tier but `tier`; or when asked to, with `always_seal`, so that a
    // cache whose every page is born at FP16 can downgrade them. The one
    // place the rule is written down; seals() is it, and a caller that must
    // size an allocator before the cache exists asks here.
    static bool seals_for(Tier tier, const TierMap &tier_map,
                          bool always_seal = false);

    // The allocator's page size at the tier the pages are stored at must be
    // a page's there: page_bytes_for(page_tokens, kv_heads * head_dim) at
    // FP16, else quantised_page_layout(...).page_bytes. At a quantised tier
    // the open pages are FP16, so its FP16 pages must be FP16-sized too; with
    // Halves other than Both, the pages themselves are stored at FP16. The
    // allocator must outlive this.
    //
    // `tier_map`, if given, names the tier each page of positions is born
    // at: tier_map[l][i] for page i of layer l, and `tier` beyond the end of
    // a layer's row. It needs a row per layer, and no diagnostic Halves, and
    // the allocator's page size at every tier it names must be a page's
    // there.
    //
    // `always_seal` makes the cache seal however its pages are born (see
    // seals_for): its pages can then be downgraded (#93).
    KVPages(PagedKVCache &allocator, int layers, int page_tokens, int kv_heads,
            int head_dim, Tier tier, Halves halves = Halves::Both,
            TierMap tier_map = {}, bool always_seal = false);
    // Frees every page this cache holds, newest first, so that each free is
    // of a tail page and moves nothing.
    ~KVPages();

    KVPages(const KVPages &) = delete;
    KVPages &operator=(const KVPages &) = delete;

    // Makes room for positions [0, tokens) in every layer, allocating only the
    // pages not already held: on demand, never for a maximum length. At a
    // quantised tier those are the pages whose every position is below
    // `tokens`, and the open pages, on the first call. If an allocation
    // fails, as it will when contention leaves no memory, everything taken
    // in the call is given back before the error propagates, and the cache
    // is as it was.
    void reserve(int tokens);

    // Writes n tokens' keys and values, (n, kv_width) each, at positions
    // [start, start + n) of `layer`. At a quantised tier a page whose last
    // position this writes is sealed.
    void store(int layer, const __half *keys, const __half *values, int start,
               int n);

    // Attention over positions [0, seq_k) of `layer`, which must all have
    // been stored, for queries at the last seq_q of them. At a quantised
    // tier `keys` and `values` are the rows the queries stored, (seq_q,
    // kv_width) each, from which each query reads its own page (see above);
    // at FP16 they are not read and may be null.
    //
    // `mass`, if given, receives a decode step's per-page attention mass
    // (#98): (heads, pages) fp32, each page's share of each query head's
    // softmax, for the pages of positions [0, seq_k), the last of them the
    // query's own, open page. Only for one query, seq_q 1. Asking for it
    // changes no bit of `out`.
    void attention(int layer, const __half *q, const __half *k_bias,
                   const RopeTable *rope, __half *out, int seq_q, int seq_k,
                   int heads, int kv_heads, int head_dim, const __half *keys,
                   const __half *values, float *mass = nullptr);

    // Moves page `page` of `layer`, a page of positions sealed at FP16, to
    // the lower tier `target` at runtime (#93): allocated at `target`, its
    // FP16 bytes copied to its shadow if it has none (#94), quantised from
    // FP16, put in its place in the page table, and the old page freed, its
    // tier's tail moving into the slot (ADR-0007). Quantised from the page
    // itself while it is FP16, and from its shadow once it is below (#95),
    // never from its codes: a page at tier T is always quantise_page(FP16,
    // T). A page born at a quantised tier has no shadow, and never moves.
    // The page then holds quantise_page's bytes for its positions at
    // `target`, as a page born there does, and attention reads it there. If
    // the allocation fails, as it will when contention leaves no memory, the
    // cache is as it was. A device fault reported by the free's synchronise
    // is a sticky CUDA error, after which the context, and this cache with
    // it, is unusable. Needs a cache that seals, and no diagnostic Halves.
    void downgrade(int layer, int page, Tier target);

    // Moves page `page` of `layer`, a page downgraded before, back up to the
    // higher tier `target` (#96), from its shadow: to FP16, the shadow's
    // bytes, exactly the page's before its first downgrade; to a quantised
    // tier, quantise_page of the shadow there. Then as a downgrade: put in
    // the page's place, the old page freed. The shadow is kept. If an
    // allocation fails, the cache is as it was.
    void upgrade(int layer, int page, Tier target);

    // What a downgrade or an upgrade took (#97): the page, its tiers and
    // its bytes at each, and the wall time of the whole move, from once the
    // device has finished what it was given before, to the end of its last
    // free. Every step of a move is synchronous, so that is its whole cost.
    // Of it, `shadow_seconds` is the shadow's copy, either way: to the host
    // on a first downgrade, back to the device on a move from the shadow.
    // A downgrade from FP16 whose shadow is held copies nothing.
    struct MoveRecord
    {
      PageKey key;
      Tier from_tier;
      Tier to_tier;
      std::size_t from_bytes;
      std::size_t to_bytes;
      double seconds;
      double shadow_seconds;
    };
    // The last move this cache made, if any.
    const std::optional<MoveRecord> &last_move() const { return last_move_; }

    // The FP16 shadows, one for each page whose downgrade has begun, in
    // pinned host memory, freed with this cache (#94). A downgrade that
    // fails after taking it leaves it, still the page's bytes, for the next.
    const ShadowStore &shadows() const { return shadows_; }

    // The tier the page table records for a page; the open pages are
    // (layer, kOpenPages[k]). PageNotFound if the table holds no such page.
    Tier page_tier(PageKey key) const;

    int page_tokens() const { return page_tokens_; }
    // Pages of positions held in each layer; the open pages are not counted.
    int pages_per_layer() const { return pages_; }
    int layers() const { return layers_; }
    int capacity_tokens() const
    {
      return (pages_ + (seals() ? 1 : 0)) * page_tokens_;
    }
    // The size of one page of positions at storage_tier(), the cache's own:
    // a page a tier map places at another tier has that tier's size.
    std::size_t page_bytes() const { return page_bytes_; }
    Tier tier() const { return tier_; }
    Halves halves() const { return halves_; }
    // Whether pages of positions are sealed (ADR-0011): whenever any is
    // born at a quantised tier. A page is then allocated once all its
    // positions are reserved, and filled in an open page until then, at
    // every tier, FP16 included, so that one rule holds for the whole cache.
    bool seals() const { return seals_; }
    // The tier page `page` of `layer` is born at: the map's, or the cache's.
    Tier birth_tier(int layer, int page) const;
    // Whether every page of positions is born at one tier: the map names no
    // tier but the cache's. A page beyond a row is born at the cache's, so a
    // map of INT4 over a cache at FP16 is two tiers.
    bool born_at_one_tier() const { return !mixed_; }
    // The tier the pages of positions are allocated at: the cache's own,
    // except under a diagnostic Halves, where it is FP16.
    Tier storage_tier() const { return storage_tier_; }

    std::size_t kv_width() const { return kv_width_; }

  private:
    // store() at a quantised tier, span by span (ADR-0011).
    void store_sealed(int layer, const __half *keys, const __half *values,
                      int start, int n);
    // Seals the page of positions `span` from rows (page_tokens, kv_width) of
    // keys and of values, into its place at its own tier.
    // Copies rows of keys and of values into an FP16 page, keys then values.
    void copy_rows(__half *page, const __half *keys, const __half *values);
    // The key of a page that can change tier, a sealed page of positions of
    // a cache that seals, without a diagnostic Halves, or why not, for
    // `move`, "downgrade" or "upgrade".
    PageKey require_movable(int layer, int page, const char *verb) const;
    // "page i of layer l", for messages.
    static std::string page_name(PageKey key);
    // A page from `current` to `target`, either way: downgrade's and
    // upgrade's shared steps, once each has checked the direction.
    void move(PageKey key, Tier current, Tier target);
    // Throws std::out_of_range unless `layer` is one of this cache's.
    void require_layer(int layer) const;
    // One page's bytes at `tier`: FP16's layout, or quant.h's.
    std::size_t bytes_at(Tier tier) const;
    // Throws unless the allocator's pages at `tier` are bytes_at(tier).
    void require_page_size(Tier tier, const char *what) const;
    void seal(int layer, int span, const __half *keys, const __half *values);
    __half *open_page(int layer, int which);
    // A page as the page table holds it now: its tier, and its address at
    // the size that tier gives it, valid until the allocator's next
    // operation. The one place a page is looked up.
    struct ResolvedPage
    {
      Tier tier;
      CUdeviceptr address;
    };
    ResolvedPage resolve_page(PageKey key) const;

    // The layer's table of pages of positions on the device, for the pages
    // held now: each page's address, and the tier the page table records
    // for it. Every layer's table is resolved at once, and again whenever
    // the allocator's generation has moved since: no address outlives an
    // allocate or a free, anyone's, and a decode step with no allocator
    // operation between its launches uploads nothing. A page reaches
    // another tier only by an allocation, so its tier is never staler than
    // its address. Both are null while no page of positions is held.
    struct LayerTable
    {
      const unsigned long long *pages;
      const std::uint8_t *tiers;
    };
    LayerTable resolve(int layer);

    PagedKVCache &allocator_;
    int layers_;
    int page_tokens_;
    int kv_heads_;
    int head_dim_;
    std::size_t kv_width_;
    Tier tier_;
    Halves halves_;
    Tier storage_tier_;
    TierMap tier_map_;
    // Every tier a page of positions may be at: the cache's, the map's, and
    // any a downgrade moved one to. Attention has a layout for each.
    std::array<bool, kTierCount> may_be_at_{};
    bool seals_ = false;
    bool mixed_ = false;
    std::size_t page_bytes_;
    int pages_ = 0;
    bool open_pages_ = false;
    // Per layer: which open page holds the positions of its last, partial
    // page, and which held the first stored page's earlier positions at the
    // last store, which attention reads.
    std::vector<int> current_open_;
    std::vector<int> attend_open_;
    // Per layer: how many pages of positions are sealed, from page 0 up.
    std::vector<int> sealed_;
    ShadowStore shadows_;
    std::optional<MoveRecord> last_move_;
    // For a diagnostic Halves only: a quantised page and its dequantised
    // halves, the scratch a seal goes through.
    std::unique_ptr<DeviceBuffer> scratch_page_;
    std::unique_ptr<DeviceBuffer> scratch_halves_;
    // layers_ tables of table_stride_ addresses each, then layers_ of as
    // many tiers.
    std::unique_ptr<DeviceBuffer> tables_;
    int table_stride_ = 0;
    int resolved_pages_ = -1;
    std::uint64_t resolved_generation_ = 0;
  };

} // namespace microinfer
