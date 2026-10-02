#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "microinfer/device_buffer.h"

namespace microinfer
{

  // The importance score of every (layer, page) of a cache (#99): an
  // exponentially weighted moving average of the page's attention mass
  // (#98), averaged over the layer's query heads, folded in at every
  // decode step: score = alpha * mass + (1 - alpha) * score. A page's
  // first observation seeds its score; a page never observed has none,
  // NaN, and the controller, to come, will order those by position (#88).
  // A mass is NaN only if the logits were, and such a page then reads as
  // never observed. The scores belong to the cache's pages, not to a pass
  // over them: a page rewritten in place, downgraded or upgraded keeps its
  // score.
  //
  // The scores stay on the device. fold() launches one kernel and copies
  // nothing; only download() copies them to the host, when a plan is made.
  // Each fp32 operation is rounded once, never fused, so a host computation
  // in fp32 in the same order is the scores to the bit.
  class ImportanceScores
  {
  public:
    // alpha in (0, 1]: the weight of the newest observation.
    static constexpr float kDefaultAlpha = 0.2f;

    ImportanceScores(int layers, int max_pages, float alpha = kDefaultAlpha);

    ImportanceScores(const ImportanceScores &) = delete;
    ImportanceScores &operator=(const ImportanceScores &) = delete;

    // Folds one decode step's masses for `layer`: `mass` is (heads, pages)
    // fp32 on the device, as the attention kernel writes it, pages counted
    // from page 0. Launches one kernel; copies nothing.
    void fold(int layer, const float *mass, int heads, int pages);

    // Every layer's scores, (layers, max_pages), NaN where a page has never
    // been observed: the one copy to the host.
    std::vector<float> download();

    int layers() const { return layers_; }
    int max_pages() const { return max_pages_; }
    float alpha() const { return alpha_; }
    // How many times the scores have been copied to the host.
    std::size_t downloads() const { return downloads_; }
    // The device memory the scores hold.
    std::size_t nbytes() const;

  private:
    int layers_;
    int max_pages_;
    float alpha_;
    std::unique_ptr<DeviceBuffer> scores_;
    std::size_t downloads_ = 0;
  };

} // namespace microinfer
