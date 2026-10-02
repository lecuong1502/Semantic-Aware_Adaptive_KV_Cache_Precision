#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

#include "microinfer/check.h"
#include "microinfer/importance_scores.h"
#include "microinfer/launch.h"

namespace microinfer
{
  namespace
  {

    constexpr int kBlockThreads = 256;

    __global__ void fill_kernel(float *scores, size_t count, float value)
    {
      for (size_t i =
               static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
           i < count; i += static_cast<size_t>(gridDim.x) * blockDim.x)
      {
        scores[i] = value;
      }
    }

    // One thread per page: its mass over the heads, summed in head order
    // and divided by their count, folded into its score. Every operation is
    // an explicitly rounded intrinsic, so nothing is fused into an FMA.
    __global__ void fold_kernel(float *__restrict__ scores,
                                const float *__restrict__ mass, int heads,
                                int pages, float alpha)
    {
      for (int page = blockIdx.x * blockDim.x + threadIdx.x; page < pages;
           page += gridDim.x * blockDim.x)
      {
        float total = 0.0f;
        for (int h = 0; h < heads; ++h)
        {
          total = __fadd_rn(total, mass[static_cast<size_t>(h) * pages + page]);
        }
        const float observed = __fdiv_rn(total, static_cast<float>(heads));
        const float old = scores[page];
        scores[page] = isnan(old)
                           ? observed
                           : __fadd_rn(__fmul_rn(alpha, observed),
                                       __fmul_rn(__fsub_rn(1.0f, alpha), old));
      }
    }

  } // namespace

  ImportanceScores::ImportanceScores(int layers, int max_pages, float alpha)
      : layers_(layers), max_pages_(max_pages), alpha_(alpha)
  {
    if (layers <= 0 || max_pages <= 0)
    {
      throw std::invalid_argument("layers and max_pages must be positive");
    }
    if (!(alpha > 0.0f && alpha <= 1.0f))
    {
      throw std::invalid_argument("alpha, the newest observation's weight, is "
                                  "in (0, 1]; got " +
                                  std::to_string(alpha));
    }
    const size_t count = static_cast<size_t>(layers) * max_pages;
    scores_ = std::make_unique<DeviceBuffer>(count * sizeof(float));
    fill_kernel<<<grid_stride_blocks(count, kBlockThreads), kBlockThreads>>>(
        scores_->as<float>(), count, NAN);
    cuda_check(cudaGetLastError(), "fill_kernel launch");
  }

  void ImportanceScores::fold(int layer, const float *mass, int heads,
                              int pages)
  {
    if (layer < 0 || layer >= layers_)
    {
      throw std::out_of_range("layer " + std::to_string(layer) + " of " +
                              std::to_string(layers_));
    }
    if (pages < 0 || pages > max_pages_ || heads <= 0)
    {
      throw std::invalid_argument(
          std::to_string(pages) + " pages of " + std::to_string(heads) +
          " heads; the scores hold up to " + std::to_string(max_pages_) +
          " pages, and a mass has a head at least");
    }
    if (pages == 0)
    {
      return;
    }
    fold_kernel<<<grid_stride_blocks(static_cast<size_t>(pages), kBlockThreads),
                  kBlockThreads>>>(scores_->as<float>() +
                                       static_cast<size_t>(layer) * max_pages_,
                                   mass, heads, pages, alpha_);
    cuda_check(cudaGetLastError(), "fold_kernel launch");
  }

  std::vector<float> ImportanceScores::download()
  {
    std::vector<float> host(static_cast<size_t>(layers_) * max_pages_);
    cuda_check(cudaMemcpy(host.data(), scores_->raw(),
                          host.size() * sizeof(float), cudaMemcpyDeviceToHost),
               "cudaMemcpy importance scores device-to-host");
    ++downloads_;
    return host;
  }

  std::size_t ImportanceScores::nbytes() const
  {
    return static_cast<size_t>(layers_) * max_pages_ * sizeof(float);
  }

} // namespace microinfer
