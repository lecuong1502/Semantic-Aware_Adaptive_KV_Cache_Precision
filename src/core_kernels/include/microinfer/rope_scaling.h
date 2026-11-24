#pragma once

#include <stdexcept>
#include <string>

namespace microinfer
{

  // How a model's RoPE frequencies depart from theta^(-2j / head_dim), if
  // they do (#209). Qwen2.5's do not: the default, `scaled` false, is plain
  // RoPE. Llama-3.2's are "llama3" scaled, as HuggingFace's
  // `_compute_llama3_parameters` forms them: a frequency whose wavelength
  // exceeds original_max_position_embeddings / low_freq_factor is divided by
  // `factor`, one shorter than original_max_position_embeddings /
  // high_freq_factor is kept, and one between is blended linearly in
  // original_max_position_embeddings / wavelength (rope_angle.cuh).
  //
  // A plain struct, passed by value into the kernels that form an angle.
  struct RopeScaling
  {
    bool scaled = false;
    double factor = 1.0;
    double low_freq_factor = 1.0;
    double high_freq_factor = 1.0;
    double original_max_position_embeddings = 0.0;

    // llama3 scaling, refused where the blend would be undefined: a factor
    // below one, high_freq_factor not above low_freq_factor (the blend
    // divides by their difference), or no original window.
    static RopeScaling llama3(double factor, double low_freq_factor,
                              double high_freq_factor,
                              double original_max_position_embeddings)
    {
      if (!(factor >= 1.0))
      {
        throw std::invalid_argument(
            "rope scaling factor must be at least 1, got " +
            std::to_string(factor));
      }
      if (!(low_freq_factor > 0.0) || !(high_freq_factor > low_freq_factor))
      {
        throw std::invalid_argument(
            "rope scaling needs 0 < low_freq_factor < high_freq_factor, got " +
            std::to_string(low_freq_factor) + " and " +
            std::to_string(high_freq_factor));
      }
      if (!(original_max_position_embeddings > 0.0))
      {
        throw std::invalid_argument(
            "rope scaling needs a positive original_max_position_embeddings, "
            "got " +
            std::to_string(original_max_position_embeddings));
      }
      return RopeScaling{true, factor, low_freq_factor, high_freq_factor,
                         original_max_position_embeddings};
    }
  };

} // namespace microinfer
