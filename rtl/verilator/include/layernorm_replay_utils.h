#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace lnreplay {

inline float fp16_to_float(uint16_t bits) {
  const bool sign = ((bits >> 15) & 1u) != 0u;
  const int exp = (bits >> 10) & 0x1F;
  const int frac = bits & 0x3FF;
  const float sign_v = sign ? -1.0f : 1.0f;
  if (exp == 0 && frac == 0) {
    return 0.0f;
  }
  if (exp == 0) {
    return sign_v * (static_cast<float>(frac) / 1024.0f) *
           std::ldexp(1.0f, -14);
  }
  if (exp == 31) {
    return sign_v * 65504.0f;
  }
  return sign_v * (1.0f + static_cast<float>(frac) / 1024.0f) *
         std::ldexp(1.0f, exp - 15);
}

inline int round_half_even(float x) {
  const long long floor_i = static_cast<long long>(std::floor(x));
  const float frac = x - static_cast<float>(floor_i);
  if (frac > 0.5f) {
    return static_cast<int>(floor_i + 1);
  }
  if (frac < 0.5f) {
    return static_cast<int>(floor_i);
  }
  if (floor_i & 1LL) {
    return static_cast<int>(floor_i + 1);
  }
  return static_cast<int>(floor_i);
}

inline uint8_t quantize_ref(float value, float out_scale) {
  if (out_scale == 0.0f) {
    return 0;
  }
  int q = round_half_even(value / out_scale);
  q = std::clamp(q, -128, 127);
  return static_cast<uint8_t>(static_cast<int8_t>(q));
}

struct LnRowReference {
  float scale0 = 0.0f;
  float scale1 = 0.0f;
  std::vector<float> gamma;
  std::vector<float> beta;
  std::vector<float> row_data;
  float mean = 0.0f;
  float var = 0.0f;
  float denom = 0.0f;
  std::array<float, 16> y_prefix{};
  std::vector<uint8_t> out_row_bytes;
};

inline LnRowReference compute_ln_row_reference(
    const std::vector<uint8_t>& input_bytes,
    const std::vector<uint8_t>& gamma_bytes,
    const std::vector<uint8_t>& beta_bytes,
    int rows,
    int cols,
    int row_idx,
    int in_scale_fp16,
    int out_scale_fp16) {
  if (rows <= 0 || cols <= 0 || row_idx < 0 || row_idx >= rows) {
    throw std::runtime_error("invalid LayerNorm replay shape");
  }
  if (input_bytes.size() != static_cast<std::size_t>(rows * cols)) {
    throw std::runtime_error("input_bytes size mismatch");
  }
  if (gamma_bytes.size() != static_cast<std::size_t>(cols * 2) ||
      beta_bytes.size() != static_cast<std::size_t>(cols * 2)) {
    throw std::runtime_error("gamma/beta payload size mismatch");
  }

  LnRowReference ref;
  ref.scale0 = fp16_to_float(static_cast<uint16_t>(in_scale_fp16));
  ref.scale1 = fp16_to_float(static_cast<uint16_t>(out_scale_fp16));
  ref.gamma.resize(static_cast<std::size_t>(cols));
  ref.beta.resize(static_cast<std::size_t>(cols));
  ref.row_data.resize(static_cast<std::size_t>(cols));
  ref.out_row_bytes.resize(static_cast<std::size_t>(cols));

  for (int i = 0; i < cols; ++i) {
    const auto gamma_bits = static_cast<uint16_t>(
        gamma_bytes[static_cast<std::size_t>(i) * 2u] |
        (static_cast<uint16_t>(gamma_bytes[static_cast<std::size_t>(i) * 2u + 1u]) << 8));
    const auto beta_bits = static_cast<uint16_t>(
        beta_bytes[static_cast<std::size_t>(i) * 2u] |
        (static_cast<uint16_t>(beta_bytes[static_cast<std::size_t>(i) * 2u + 1u]) << 8));
    ref.gamma[static_cast<std::size_t>(i)] = fp16_to_float(gamma_bits);
    ref.beta[static_cast<std::size_t>(i)] = fp16_to_float(beta_bits);
  }

  const std::size_t row_base = static_cast<std::size_t>(row_idx) * static_cast<std::size_t>(cols);
  float sum = 0.0f;
  for (int c = 0; c < cols; ++c) {
    const int8_t x_i8 = static_cast<int8_t>(input_bytes[row_base + static_cast<std::size_t>(c)]);
    const float x = static_cast<float>(x_i8) * ref.scale0;
    ref.row_data[static_cast<std::size_t>(c)] = x;
    sum += x;
  }
  ref.mean = sum / static_cast<float>(cols);

  float var_sum = 0.0f;
  for (int c = 0; c < cols; ++c) {
    const float d = ref.row_data[static_cast<std::size_t>(c)] - ref.mean;
    var_sum += d * d;
  }
  ref.var = var_sum / static_cast<float>(cols);
  ref.denom = std::sqrt(ref.var + 1.0e-6f);

  for (int c = 0; c < cols; ++c) {
    const float y =
        ((ref.row_data[static_cast<std::size_t>(c)] - ref.mean) / ref.denom) *
            ref.gamma[static_cast<std::size_t>(c)] +
        ref.beta[static_cast<std::size_t>(c)];
    if (c < 16) {
      ref.y_prefix[static_cast<std::size_t>(c)] = y;
    }
    ref.out_row_bytes[static_cast<std::size_t>(c)] = quantize_ref(y, ref.scale1);
  }

  return ref;
}

}  // namespace lnreplay
