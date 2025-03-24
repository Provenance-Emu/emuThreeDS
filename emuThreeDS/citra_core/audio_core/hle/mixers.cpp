// Copyright 2016 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <cstddef>
#include "audio_core/hle/mixers.h"
#include "common/assert.h"
#include "common/logging/log.h"


#if (defined(__ARM_NEON) || defined(__aarch64__))
#define USE_NEON 0

#include "audio_core/hle/mixer_neon.h"
#endif

namespace AudioCore::HLE {

void Mixers::Reset() {
    current_frame.fill({});
    state = {};
}

DspStatus Mixers::Tick(DspConfiguration& config, const IntermediateMixSamples& read_samples,
                       IntermediateMixSamples& write_samples,
                       const std::array<QuadFrame32, 3>& input) {
    ParseConfig(config);

    AuxReturn(read_samples);
    AuxSend(write_samples, input);

    MixCurrentFrame();

    return GetCurrentStatus();
}

void Mixers::ParseConfig(DspConfiguration& config) {
    if (!config.dirty_raw) {
        return;
    }

    if (config.mixer1_enabled_dirty) {
        config.mixer1_enabled_dirty.Assign(0);
        state.mixer1_enabled = config.mixer1_enabled != 0;
        LOG_TRACE(Audio_DSP, "mixers mixer1_enabled = {}", config.mixer1_enabled);
    }

    if (config.mixer2_enabled_dirty) {
        config.mixer2_enabled_dirty.Assign(0);
        state.mixer2_enabled = config.mixer2_enabled != 0;
        LOG_TRACE(Audio_DSP, "mixers mixer2_enabled = {}", config.mixer2_enabled);
    }

    if (config.volume_0_dirty) {
        config.volume_0_dirty.Assign(0);
        state.intermediate_mixer_volume[0] = config.volume[0];
        LOG_TRACE(Audio_DSP, "mixers volume[0] = {}", config.volume[0]);
    }

    if (config.volume_1_dirty) {
        config.volume_1_dirty.Assign(0);
        state.intermediate_mixer_volume[1] = config.volume[1];
        LOG_TRACE(Audio_DSP, "mixers volume[1] = {}", config.volume[1]);
    }

    if (config.volume_2_dirty) {
        config.volume_2_dirty.Assign(0);
        state.intermediate_mixer_volume[2] = config.volume[2];
        LOG_TRACE(Audio_DSP, "mixers volume[2] = {}", config.volume[2]);
    }

    if (config.output_format_dirty) {
        config.output_format_dirty.Assign(0);
        state.output_format = config.output_format;
        LOG_TRACE(Audio_DSP, "mixers output_format = {}",
                  static_cast<std::size_t>(config.output_format));
    }

    if (config.headphones_connected_dirty) {
        config.headphones_connected_dirty.Assign(0);
        // Do nothing. (Note: Whether headphones are connected does affect coefficients used for
        // surround sound.)
        LOG_TRACE(Audio_DSP, "mixers headphones_connected={}", config.headphones_connected);
    }

    if (config.dirty_raw) {
        LOG_DEBUG(Audio_DSP, "mixers remaining_dirty={:x}", config.dirty_raw);
    }

    config.dirty_raw = 0;
}

static s16 ClampToS16(s32 value) {
    return static_cast<s16>(std::clamp(value, -32768, 32767));
}

static std::array<s16, 2> AddAndClampToS16(const std::array<s16, 2>& a,
                                           const std::array<s16, 2>& b) {
    return {ClampToS16(static_cast<s32>(a[0]) + static_cast<s32>(b[0])),
            ClampToS16(static_cast<s32>(a[1]) + static_cast<s32>(b[1]))};
}

// Optimized version for adding and clamping multiple stereo samples
static void AddAndClampToS16Batch(std::array<s16, 2>* dst, const std::array<s16, 2>* src, int count) {
#if USE_NEON
    // Use NEON-optimized version if available
    AddAndClampToS16_NEON(dst, src, count);
#else
    // Fallback for non-NEON platforms
    for (int i = 0; i < count; i++) {
        dst[i] = AddAndClampToS16(dst[i], src[i]);
    }
#endif
}

void Mixers::DownmixAndMixIntoCurrentFrame(float gain, const QuadFrame32& samples) {
    // TODO(merry): Limiter. (Currently we're performing final mixing assuming a disabled limiter.)

#if USE_NEON
    // Use NEON-optimized versions for ARM platforms
    switch (state.output_format) {
    case OutputFormat::Mono:
        DownmixQuadToMono_NEON(current_frame.data(), samples.data(), gain, samples_per_frame);
        return;

    case OutputFormat::Surround:
        // TODO(merry): Implement surround sound.
        // fallthrough

    case OutputFormat::Stereo:
        DownmixQuadToStereo_NEON(current_frame.data(), samples.data(), gain, samples_per_frame);
        return;
    }
#else
    // Original implementation for non-ARM platforms
    switch (state.output_format) {
    case OutputFormat::Mono:
        std::transform(
            current_frame.begin(), current_frame.end(), samples.begin(), current_frame.begin(),
            [gain](const std::array<s16, 2>& accumulator,
                   const std::array<s32, 4>& sample) -> std::array<s16, 2> {
                // Downmix to mono
                s16 mono = ClampToS16(static_cast<s32>(
                    (gain * sample[0] + gain * sample[1] + gain * sample[2] + gain * sample[3]) /
                    2));
                // Mix into current frame
                return AddAndClampToS16(accumulator, {mono, mono});
            });
        return;

    case OutputFormat::Surround:
        // TODO(merry): Implement surround sound.
        // fallthrough

    case OutputFormat::Stereo:
        std::transform(
            current_frame.begin(), current_frame.end(), samples.begin(), current_frame.begin(),
            [gain](const std::array<s16, 2>& accumulator,
                   const std::array<s32, 4>& sample) -> std::array<s16, 2> {
                // Downmix to stereo
                s16 left = ClampToS16(static_cast<s32>(gain * sample[0] + gain * sample[2]));
                s16 right = ClampToS16(static_cast<s32>(gain * sample[1] + gain * sample[3]));
                // Mix into current frame
                return AddAndClampToS16(accumulator, {left, right});
            });
        return;
    }
#endif

    UNREACHABLE_MSG("Invalid output_format {}", static_cast<std::size_t>(state.output_format));
}

void Mixers::AuxReturn(const IntermediateMixSamples& read_samples) {
    // NOTE: read_samples.mix{1,2}.pcm32 annoyingly have their dimensions in reverse order to
    // QuadFrame32.

#if USE_NEON
    if (state.mixer1_enabled) {
        // Process 4 samples at a time using NEON
        for (std::size_t sample = 0; sample < samples_per_frame; sample += 4) {
            for (std::size_t channel = 0; channel < 4; channel++) {
                // Load 4 samples at once
                if (sample + 4 <= samples_per_frame) {
                    int32x4_t samples = vld1q_s32(&read_samples.mix1.pcm32[channel][sample]);
                    
                    // Store samples with transposed indices - unrolled loop with constant indices
                    state.intermediate_mix_buffer[1][sample][channel] = vgetq_lane_s32(samples, 0);
                    state.intermediate_mix_buffer[1][sample + 1][channel] = vgetq_lane_s32(samples, 1);
                    state.intermediate_mix_buffer[1][sample + 2][channel] = vgetq_lane_s32(samples, 2);
                    state.intermediate_mix_buffer[1][sample + 3][channel] = vgetq_lane_s32(samples, 3);
                } else {
                    // Handle remaining samples
                    for (std::size_t i = 0; i < samples_per_frame - sample; i++) {
                        state.intermediate_mix_buffer[1][sample + i][channel] = read_samples.mix1.pcm32[channel][sample + i];
                    }
                }
            }
        }
    }

    if (state.mixer2_enabled) {
        // Process 4 samples at a time using NEON
        for (std::size_t sample = 0; sample < samples_per_frame; sample += 4) {
            for (std::size_t channel = 0; channel < 4; channel++) {
                // Load 4 samples at once
                if (sample + 4 <= samples_per_frame) {
                    int32x4_t samples = vld1q_s32(&read_samples.mix2.pcm32[channel][sample]);
                    
                    // Store samples with transposed indices - unrolled loop with constant indices
                    state.intermediate_mix_buffer[2][sample][channel] = vgetq_lane_s32(samples, 0);
                    state.intermediate_mix_buffer[2][sample + 1][channel] = vgetq_lane_s32(samples, 1);
                    state.intermediate_mix_buffer[2][sample + 2][channel] = vgetq_lane_s32(samples, 2);
                    state.intermediate_mix_buffer[2][sample + 3][channel] = vgetq_lane_s32(samples, 3);
                } else {
                    // Handle remaining samples
                    for (std::size_t i = 0; i < samples_per_frame - sample; i++) {
                        state.intermediate_mix_buffer[2][sample + i][channel] = read_samples.mix2.pcm32[channel][sample + i];
                    }
                }
            }
        }
    }
#else
    if (state.mixer1_enabled) {
        for (std::size_t sample = 0; sample < samples_per_frame; sample++) {
            for (std::size_t channel = 0; channel < 4; channel++) {
                state.intermediate_mix_buffer[1][sample][channel] =
                    read_samples.mix1.pcm32[channel][sample];
            }
        }
    }

    if (state.mixer2_enabled) {
        for (std::size_t sample = 0; sample < samples_per_frame; sample++) {
            for (std::size_t channel = 0; channel < 4; channel++) {
                state.intermediate_mix_buffer[2][sample][channel] =
                    read_samples.mix2.pcm32[channel][sample];
            }
        }
    }
#endif
}

void Mixers::AuxSend(IntermediateMixSamples& write_samples,
                     const std::array<QuadFrame32, 3>& input) {
    // NOTE: read_samples.mix{1,2}.pcm32 annoyingly have their dimensions in reverse order to
    // QuadFrame32.

    state.intermediate_mix_buffer[0] = input[0];

#if USE_NEON
    if (state.mixer1_enabled) {
        // Process 4 samples at a time using NEON
        for (std::size_t sample = 0; sample < samples_per_frame; sample += 4) {
            for (std::size_t channel = 0; channel < 4; channel++) {
                if (sample + 4 <= samples_per_frame) {
                    // Prepare to load 4 samples with transposed indices - unrolled with constant indices
                    int32x4_t samples;
                    samples = vsetq_lane_s32(input[1][sample][channel], vdupq_n_s32(0), 0);
                    samples = vsetq_lane_s32(input[1][sample + 1][channel], samples, 1);
                    samples = vsetq_lane_s32(input[1][sample + 2][channel], samples, 2);
                    samples = vsetq_lane_s32(input[1][sample + 3][channel], samples, 3);
                    
                    // Store 4 samples at once
                    vst1q_s32(&write_samples.mix1.pcm32[channel][sample], samples);
                } else {
                    // Handle remaining samples
                    for (std::size_t i = 0; i < samples_per_frame - sample; i++) {
                        write_samples.mix1.pcm32[channel][sample + i] = input[1][sample + i][channel];
                    }
                }
            }
        }
    } else {
        state.intermediate_mix_buffer[1] = input[1];
    }

    if (state.mixer2_enabled) {
        // Process 4 samples at a time using NEON
        for (std::size_t sample = 0; sample < samples_per_frame; sample += 4) {
            for (std::size_t channel = 0; channel < 4; channel++) {
                if (sample + 4 <= samples_per_frame) {
                    // Prepare to load 4 samples with transposed indices - unrolled with constant indices
                    int32x4_t samples;
                    samples = vsetq_lane_s32(input[2][sample][channel], vdupq_n_s32(0), 0);
                    samples = vsetq_lane_s32(input[2][sample + 1][channel], samples, 1);
                    samples = vsetq_lane_s32(input[2][sample + 2][channel], samples, 2);
                    samples = vsetq_lane_s32(input[2][sample + 3][channel], samples, 3);
                    
                    // Store 4 samples at once
                    vst1q_s32(&write_samples.mix2.pcm32[channel][sample], samples);
                } else {
                    // Handle remaining samples
                    for (std::size_t i = 0; i < samples_per_frame - sample; i++) {
                        write_samples.mix2.pcm32[channel][sample + i] = input[2][sample + i][channel];
                    }
                }
            }
        }
    } else {
        state.intermediate_mix_buffer[2] = input[2];
    }
#else
    if (state.mixer1_enabled) {
        for (std::size_t sample = 0; sample < samples_per_frame; sample++) {
            for (std::size_t channel = 0; channel < 4; channel++) {
                write_samples.mix1.pcm32[channel][sample] = input[1][sample][channel];
            }
        }
    } else {
        state.intermediate_mix_buffer[1] = input[1];
    }

    if (state.mixer2_enabled) {
        for (std::size_t sample = 0; sample < samples_per_frame; sample++) {
            for (std::size_t channel = 0; channel < 4; channel++) {
                write_samples.mix2.pcm32[channel][sample] = input[2][sample][channel];
            }
        }
    } else {
        state.intermediate_mix_buffer[2] = input[2];
    }
#endif
}

void Mixers::MixCurrentFrame() {
    current_frame.fill({});

    for (std::size_t mix = 0; mix < 3; mix++) {
        DownmixAndMixIntoCurrentFrame(state.intermediate_mixer_volume[mix],
                                      state.intermediate_mix_buffer[mix]);
    }

    // TODO(merry): Compressor. (We currently assume a disabled compressor.)
}

DspStatus Mixers::GetCurrentStatus() const {
    DspStatus status;
    status.unknown = 0;
    status.dropped_frames = 0;
    return status;
}

} // namespace AudioCore::HLE
