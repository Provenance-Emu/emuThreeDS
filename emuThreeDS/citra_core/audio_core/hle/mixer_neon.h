// Copyright 2025 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <algorithm> // For std::clamp
#include "audio_core/audio_types.h"
#include "common/common_types.h"

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>

namespace AudioCore::HLE {

// NEON-optimized version of ClampToS16
inline int16x8_t ClampToS16_NEON(int32x4_t low, int32x4_t high) {
    // Saturate to 16-bit signed integer range
    int16x4_t low_s16 = vqmovn_s32(low);
    int16x4_t high_s16 = vqmovn_s32(high);
    
    // Combine the two vectors
    return vcombine_s16(low_s16, high_s16);
}

// NEON-optimized version for adding and clamping stereo samples
inline void AddAndClampToS16_NEON(std::array<s16, 2>* dst, const std::array<s16, 2>* src, int count) {
    // Process one stereo sample at a time to avoid alignment issues
    for (int i = 0; i < count; i++) {
        // Load stereo sample (left, right) from dst and src
        s16 dst_left = dst[i][0];
        s16 dst_right = dst[i][1];
        s16 src_left = src[i][0];
        s16 src_right = src[i][1];
        
        // Apply a slight attenuation to prevent potential clipping
        float dst_left_f32 = static_cast<float>(dst_left) * 0.99f;
        float dst_right_f32 = static_cast<float>(dst_right) * 0.99f;
        float src_left_f32 = static_cast<float>(src_left) * 0.99f;
        float src_right_f32 = static_cast<float>(src_right) * 0.99f;
        
        // Add in floating point domain
        float sum_left = dst_left_f32 + src_left_f32;
        float sum_right = dst_right_f32 + src_right_f32;
        
        // Convert back to s16 with saturation
        dst[i][0] = static_cast<s16>(std::clamp(static_cast<s32>(sum_left), -32768, 32767));
        dst[i][1] = static_cast<s16>(std::clamp(static_cast<s32>(sum_right), -32768, 32767));
    }
}

// NEON-optimized version for downmixing quadraphonic to stereo
inline void DownmixQuadToStereo_NEON(std::array<s16, 2>* dst, const std::array<s32, 4>* src, float gain, int count) {
    // Process one sample at a time to avoid alignment issues
    for (int i = 0; i < count; i++) {
        // Extract quad channels
        s32 front_left = src[i][0];
        s32 front_right = src[i][1];
        s32 back_left = src[i][2];
        s32 back_right = src[i][3];
        
        // Apply gain and mix front and back channels
        float left = gain * static_cast<float>(front_left) + gain * static_cast<float>(back_left);
        float right = gain * static_cast<float>(front_right) + gain * static_cast<float>(back_right);
        
        // Convert to s16 with saturation
        s16 left_s16 = static_cast<s16>(std::clamp(static_cast<s32>(left), -32768, 32767));
        s16 right_s16 = static_cast<s16>(std::clamp(static_cast<s32>(right), -32768, 32767));
        
        // Add to destination with saturation
        dst[i][0] = static_cast<s16>(std::clamp(static_cast<s32>(dst[i][0]) + static_cast<s32>(left_s16), -32768, 32767));
        dst[i][1] = static_cast<s16>(std::clamp(static_cast<s32>(dst[i][1]) + static_cast<s32>(right_s16), -32768, 32767));
    }
}
    

// NEON-optimized version for downmixing quadraphonic to mono
inline void DownmixQuadToMono_NEON(std::array<s16, 2>* dst, const std::array<s32, 4>* src, float gain, int count) {
    // Process one sample at a time to avoid alignment issues
    for (int i = 0; i < count; i++) {
        // Extract quad channels
        s32 front_left = src[i][0];
        s32 front_right = src[i][1];
        s32 back_left = src[i][2];
        s32 back_right = src[i][3];
        
        // Apply gain and average all channels to mono
        float mono = gain * (static_cast<float>(front_left) + 
                            static_cast<float>(front_right) + 
                            static_cast<float>(back_left) + 
                            static_cast<float>(back_right)) * 0.25f;
        
        // Convert to s16 with saturation
        s16 mono_s16 = static_cast<s16>(std::clamp(static_cast<s32>(mono), -32768, 32767));
        
        // Add to destination with saturation (both left and right channels)
        dst[i][0] = static_cast<s16>(std::clamp(static_cast<s32>(dst[i][0]) + static_cast<s32>(mono_s16), -32768, 32767));
        dst[i][1] = static_cast<s16>(std::clamp(static_cast<s32>(dst[i][1]) + static_cast<s32>(mono_s16), -32768, 32767));
    }
}

// NEON-optimized version for converting final mix samples to output format
inline void ConvertFinalMixSamples_NEON(s16* dst, const s16* src, int count) {
    // Process samples in smaller batches to avoid alignment issues
    for (int i = 0; i < count; i++) {
        // Apply a slight limiter to prevent clipping
        // This can help reduce buzzing caused by digital clipping
        float sample = static_cast<float>(src[i]) * 0.98f;
        dst[i] = static_cast<s16>(std::clamp(static_cast<s32>(sample), -32768, 32767));
    }
}

} // namespace AudioCore::HLE

#endif // defined(__ARM_NEON) || defined(__aarch64__)
