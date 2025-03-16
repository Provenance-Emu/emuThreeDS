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
    // Process 4 samples at a time using NEON
    int i = 0;
    const int32_t gain_fixed = static_cast<int32_t>(gain * 16384.0f); // Fixed-point gain
    
    // Process blocks of 4 samples
    for (; i + 3 < count; i += 4) {
        // Load 4 stereo destination samples (8 values total)
        int16x8_t dst_vec = vld1q_s16(reinterpret_cast<const int16_t*>(&dst[i]));
        int32x4_t dst_left = vmovl_s16(vget_low_s16(dst_vec));
        int32x4_t dst_right = vmovl_s16(vget_high_s16(dst_vec));
        
        // Process 4 samples at once
        int32x4_t front_left, front_right, back_left, back_right;
        int32x4_t mixed_left, mixed_right;
        
        // Load quad channels for 4 samples with better memory access pattern
        // Instead of using vsetq_lane_s32 in a loop (which requires constant indices),
        // we'll manually load the values into arrays and then create vectors from them
        int32_t fl_array[4], fr_array[4], bl_array[4], br_array[4];
        
        // Load values into temporary arrays
        for (int j = 0; j < 4; j++) {
            fl_array[j] = src[i+j][0];
            fr_array[j] = src[i+j][1];
            bl_array[j] = src[i+j][2];
            br_array[j] = src[i+j][3];
        }
        
        // Create NEON vectors from the arrays
        front_left = vld1q_s32(fl_array);
        front_right = vld1q_s32(fr_array);
        back_left = vld1q_s32(bl_array);
        back_right = vld1q_s32(br_array);
        
        // Apply fixed-point gain and mix front and back channels
        // Use NEON multiply-accumulate operations for better performance
        mixed_left = vmulq_n_s32(front_left, gain_fixed);
        mixed_left = vmlaq_n_s32(mixed_left, back_left, gain_fixed);
        
        mixed_right = vmulq_n_s32(front_right, gain_fixed);
        mixed_right = vmlaq_n_s32(mixed_right, back_right, gain_fixed);
        
        // Shift right to account for fixed-point multiplication
        mixed_left = vshrq_n_s32(mixed_left, 14);
        mixed_right = vshrq_n_s32(mixed_right, 14);
        
        // Convert to s16 with saturation
        int16x4_t left_s16 = vqmovn_s32(mixed_left);
        int16x4_t right_s16 = vqmovn_s32(mixed_right);
        
        // Add to destination with saturation
        int16x4_t result_left = vqadd_s16(vqmovn_s32(dst_left), left_s16);
        int16x4_t result_right = vqadd_s16(vqmovn_s32(dst_right), right_s16);
        
        // Combine and store results
        int16x8_t result = vcombine_s16(result_left, result_right);
        vst1q_s16(reinterpret_cast<int16_t*>(&dst[i]), result);
    }
    
    // Handle remaining samples
    for (; i < count; i++) {
        // Extract quad channels
        s32 front_left = src[i][0];
        s32 front_right = src[i][1];
        s32 back_left = src[i][2];
        s32 back_right = src[i][3];
        
        // Apply gain and mix front and back channels (fixed-point)
        s32 left = (gain_fixed * front_left + gain_fixed * back_left) >> 14;
        s32 right = (gain_fixed * front_right + gain_fixed * back_right) >> 14;
        
        // Convert to s16 with saturation
        s16 left_s16 = static_cast<s16>(std::clamp(left, -32768, 32767));
        s16 right_s16 = static_cast<s16>(std::clamp(right, -32768, 32767));
        
        // Add to destination with saturation
        dst[i][0] = static_cast<s16>(std::clamp(static_cast<s32>(dst[i][0]) + static_cast<s32>(left_s16), -32768, 32767));
        dst[i][1] = static_cast<s16>(std::clamp(static_cast<s32>(dst[i][1]) + static_cast<s32>(right_s16), -32768, 32767));
    }
}
    

// NEON-optimized version for downmixing quadraphonic to mono
inline void DownmixQuadToMono_NEON(std::array<s16, 2>* dst, const std::array<s32, 4>* src, float gain, int count) {
    // Process 4 samples at a time using NEON
    int i = 0;
    const int32_t gain_fixed = static_cast<int32_t>(gain * 4096.0f); // Fixed-point gain
    
    // Process blocks of 4 samples
    for (; i + 3 < count; i += 4) {
        // Load 4 stereo destination samples (8 values total)
        int16x8_t dst_vec = vld1q_s16(reinterpret_cast<const int16_t*>(&dst[i]));
        int32x4_t dst_left = vmovl_s16(vget_low_s16(dst_vec));
        int32x4_t dst_right = vmovl_s16(vget_high_s16(dst_vec));
        
        // Process 4 samples at once
        int32x4_t front_left, front_right, back_left, back_right;
        int32x4_t mono_samples;
        
        // Load quad channels for 4 samples with better memory access pattern
        // Instead of using vsetq_lane_s32 in a loop (which requires constant indices),
        // we'll manually load the values into arrays and then create vectors from them
        int32_t fl_array[4], fr_array[4], bl_array[4], br_array[4];
        
        // Load values into temporary arrays
        for (int j = 0; j < 4; j++) {
            fl_array[j] = src[i+j][0];
            fr_array[j] = src[i+j][1];
            bl_array[j] = src[i+j][2];
            br_array[j] = src[i+j][3];
        }
        
        // Create NEON vectors from the arrays
        front_left = vld1q_s32(fl_array);
        front_right = vld1q_s32(fr_array);
        back_left = vld1q_s32(bl_array);
        back_right = vld1q_s32(br_array);
        
        // Sum all channels
        mono_samples = vaddq_s32(front_left, front_right);
        mono_samples = vaddq_s32(mono_samples, back_left);
        mono_samples = vaddq_s32(mono_samples, back_right);
        
        // Apply gain and divide by 4 (right shift by 2) for averaging
        mono_samples = vmulq_n_s32(mono_samples, gain_fixed);
        mono_samples = vshrq_n_s32(mono_samples, 14); // 12 bits for gain + 2 bits for divide by 4
        
        // Convert to s16 with saturation
        int16x4_t mono_s16 = vqmovn_s32(mono_samples);
        
        // Add to destination with saturation (both left and right channels)
        int16x4_t result_left = vqadd_s16(vqmovn_s32(dst_left), mono_s16);
        int16x4_t result_right = vqadd_s16(vqmovn_s32(dst_right), mono_s16);
        
        // Combine and store results
        int16x8_t result = vcombine_s16(result_left, result_right);
        vst1q_s16(reinterpret_cast<int16_t*>(&dst[i]), result);
    }
    
    // Handle remaining samples
    for (; i < count; i++) {
        // Extract quad channels
        s32 front_left = src[i][0];
        s32 front_right = src[i][1];
        s32 back_left = src[i][2];
        s32 back_right = src[i][3];
        
        // Apply gain and average all channels to mono (fixed-point)
        s32 mono = (gain_fixed * (front_left + front_right + back_left + back_right)) >> 14;
        
        // Convert to s16 with saturation
        s16 mono_s16 = static_cast<s16>(std::clamp(mono, -32768, 32767));
        
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
