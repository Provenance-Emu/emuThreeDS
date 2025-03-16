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
    // Process 4 stereo samples at a time using NEON
    int i = 0;
    
    // Process blocks of 4 stereo samples (8 values total)
    for (; i + 3 < count; i += 4) {
        // Load 4 stereo samples (8 values) from dst and src
        int16x8_t dst_vec = vld1q_s16(reinterpret_cast<const int16_t*>(&dst[i]));
        int16x8_t src_vec = vld1q_s16(reinterpret_cast<const int16_t*>(&src[i]));
        
        // Convert to 32-bit to prevent overflow during addition
        int32x4_t dst_low = vmovl_s16(vget_low_s16(dst_vec));
        int32x4_t dst_high = vmovl_s16(vget_high_s16(dst_vec));
        int32x4_t src_low = vmovl_s16(vget_low_s16(src_vec));
        int32x4_t src_high = vmovl_s16(vget_high_s16(src_vec));
        
        // Add with saturation in 32-bit domain
        int32x4_t sum_low = vaddq_s32(dst_low, src_low);
        int32x4_t sum_high = vaddq_s32(dst_high, src_high);
        
        // Convert back to 16-bit with saturation
        int16x8_t result = vcombine_s16(vqmovn_s32(sum_low), vqmovn_s32(sum_high));
        
        // Store the result
        vst1q_s16(reinterpret_cast<int16_t*>(&dst[i]), result);
    }
    
    // Handle remaining samples
    for (; i < count; i++) {
        // Direct NEON operation for individual sample
        int16x4_t dst_vec = vld1_s16(reinterpret_cast<const int16_t*>(&dst[i]));
        int16x4_t src_vec = vld1_s16(reinterpret_cast<const int16_t*>(&src[i]));
        
        // Add with saturation
        int16x4_t result = vqadd_s16(dst_vec, src_vec);
        
        // Store the result
        vst1_s16(reinterpret_cast<int16_t*>(&dst[i]), result);
    }
}

// NEON-optimized version for downmixing quadraphonic to stereo
inline void DownmixQuadToStereo_NEON(std::array<s16, 2>* dst, const std::array<s32, 4>* src, float gain, int count) {
    // Process 4 samples at a time using NEON
    int i = 0;
    
    // Use float32 NEON operations for better precision
    float32x4_t gain_vec = vdupq_n_f32(gain);
    
    // Process blocks of 4 samples
    for (; i + 3 < count; i += 4) {
        // Load 4 stereo destination samples (8 values total)
        int16x8_t dst_vec = vld1q_s16(reinterpret_cast<const int16_t*>(&dst[i]));
        
        // Convert destination samples to 32-bit for processing
        int32x4_t dst_left_s32 = vmovl_s16(vget_low_s16(dst_vec));
        int32x4_t dst_right_s32 = vmovl_s16(vget_high_s16(dst_vec));
        
        // Convert to float for better precision
        float32x4_t dst_left_f32 = vcvtq_f32_s32(dst_left_s32);
        float32x4_t dst_right_f32 = vcvtq_f32_s32(dst_right_s32);
        
        // Prepare arrays for quad channel data
        float32_t fl_array[4], fr_array[4], bl_array[4], br_array[4];
        
        // Extract and convert source samples to float
        for (int j = 0; j < 4; j++) {
            fl_array[j] = static_cast<float>(src[i+j][0]);
            fr_array[j] = static_cast<float>(src[i+j][1]);
            bl_array[j] = static_cast<float>(src[i+j][2]);
            br_array[j] = static_cast<float>(src[i+j][3]);
        }
        
        // Load into NEON registers
        float32x4_t front_left = vld1q_f32(fl_array);
        float32x4_t front_right = vld1q_f32(fr_array);
        float32x4_t back_left = vld1q_f32(bl_array);
        float32x4_t back_right = vld1q_f32(br_array);
        
        // Apply gain to all channels
        front_left = vmulq_f32(front_left, gain_vec);
        front_right = vmulq_f32(front_right, gain_vec);
        back_left = vmulq_f32(back_left, gain_vec);
        back_right = vmulq_f32(back_right, gain_vec);
        
        // Mix front and back channels for left and right
        float32x4_t mixed_left = vaddq_f32(front_left, back_left);
        float32x4_t mixed_right = vaddq_f32(front_right, back_right);
        
        // Add to destination
        float32x4_t result_left_f32 = vaddq_f32(dst_left_f32, mixed_left);
        float32x4_t result_right_f32 = vaddq_f32(dst_right_f32, mixed_right);
        
        // Convert back to int32 with rounding
        int32x4_t result_left_s32 = vcvtnq_s32_f32(result_left_f32);
        int32x4_t result_right_s32 = vcvtnq_s32_f32(result_right_f32);
        
        // Saturate to 16-bit
        int16x4_t result_left_s16 = vqmovn_s32(result_left_s32);
        int16x4_t result_right_s16 = vqmovn_s32(result_right_s32);
        
        // Combine and store results
        int16x8_t result = vcombine_s16(result_left_s16, result_right_s16);
        vst1q_s16(reinterpret_cast<int16_t*>(&dst[i]), result);
    }
    
    // Handle remaining samples
    for (; i < count; i++) {
        // Extract quad channels
        float front_left = static_cast<float>(src[i][0]);
        float front_right = static_cast<float>(src[i][1]);
        float back_left = static_cast<float>(src[i][2]);
        float back_right = static_cast<float>(src[i][3]);
        
        // Apply gain and mix front and back channels
        float left = gain * front_left + gain * back_left;
        float right = gain * front_right + gain * back_right;
        
        // Add to destination with saturation
        float dst_left = static_cast<float>(dst[i][0]);
        float dst_right = static_cast<float>(dst[i][1]);
        
        dst_left += left;
        dst_right += right;
        
        // Convert back to s16 with saturation
        dst[i][0] = static_cast<s16>(std::clamp(static_cast<s32>(dst_left), -32768, 32767));
        dst[i][1] = static_cast<s16>(std::clamp(static_cast<s32>(dst_right), -32768, 32767));
    }
}
    

// NEON-optimized version for downmixing quadraphonic to mono
inline void DownmixQuadToMono_NEON(std::array<s16, 2>* dst, const std::array<s32, 4>* src, float gain, int count) {
    // Process 4 samples at a time using NEON
    int i = 0;
    
    // Use float32 NEON operations for better precision
    float32x4_t gain_vec = vdupq_n_f32(gain);
    float32x4_t quarter = vdupq_n_f32(0.25f); // For averaging 4 channels
    
    // Process blocks of 4 samples
    for (; i + 3 < count; i += 4) {
        // Load 4 stereo destination samples (8 values total)
        int16x8_t dst_vec = vld1q_s16(reinterpret_cast<const int16_t*>(&dst[i]));
        
        // Convert destination samples to 32-bit for processing
        int32x4_t dst_left_s32 = vmovl_s16(vget_low_s16(dst_vec));
        int32x4_t dst_right_s32 = vmovl_s16(vget_high_s16(dst_vec));
        
        // Convert to float for better precision
        float32x4_t dst_left_f32 = vcvtq_f32_s32(dst_left_s32);
        float32x4_t dst_right_f32 = vcvtq_f32_s32(dst_right_s32);
        
        // Prepare arrays for quad channel data
        float32_t fl_array[4], fr_array[4], bl_array[4], br_array[4];
        
        // Extract and convert source samples to float
        for (int j = 0; j < 4; j++) {
            fl_array[j] = static_cast<float>(src[i+j][0]);
            fr_array[j] = static_cast<float>(src[i+j][1]);
            bl_array[j] = static_cast<float>(src[i+j][2]);
            br_array[j] = static_cast<float>(src[i+j][3]);
        }
        
        // Load into NEON registers
        float32x4_t front_left = vld1q_f32(fl_array);
        float32x4_t front_right = vld1q_f32(fr_array);
        float32x4_t back_left = vld1q_f32(bl_array);
        float32x4_t back_right = vld1q_f32(br_array);
        
        // Apply gain to all channels
        front_left = vmulq_f32(front_left, gain_vec);
        front_right = vmulq_f32(front_right, gain_vec);
        back_left = vmulq_f32(back_left, gain_vec);
        back_right = vmulq_f32(back_right, gain_vec);
        
        // Sum all channels
        float32x4_t sum = vaddq_f32(front_left, front_right);
        sum = vaddq_f32(sum, back_left);
        sum = vaddq_f32(sum, back_right);
        
        // Average to get mono (divide by 4)
        float32x4_t mono = vmulq_f32(sum, quarter);
        
        // Add mono to both left and right channels
        float32x4_t result_left_f32 = vaddq_f32(dst_left_f32, mono);
        float32x4_t result_right_f32 = vaddq_f32(dst_right_f32, mono);
        
        // Convert back to int32 with rounding
        int32x4_t result_left_s32 = vcvtnq_s32_f32(result_left_f32);
        int32x4_t result_right_s32 = vcvtnq_s32_f32(result_right_f32);
        
        // Saturate to 16-bit
        int16x4_t result_left_s16 = vqmovn_s32(result_left_s32);
        int16x4_t result_right_s16 = vqmovn_s32(result_right_s32);
        
        // Combine and store results
        int16x8_t result = vcombine_s16(result_left_s16, result_right_s16);
        vst1q_s16(reinterpret_cast<int16_t*>(&dst[i]), result);
    }
    
    // Handle remaining samples
    for (; i < count; i++) {
        // Extract quad channels
        float front_left = static_cast<float>(src[i][0]);
        float front_right = static_cast<float>(src[i][1]);
        float back_left = static_cast<float>(src[i][2]);
        float back_right = static_cast<float>(src[i][3]);
        
        // Apply gain and mix all channels to mono
        float mono = gain * (front_left + front_right + back_left + back_right) * 0.25f;
        
        // Add to destination with saturation
        float dst_left = static_cast<float>(dst[i][0]);
        float dst_right = static_cast<float>(dst[i][1]);
        
        dst_left += mono;
        dst_right += mono;
        
        // Convert back to s16 with saturation
        dst[i][0] = static_cast<s16>(std::clamp(static_cast<s32>(dst_left), -32768, 32767));
        dst[i][1] = static_cast<s16>(std::clamp(static_cast<s32>(dst_right), -32768, 32767));
    }
}

// NEON-optimized version for converting final mix samples to output format
inline void ConvertFinalMixSamples_NEON(s16* dst, const s16* src, int count) {
    // Use NEON to process 8 samples at a time
    int i = 0;
    
    // Create a scaling factor to prevent clipping (soft limiter)
    float32x4_t scale_factor = vdupq_n_f32(0.98f);
    
    // Process blocks of 8 samples
    for (; i + 7 < count; i += 8) {
        // Load 8 source samples
        int16x8_t src_vec = vld1q_s16(&src[i]);
        
        // Split into two 4-element vectors and convert to 32-bit integers
        int32x4_t src_low_s32 = vmovl_s16(vget_low_s16(src_vec));
        int32x4_t src_high_s32 = vmovl_s16(vget_high_s16(src_vec));
        
        // Convert to float for better precision
        float32x4_t src_low_f32 = vcvtq_f32_s32(src_low_s32);
        float32x4_t src_high_f32 = vcvtq_f32_s32(src_high_s32);
        
        // Apply soft limiter
        src_low_f32 = vmulq_f32(src_low_f32, scale_factor);
        src_high_f32 = vmulq_f32(src_high_f32, scale_factor);
        
        // Convert back to int32 with proper rounding
        src_low_s32 = vcvtnq_s32_f32(src_low_f32);
        src_high_s32 = vcvtnq_s32_f32(src_high_f32);
        
        // Convert back to int16 with saturation to prevent overflow
        int16x4_t result_low = vqmovn_s32(src_low_s32);
        int16x4_t result_high = vqmovn_s32(src_high_s32);
        
        // Combine results
        int16x8_t result = vcombine_s16(result_low, result_high);
        
        // Store the result
        vst1q_s16(&dst[i], result);
    }
    
    // Handle remaining samples
    for (; i < count; i++) {
        // Apply a slight limiter to prevent clipping
        float sample = static_cast<float>(src[i]) * 0.98f;
        dst[i] = static_cast<s16>(std::clamp(static_cast<s32>(sample), -32768, 32767));
    }
}

} // namespace AudioCore::HLE

#endif // defined(__ARM_NEON) || defined(__aarch64__)
