// Copyright 2025 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
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
    // Process 4 stereo samples (8 values) at a time
    for (int i = 0; i < count; i += 4) {
        // Load 4 stereo samples (8 values) from dst and src
        int16x8_t dst_vec = vld1q_s16(reinterpret_cast<const int16_t*>(&dst[i]));
        int16x8_t src_vec = vld1q_s16(reinterpret_cast<const int16_t*>(&src[i]));
        
        // Add with saturation
        int16x8_t result = vqaddq_s16(dst_vec, src_vec);
        
        // Store the result back to dst
        vst1q_s16(reinterpret_cast<int16_t*>(&dst[i]), result);
    }
    
    // Handle remaining samples
    int remaining = count & 3; // count % 4
    if (remaining > 0) {
        for (int i = count - remaining; i < count; i++) {
            dst[i][0] = static_cast<s16>(std::clamp(static_cast<s32>(dst[i][0]) + static_cast<s32>(src[i][0]), -32768, 32767));
            dst[i][1] = static_cast<s16>(std::clamp(static_cast<s32>(dst[i][1]) + static_cast<s32>(src[i][1]), -32768, 32767));
        }
    }
}

// NEON-optimized version for downmixing quadraphonic to stereo
inline void DownmixQuadToStereo_NEON(std::array<s16, 2>* dst, const std::array<s32, 4>* src, float gain, int count) {
    // Create a vector of the gain value
    float32x4_t gain_vec = vdupq_n_f32(gain);
    
    // Process 4 samples at a time
    for (int i = 0; i < count; i += 4) {
        // Handle remaining count
        if (i + 4 > count) {
            // Process remaining samples individually
            for (int j = i; j < count; j++) {
                s16 left = static_cast<s16>(std::clamp(static_cast<s32>(gain * src[j][0] + gain * src[j][2]), -32768, 32767));
                s16 right = static_cast<s16>(std::clamp(static_cast<s32>(gain * src[j][1] + gain * src[j][3]), -32768, 32767));
                
                dst[j][0] = static_cast<s16>(std::clamp(static_cast<s32>(dst[j][0]) + static_cast<s32>(left), -32768, 32767));
                dst[j][1] = static_cast<s16>(std::clamp(static_cast<s32>(dst[j][1]) + static_cast<s32>(right), -32768, 32767));
            }
            break;
        }
        
        // Process 4 samples with unrolled loop
        // Sample 0
        int32x4_t quad_sample0 = vld1q_s32(reinterpret_cast<const int32_t*>(&src[i]));
        float32x4_t float_sample0 = vcvtq_f32_s32(quad_sample0);
        float32x4_t gained_sample0 = vmulq_f32(float_sample0, gain_vec);
        float32x2_t left_vec0 = vadd_f32(vget_low_f32(gained_sample0), vget_high_f32(gained_sample0));
        int32x2_t left_s32_0 = vcvt_s32_f32(left_vec0);
        int16x4_t stereo_s16_0 = vqmovn_s32(vcombine_s32(left_s32_0, left_s32_0));
        
        // Sample 1
        int32x4_t quad_sample1 = vld1q_s32(reinterpret_cast<const int32_t*>(&src[i+1]));
        float32x4_t float_sample1 = vcvtq_f32_s32(quad_sample1);
        float32x4_t gained_sample1 = vmulq_f32(float_sample1, gain_vec);
        float32x2_t left_vec1 = vadd_f32(vget_low_f32(gained_sample1), vget_high_f32(gained_sample1));
        int32x2_t left_s32_1 = vcvt_s32_f32(left_vec1);
        int16x4_t stereo_s16_1 = vqmovn_s32(vcombine_s32(left_s32_1, left_s32_1));
        
        // Sample 2
        int32x4_t quad_sample2 = vld1q_s32(reinterpret_cast<const int32_t*>(&src[i+2]));
        float32x4_t float_sample2 = vcvtq_f32_s32(quad_sample2);
        float32x4_t gained_sample2 = vmulq_f32(float_sample2, gain_vec);
        float32x2_t left_vec2 = vadd_f32(vget_low_f32(gained_sample2), vget_high_f32(gained_sample2));
        int32x2_t left_s32_2 = vcvt_s32_f32(left_vec2);
        int16x4_t stereo_s16_2 = vqmovn_s32(vcombine_s32(left_s32_2, left_s32_2));
        
        // Sample 3
        int32x4_t quad_sample3 = vld1q_s32(reinterpret_cast<const int32_t*>(&src[i+3]));
        float32x4_t float_sample3 = vcvtq_f32_s32(quad_sample3);
        float32x4_t gained_sample3 = vmulq_f32(float_sample3, gain_vec);
        float32x2_t left_vec3 = vadd_f32(vget_low_f32(gained_sample3), vget_high_f32(gained_sample3));
        int32x2_t left_s32_3 = vcvt_s32_f32(left_vec3);
        int16x4_t stereo_s16_3 = vqmovn_s32(vcombine_s32(left_s32_3, left_s32_3));
        
        // Store results directly to destination with saturation
        int16x8_t dst_vec0 = vld1q_s16(reinterpret_cast<const int16_t*>(&dst[i]));
        int16x8_t dst_vec1 = vld1q_s16(reinterpret_cast<const int16_t*>(&dst[i+2]));
        
        // Create result vectors - manually create vectors with the correct values
        int16x4_t result_low = vdup_n_s16(0);
        result_low = vset_lane_s16(vget_lane_s16(stereo_s16_0, 0), result_low, 0);
        result_low = vset_lane_s16(vget_lane_s16(stereo_s16_0, 1), result_low, 1);
        result_low = vset_lane_s16(vget_lane_s16(stereo_s16_1, 0), result_low, 2);
        result_low = vset_lane_s16(vget_lane_s16(stereo_s16_1, 1), result_low, 3);
        
        int16x4_t result_high = vdup_n_s16(0);
        result_high = vset_lane_s16(vget_lane_s16(stereo_s16_2, 0), result_high, 0);
        result_high = vset_lane_s16(vget_lane_s16(stereo_s16_2, 1), result_high, 1);
        result_high = vset_lane_s16(vget_lane_s16(stereo_s16_3, 0), result_high, 2);
        result_high = vset_lane_s16(vget_lane_s16(stereo_s16_3, 1), result_high, 3);
        
        // Add and saturate
        int16x4_t final_low = vqadd_s16(vget_low_s16(dst_vec0), result_low);
        int16x4_t final_high = vqadd_s16(vget_high_s16(dst_vec1), result_high);
        
        // Store results
        vst1_s16(reinterpret_cast<int16_t*>(&dst[i]), final_low);
        vst1_s16(reinterpret_cast<int16_t*>(&dst[i+2]), final_high);
    }
    
    // Handle remaining samples
    int remaining = count & 3; // count % 4
    if (remaining > 0) {
        for (int i = count - remaining; i < count; i++) {
            s16 left = static_cast<s16>(std::clamp(static_cast<s32>(gain * src[i][0] + gain * src[i][2]), -32768, 32767));
            s16 right = static_cast<s16>(std::clamp(static_cast<s32>(gain * src[i][1] + gain * src[i][3]), -32768, 32767));
            
            dst[i][0] = static_cast<s16>(std::clamp(static_cast<s32>(dst[i][0]) + static_cast<s32>(left), -32768, 32767));
            dst[i][1] = static_cast<s16>(std::clamp(static_cast<s32>(dst[i][1]) + static_cast<s32>(right), -32768, 32767));
        }
    }
}

// NEON-optimized version for downmixing quadraphonic to mono
inline void DownmixQuadToMono_NEON(std::array<s16, 2>* dst, const std::array<s32, 4>* src, float gain, int count) {
    // Create a vector of the gain value
    float32x4_t gain_vec = vdupq_n_f32(gain);
    float32x2_t half_vec = vdup_n_f32(0.5f); // For averaging the 4 channels to mono
    
    // Process 4 samples at a time
    for (int i = 0; i < count; i += 4) {
        // Handle remaining count
        if (i + 4 > count) {
            // Process remaining samples individually
            for (int j = i; j < count; j++) {
                s16 mono = static_cast<s16>(std::clamp(static_cast<s32>((gain * src[j][0] + gain * src[j][1] + 
                                                                        gain * src[j][2] + gain * src[j][3]) / 2), 
                                                    -32768, 32767));
                
                dst[j][0] = static_cast<s16>(std::clamp(static_cast<s32>(dst[j][0]) + static_cast<s32>(mono), -32768, 32767));
                dst[j][1] = static_cast<s16>(std::clamp(static_cast<s32>(dst[j][1]) + static_cast<s32>(mono), -32768, 32767));
            }
            break;
        }
        
        // Process 4 samples with unrolled loop
        // Sample 0
        int32x4_t quad_sample0 = vld1q_s32(reinterpret_cast<const int32_t*>(&src[i]));
        float32x4_t float_sample0 = vcvtq_f32_s32(quad_sample0);
        float32x4_t gained_sample0 = vmulq_f32(float_sample0, gain_vec);
        float32x2_t sum_low0 = vadd_f32(vget_low_f32(gained_sample0), vget_high_f32(gained_sample0));
        float32x2_t sum_fold0 = vpadd_f32(sum_low0, sum_low0);
        float32x2_t mono_f32_0 = vmul_f32(sum_fold0, half_vec);
        int32x2_t mono_s32_0 = vcvt_s32_f32(mono_f32_0);
        s16 mono0 = vget_lane_s16(vqmovn_s32(vcombine_s32(mono_s32_0, vdup_n_s32(0))), 0);
        
        // Sample 1
        int32x4_t quad_sample1 = vld1q_s32(reinterpret_cast<const int32_t*>(&src[i+1]));
        float32x4_t float_sample1 = vcvtq_f32_s32(quad_sample1);
        float32x4_t gained_sample1 = vmulq_f32(float_sample1, gain_vec);
        float32x2_t sum_low1 = vadd_f32(vget_low_f32(gained_sample1), vget_high_f32(gained_sample1));
        float32x2_t sum_fold1 = vpadd_f32(sum_low1, sum_low1);
        float32x2_t mono_f32_1 = vmul_f32(sum_fold1, half_vec);
        int32x2_t mono_s32_1 = vcvt_s32_f32(mono_f32_1);
        s16 mono1 = vget_lane_s16(vqmovn_s32(vcombine_s32(mono_s32_1, vdup_n_s32(0))), 0);
        
        // Sample 2
        int32x4_t quad_sample2 = vld1q_s32(reinterpret_cast<const int32_t*>(&src[i+2]));
        float32x4_t float_sample2 = vcvtq_f32_s32(quad_sample2);
        float32x4_t gained_sample2 = vmulq_f32(float_sample2, gain_vec);
        float32x2_t sum_low2 = vadd_f32(vget_low_f32(gained_sample2), vget_high_f32(gained_sample2));
        float32x2_t sum_fold2 = vpadd_f32(sum_low2, sum_low2);
        float32x2_t mono_f32_2 = vmul_f32(sum_fold2, half_vec);
        int32x2_t mono_s32_2 = vcvt_s32_f32(mono_f32_2);
        s16 mono2 = vget_lane_s16(vqmovn_s32(vcombine_s32(mono_s32_2, vdup_n_s32(0))), 0);
        
        // Sample 3
        int32x4_t quad_sample3 = vld1q_s32(reinterpret_cast<const int32_t*>(&src[i+3]));
        float32x4_t float_sample3 = vcvtq_f32_s32(quad_sample3);
        float32x4_t gained_sample3 = vmulq_f32(float_sample3, gain_vec);
        float32x2_t sum_low3 = vadd_f32(vget_low_f32(gained_sample3), vget_high_f32(gained_sample3));
        float32x2_t sum_fold3 = vpadd_f32(sum_low3, sum_low3);
        float32x2_t mono_f32_3 = vmul_f32(sum_fold3, half_vec);
        int32x2_t mono_s32_3 = vcvt_s32_f32(mono_f32_3);
        s16 mono3 = vget_lane_s16(vqmovn_s32(vcombine_s32(mono_s32_3, vdup_n_s32(0))), 0);
        
        // Add to destination with saturation
        // Load destination values
        int16x4_t dst0_left = vld1_s16(reinterpret_cast<int16_t*>(&dst[i][0]));
        int16x4_t dst0_right = vld1_s16(reinterpret_cast<int16_t*>(&dst[i][1]));
        
        // Create mono vectors with explicit lane assignments
        int16x4_t mono_vec = vdup_n_s16(0);
        mono_vec = vset_lane_s16(mono0, mono_vec, 0);
        mono_vec = vset_lane_s16(mono1, mono_vec, 1);
        mono_vec = vset_lane_s16(mono2, mono_vec, 2);
        mono_vec = vset_lane_s16(mono3, mono_vec, 3);
        
        // Add and saturate
        int16x4_t final_left = vqadd_s16(dst0_left, mono_vec);
        int16x4_t final_right = vqadd_s16(dst0_right, mono_vec);
        
        // Store results - interleaved stereo format with constant indices
        dst[i][0] = vget_lane_s16(final_left, 0);
        dst[i][1] = vget_lane_s16(final_right, 0);
        dst[i+1][0] = vget_lane_s16(final_left, 1);
        dst[i+1][1] = vget_lane_s16(final_right, 1);
        dst[i+2][0] = vget_lane_s16(final_left, 2);
        dst[i+2][1] = vget_lane_s16(final_right, 2);
        dst[i+3][0] = vget_lane_s16(final_left, 3);
        dst[i+3][1] = vget_lane_s16(final_right, 3);
    }
    
    // Handle remaining samples
    int remaining = count & 3; // count % 4
    if (remaining > 0) {
        for (int i = count - remaining; i < count; i++) {
            s16 mono = static_cast<s16>(std::clamp(static_cast<s32>((gain * src[i][0] + gain * src[i][1] + gain * src[i][2] + gain * src[i][3]) / 2), -32768, 32767));
            
            dst[i][0] = static_cast<s16>(std::clamp(static_cast<s32>(dst[i][0]) + static_cast<s32>(mono), -32768, 32767));
            dst[i][1] = static_cast<s16>(std::clamp(static_cast<s32>(dst[i][1]) + static_cast<s32>(mono), -32768, 32767));
        }
    }
}

// NEON-optimized version for converting final mix samples to output format
inline void ConvertFinalMixSamples_NEON(s16* dst, const s16* src, int count) {
    // Process 8 samples at a time
    for (int i = 0; i < count; i += 8) {
        // Load 8 samples
        int16x8_t src_vec = vld1q_s16(&src[i]);
        
        // Store 8 samples
        vst1q_s16(&dst[i], src_vec);
    }
    
    // Handle remaining samples
    int remaining = count & 7; // count % 8
    if (remaining > 0) {
        for (int i = count - remaining; i < count; i++) {
            dst[i] = src[i];
        }
    }
}

} // namespace AudioCore::HLE

#endif // defined(__ARM_NEON) || defined(__aarch64__)
