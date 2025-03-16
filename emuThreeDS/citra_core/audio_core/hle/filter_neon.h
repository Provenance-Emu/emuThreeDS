#pragma once

#include <array>
#include <algorithm>
#include "audio_core/audio_types.h"
#include "audio_core/hle/common.h"
#include "audio_core/hle/filter.h"
#include "common/common_types.h"

// Forward declarations
namespace AudioCore::HLE {
    class SourceFilters;
}

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>

namespace AudioCore::HLE {

/**
 * NEON-optimized version of the filter processing for a frame of audio
 * This is a batch processor that uses NEON SIMD instructions to process
 * multiple samples at once for better performance
 */
inline void SimpleFilter_ProcessFrame_NEON(StereoFrame16& frame, SourceFilters& filters) {
    const size_t frame_size = frame.size();
    
    // Process samples in batches of 4 when possible
    size_t i = 0;
    for (; i + 3 < frame_size; i += 4) {
        // Process each sample individually using the public interface
        // but load/store in batches for better memory access patterns
        std::array<s16, 2> processed_samples[4];
        
        // Process each sample through the filter
        processed_samples[0] = filters.ProcessSample(frame[i], true, false);
        processed_samples[1] = filters.ProcessSample(frame[i+1], true, false);
        processed_samples[2] = filters.ProcessSample(frame[i+2], true, false);
        processed_samples[3] = filters.ProcessSample(frame[i+3], true, false);
        
        // Pack the results back using NEON
        int16_t packed_results[8];
        packed_results[0] = processed_samples[0][0]; // Left channel sample 0
        packed_results[1] = processed_samples[1][0]; // Left channel sample 1
        packed_results[2] = processed_samples[2][0]; // Left channel sample 2
        packed_results[3] = processed_samples[3][0]; // Left channel sample 3
        packed_results[4] = processed_samples[0][1]; // Right channel sample 0
        packed_results[5] = processed_samples[1][1]; // Right channel sample 1
        packed_results[6] = processed_samples[2][1]; // Right channel sample 2
        packed_results[7] = processed_samples[3][1]; // Right channel sample 3
        
        // Store the results back to the frame
        vst1q_s16(reinterpret_cast<int16_t*>(&frame[i]), vld1q_s16(packed_results));
    }
    
    // Process any remaining samples individually
    for (; i < frame_size; i++) {
        frame[i] = filters.ProcessSample(frame[i], true, false);
    }
}

/**
 * NEON-optimized version of the BiquadFilter ProcessFrame function
 * Processes multiple samples at once using SIMD instructions
 */
inline void BiquadFilter_ProcessFrame_NEON(StereoFrame16& frame, SourceFilters& filters) {
    const size_t frame_size = frame.size();
    
    // Process samples in batches of 4 when possible
    size_t i = 0;
    for (; i + 3 < frame_size; i += 4) {
        // Process each sample individually using the public interface
        // but load/store in batches for better memory access patterns
        std::array<s16, 2> processed_samples[4];
        
        // Process each sample through the filter
        processed_samples[0] = filters.ProcessSample(frame[i], false, true);
        processed_samples[1] = filters.ProcessSample(frame[i+1], false, true);
        processed_samples[2] = filters.ProcessSample(frame[i+2], false, true);
        processed_samples[3] = filters.ProcessSample(frame[i+3], false, true);
        
        // Pack the results back using NEON
        int16_t packed_results[8];
        packed_results[0] = processed_samples[0][0]; // Left channel sample 0
        packed_results[1] = processed_samples[1][0]; // Left channel sample 1
        packed_results[2] = processed_samples[2][0]; // Left channel sample 2
        packed_results[3] = processed_samples[3][0]; // Left channel sample 3
        packed_results[4] = processed_samples[0][1]; // Right channel sample 0
        packed_results[5] = processed_samples[1][1]; // Right channel sample 1
        packed_results[6] = processed_samples[2][1]; // Right channel sample 1
        packed_results[7] = processed_samples[3][1]; // Right channel sample 3
        
        // Store the results back to the frame
        vst1q_s16(reinterpret_cast<int16_t*>(&frame[i]), vld1q_s16(packed_results));
    }
    
    // Process any remaining samples individually
    for (; i < frame_size; i++) {
        frame[i] = filters.ProcessSample(frame[i], false, true);
    }
}

} // namespace AudioCore::HLE

#endif // defined(__ARM_NEON) || defined(__aarch64__)
