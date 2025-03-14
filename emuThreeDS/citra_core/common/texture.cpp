// Copyright 2019 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include "common/assert.h"
#include "common/texture.h"

// Use ARM NEON intrinsics for ARM64 platforms
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace Common {

void FlipRGBA8Texture(std::span<u8> tex, u32 width, u32 height) {
    ASSERT(tex.size() == width * height * 4);
    const u32 line_size = width * 4;
    
#if defined(__ARM_NEON) || defined(__aarch64__)
    // Use ARM NEON for faster texture flipping
    for (u32 line = 0; line < height / 2; line++) {
        const u32 offset_1 = line * line_size;
        const u32 offset_2 = (height - line - 1) * line_size;
        
        u8* line1 = tex.data() + offset_1;
        u8* line2 = tex.data() + offset_2;
        
        // Process 16 bytes (4 pixels) at a time using NEON
        for (u32 i = 0; i < line_size; i += 16) {
            if (i + 16 <= line_size) {
                // Load 16 bytes from each line
                uint8x16_t v_line1 = vld1q_u8(line1 + i);
                uint8x16_t v_line2 = vld1q_u8(line2 + i);
                
                // Swap the loaded chunks
                vst1q_u8(line1 + i, v_line2);
                vst1q_u8(line2 + i, v_line1);
            } else {
                // Handle remaining bytes (less than 16)
                for (u32 j = i; j < line_size; j++) {
                    std::swap(line1[j], line2[j]);
                }
            }
        }
    }
#else
    // Original implementation for non-ARM platforms
    for (u32 line = 0; line < height / 2; line++) {
        const u32 offset_1 = line * line_size;
        const u32 offset_2 = (height - line - 1) * line_size;
        // Swap lines
        std::swap_ranges(tex.begin() + offset_1, tex.begin() + offset_1 + line_size,
                         tex.begin() + offset_2);
    }
#endif
}

} // namespace Common
