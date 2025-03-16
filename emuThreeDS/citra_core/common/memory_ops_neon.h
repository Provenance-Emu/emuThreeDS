// Copyright 2025 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

// Only use ARM NEON intrinsics on ARM64 platforms
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>

namespace Common {
namespace Memory {

/**
 * ARM NEON optimized memory operations
 * These functions provide SIMD-accelerated implementations of common memory operations
 * used throughout the emulator, particularly in memory copy, fill, and comparison operations.
 */
namespace NEON {

/**
 * Optimized memory copy using NEON intrinsics
 * @param dest Destination memory address
 * @param src Source memory address
 * @param size Number of bytes to copy
 */
inline void FastCopy(void* dest, const void* src, size_t size) {
    uint8_t* d = static_cast<uint8_t*>(dest);
    const uint8_t* s = static_cast<const uint8_t*>(src);
    
    // For very small copies, use standard memcpy
    if (size < 16) {
        std::memcpy(d, s, size);
        return;
    }
    
    // Handle unaligned start
    size_t align_offset = reinterpret_cast<uintptr_t>(d) & 0xF;
    if (align_offset != 0) {
        size_t prefix = 16 - align_offset;
        if (prefix > size) {
            prefix = size;
        }
        std::memcpy(d, s, prefix);
        d += prefix;
        s += prefix;
        size -= prefix;
    }
    
    // Fast path: copy 128 bytes at a time
    while (size >= 128) {
        // Prefetch ahead to reduce cache misses
        __builtin_prefetch(s + 256, 0, 0);
        __builtin_prefetch(d + 256, 1, 0);
        
        // Load 128 bytes from source
        uint8x16_t v0 = vld1q_u8(s);
        uint8x16_t v1 = vld1q_u8(s + 16);
        uint8x16_t v2 = vld1q_u8(s + 32);
        uint8x16_t v3 = vld1q_u8(s + 48);
        uint8x16_t v4 = vld1q_u8(s + 64);
        uint8x16_t v5 = vld1q_u8(s + 80);
        uint8x16_t v6 = vld1q_u8(s + 96);
        uint8x16_t v7 = vld1q_u8(s + 112);
        
        // Store 128 bytes to destination
        vst1q_u8(d, v0);
        vst1q_u8(d + 16, v1);
        vst1q_u8(d + 32, v2);
        vst1q_u8(d + 48, v3);
        vst1q_u8(d + 64, v4);
        vst1q_u8(d + 80, v5);
        vst1q_u8(d + 96, v6);
        vst1q_u8(d + 112, v7);
        
        s += 128;
        d += 128;
        size -= 128;
    }
    
    // Copy 64 bytes at a time
    if (size >= 64) {
        uint8x16_t v0 = vld1q_u8(s);
        uint8x16_t v1 = vld1q_u8(s + 16);
        uint8x16_t v2 = vld1q_u8(s + 32);
        uint8x16_t v3 = vld1q_u8(s + 48);
        
        vst1q_u8(d, v0);
        vst1q_u8(d + 16, v1);
        vst1q_u8(d + 32, v2);
        vst1q_u8(d + 48, v3);
        
        s += 64;
        d += 64;
        size -= 64;
    }
    
    // Copy 32 bytes at a time
    if (size >= 32) {
        uint8x16_t v0 = vld1q_u8(s);
        uint8x16_t v1 = vld1q_u8(s + 16);
        
        vst1q_u8(d, v0);
        vst1q_u8(d + 16, v1);
        
        s += 32;
        d += 32;
        size -= 32;
    }
    
    // Copy 16 bytes
    if (size >= 16) {
        uint8x16_t v0 = vld1q_u8(s);
        vst1q_u8(d, v0);
        
        s += 16;
        d += 16;
        size -= 16;
    }
    
    // Handle remaining bytes
    if (size > 0) {
        std::memcpy(d, s, size);
    }
}

/**
 * Optimized memory fill using NEON intrinsics
 * @param dest Destination memory address
 * @param value Byte value to fill with
 * @param size Number of bytes to fill
 */
inline void FastFill(void* dest, uint8_t value, size_t size) {
    uint8_t* d = static_cast<uint8_t*>(dest);
    
    // For very small fills, use standard memset
    if (size < 16) {
        std::memset(d, value, size);
        return;
    }
    
    // Create NEON register with the fill value replicated
    uint8x16_t fill_vec = vdupq_n_u8(value);
    
    // Handle unaligned start
    size_t align_offset = reinterpret_cast<uintptr_t>(d) & 0xF;
    if (align_offset != 0) {
        size_t prefix = 16 - align_offset;
        if (prefix > size) {
            prefix = size;
        }
        std::memset(d, value, prefix);
        d += prefix;
        size -= prefix;
    }
    
    // Fast path: fill 128 bytes at a time
    while (size >= 128) {
        // Prefetch ahead to reduce cache misses
        __builtin_prefetch(d + 256, 1, 0);
        
        // Store 128 bytes to destination
        vst1q_u8(d, fill_vec);
        vst1q_u8(d + 16, fill_vec);
        vst1q_u8(d + 32, fill_vec);
        vst1q_u8(d + 48, fill_vec);
        vst1q_u8(d + 64, fill_vec);
        vst1q_u8(d + 80, fill_vec);
        vst1q_u8(d + 96, fill_vec);
        vst1q_u8(d + 112, fill_vec);
        
        d += 128;
        size -= 128;
    }
    
    // Fill 64 bytes at a time
    if (size >= 64) {
        vst1q_u8(d, fill_vec);
        vst1q_u8(d + 16, fill_vec);
        vst1q_u8(d + 32, fill_vec);
        vst1q_u8(d + 48, fill_vec);
        
        d += 64;
        size -= 64;
    }
    
    // Fill 32 bytes at a time
    if (size >= 32) {
        vst1q_u8(d, fill_vec);
        vst1q_u8(d + 16, fill_vec);
        
        d += 32;
        size -= 32;
    }
    
    // Fill 16 bytes
    if (size >= 16) {
        vst1q_u8(d, fill_vec);
        
        d += 16;
        size -= 16;
    }
    
    // Handle remaining bytes
    if (size > 0) {
        std::memset(d, value, size);
    }
}

/**
 * Optimized memory zero-fill using NEON intrinsics
 * @param dest Destination memory address
 * @param size Number of bytes to zero
 */
inline void FastZero(void* dest, size_t size) {
    uint8_t* d = static_cast<uint8_t*>(dest);
    
    // For very small fills, use standard memset
    if (size < 16) {
        std::memset(d, 0, size);
        return;
    }
    
    // Create NEON register with zeros
    uint8x16_t zero_vec = vdupq_n_u8(0);
    
    // Handle unaligned start
    size_t align_offset = reinterpret_cast<uintptr_t>(d) & 0xF;
    if (align_offset != 0) {
        size_t prefix = 16 - align_offset;
        if (prefix > size) {
            prefix = size;
        }
        std::memset(d, 0, prefix);
        d += prefix;
        size -= prefix;
    }
    
    // Fast path: zero 128 bytes at a time
    while (size >= 128) {
        // Prefetch ahead to reduce cache misses
        __builtin_prefetch(d + 256, 1, 0);
        
        // Store 128 bytes of zeros to destination
        vst1q_u8(d, zero_vec);
        vst1q_u8(d + 16, zero_vec);
        vst1q_u8(d + 32, zero_vec);
        vst1q_u8(d + 48, zero_vec);
        vst1q_u8(d + 64, zero_vec);
        vst1q_u8(d + 80, zero_vec);
        vst1q_u8(d + 96, zero_vec);
        vst1q_u8(d + 112, zero_vec);
        
        d += 128;
        size -= 128;
    }
    
    // Zero 64 bytes at a time
    if (size >= 64) {
        vst1q_u8(d, zero_vec);
        vst1q_u8(d + 16, zero_vec);
        vst1q_u8(d + 32, zero_vec);
        vst1q_u8(d + 48, zero_vec);
        
        d += 64;
        size -= 64;
    }
    
    // Zero 32 bytes at a time
    if (size >= 32) {
        vst1q_u8(d, zero_vec);
        vst1q_u8(d + 16, zero_vec);
        
        d += 32;
        size -= 32;
    }
    
    // Zero 16 bytes
    if (size >= 16) {
        vst1q_u8(d, zero_vec);
        
        d += 16;
        size -= 16;
    }
    
    // Handle remaining bytes
    if (size > 0) {
        std::memset(d, 0, size);
    }
}

/**
 * Optimized memory compare using NEON intrinsics
 * @param buf1 First memory buffer
 * @param buf2 Second memory buffer
 * @param size Number of bytes to compare
 * @return true if buffers are equal, false otherwise
 */
inline bool FastCompare(const void* buf1, const void* buf2, size_t size) {
    const uint8_t* b1 = static_cast<const uint8_t*>(buf1);
    const uint8_t* b2 = static_cast<const uint8_t*>(buf2);
    
    // For very small compares, use standard memcmp
    if (size < 16) {
        return std::memcmp(b1, b2, size) == 0;
    }
    
    // Handle unaligned start
    size_t align_offset = reinterpret_cast<uintptr_t>(b1) & 0xF;
    if (align_offset != 0) {
        size_t prefix = 16 - align_offset;
        if (prefix > size) {
            prefix = size;
        }
        if (std::memcmp(b1, b2, prefix) != 0) {
            return false;
        }
        b1 += prefix;
        b2 += prefix;
        size -= prefix;
    }
    
    // Compare 64 bytes at a time
    while (size >= 64) {
        // Load 64 bytes from each buffer
        uint8x16_t v1_0 = vld1q_u8(b1);
        uint8x16_t v1_1 = vld1q_u8(b1 + 16);
        uint8x16_t v1_2 = vld1q_u8(b1 + 32);
        uint8x16_t v1_3 = vld1q_u8(b1 + 48);
        
        uint8x16_t v2_0 = vld1q_u8(b2);
        uint8x16_t v2_1 = vld1q_u8(b2 + 16);
        uint8x16_t v2_2 = vld1q_u8(b2 + 32);
        uint8x16_t v2_3 = vld1q_u8(b2 + 48);
        
        // Compare 16-byte chunks
        uint8x16_t cmp0 = vceqq_u8(v1_0, v2_0);
        uint8x16_t cmp1 = vceqq_u8(v1_1, v2_1);
        uint8x16_t cmp2 = vceqq_u8(v1_2, v2_2);
        uint8x16_t cmp3 = vceqq_u8(v1_3, v2_3);
        
        // Combine comparison results
        uint8x16_t and0 = vandq_u8(cmp0, cmp1);
        uint8x16_t and1 = vandq_u8(cmp2, cmp3);
        uint8x16_t and_all = vandq_u8(and0, and1);
        
        // Check if any byte is different
        uint64x2_t bitwise_and = vreinterpretq_u64_u8(and_all);
        if (vgetq_lane_u64(bitwise_and, 0) != UINT64_MAX || 
            vgetq_lane_u64(bitwise_and, 1) != UINT64_MAX) {
            return false;
        }
        
        b1 += 64;
        b2 += 64;
        size -= 64;
    }
    
    // Compare 32 bytes at a time
    if (size >= 32) {
        uint8x16_t v1_0 = vld1q_u8(b1);
        uint8x16_t v1_1 = vld1q_u8(b1 + 16);
        
        uint8x16_t v2_0 = vld1q_u8(b2);
        uint8x16_t v2_1 = vld1q_u8(b2 + 16);
        
        uint8x16_t cmp0 = vceqq_u8(v1_0, v2_0);
        uint8x16_t cmp1 = vceqq_u8(v1_1, v2_1);
        
        uint8x16_t and_all = vandq_u8(cmp0, cmp1);
        
        uint64x2_t bitwise_and = vreinterpretq_u64_u8(and_all);
        if (vgetq_lane_u64(bitwise_and, 0) != UINT64_MAX || 
            vgetq_lane_u64(bitwise_and, 1) != UINT64_MAX) {
            return false;
        }
        
        b1 += 32;
        b2 += 32;
        size -= 32;
    }
    
    // Compare 16 bytes
    if (size >= 16) {
        uint8x16_t v1 = vld1q_u8(b1);
        uint8x16_t v2 = vld1q_u8(b2);
        
        uint8x16_t cmp = vceqq_u8(v1, v2);
        
        uint64x2_t bitwise_and = vreinterpretq_u64_u8(cmp);
        if (vgetq_lane_u64(bitwise_and, 0) != UINT64_MAX || 
            vgetq_lane_u64(bitwise_and, 1) != UINT64_MAX) {
            return false;
        }
        
        b1 += 16;
        b2 += 16;
        size -= 16;
    }
    
    // Handle remaining bytes
    if (size > 0) {
        return std::memcmp(b1, b2, size) == 0;
    }
    
    return true;
}

/**
 * Optimized memory block read using NEON intrinsics
 * Reads a block of memory from src to dest with potential format conversion
 * @param dest Destination buffer (32-bit aligned)
 * @param src Source buffer
 * @param size Number of bytes to read
 */
inline void ReadBlock(uint32_t* dest, const uint8_t* src, size_t size) {
    // For very small reads, use standard approach
    if (size < 16) {
        for (size_t i = 0; i < size / 4; i++) {
            dest[i] = *reinterpret_cast<const uint32_t*>(src + i * 4);
        }
        return;
    }
    
    // Process 64 bytes (16 uint32_t values) at a time
    while (size >= 64) {
        // Prefetch ahead to reduce cache misses
        __builtin_prefetch(src + 128, 0, 0);
        
        // Load 64 bytes from source
        uint8x16x4_t data = vld4q_u8(src);
        
        // Reinterpret as 32-bit values and store
        vst1q_u32(dest, vreinterpretq_u32_u8(data.val[0]));
        vst1q_u32(dest + 4, vreinterpretq_u32_u8(data.val[1]));
        vst1q_u32(dest + 8, vreinterpretq_u32_u8(data.val[2]));
        vst1q_u32(dest + 12, vreinterpretq_u32_u8(data.val[3]));
        
        src += 64;
        dest += 16;
        size -= 64;
    }
    
    // Process 16 bytes (4 uint32_t values) at a time
    while (size >= 16) {
        uint8x16_t data = vld1q_u8(src);
        vst1q_u32(dest, vreinterpretq_u32_u8(data));
        
        src += 16;
        dest += 4;
        size -= 16;
    }
    
    // Handle remaining bytes
    for (size_t i = 0; i < size / 4; i++) {
        dest[i] = *reinterpret_cast<const uint32_t*>(src + i * 4);
    }
}

/**
 * Optimized memory block write using NEON intrinsics
 * Writes a block of memory from src to dest with potential format conversion
 * @param dest Destination buffer
 * @param src Source buffer (32-bit aligned)
 * @param size Number of bytes to write
 */
inline void WriteBlock(uint8_t* dest, const uint32_t* src, size_t size) {
    // For very small writes, use standard approach
    if (size < 16) {
        for (size_t i = 0; i < size / 4; i++) {
            *reinterpret_cast<uint32_t*>(dest + i * 4) = src[i];
        }
        return;
    }
    
    // Process 64 bytes (16 uint32_t values) at a time
    while (size >= 64) {
        // Prefetch ahead to reduce cache misses
        __builtin_prefetch(dest + 128, 1, 0);
        
        // Load 16 uint32_t values from source
        uint32x4_t data0 = vld1q_u32(src);
        uint32x4_t data1 = vld1q_u32(src + 4);
        uint32x4_t data2 = vld1q_u32(src + 8);
        uint32x4_t data3 = vld1q_u32(src + 12);
        
        // Reinterpret as 8-bit values and store
        vst1q_u8(dest, vreinterpretq_u8_u32(data0));
        vst1q_u8(dest + 16, vreinterpretq_u8_u32(data1));
        vst1q_u8(dest + 32, vreinterpretq_u8_u32(data2));
        vst1q_u8(dest + 48, vreinterpretq_u8_u32(data3));
        
        dest += 64;
        src += 16;
        size -= 64;
    }
    
    // Process 16 bytes (4 uint32_t values) at a time
    while (size >= 16) {
        uint32x4_t data = vld1q_u32(src);
        vst1q_u8(dest, vreinterpretq_u8_u32(data));
        
        dest += 16;
        src += 4;
        size -= 16;
    }
    
    // Handle remaining bytes
    for (size_t i = 0; i < size / 4; i++) {
        *reinterpret_cast<uint32_t*>(dest + i * 4) = src[i];
    }
}

} // namespace NEON
} // namespace Memory
} // namespace Common

#endif // defined(__ARM_NEON) || defined(__aarch64__)
