// Copyright 2014 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "common/common_types.h"

#define BITS(s, a, b) ((s << ((sizeof(s) * 8 - 1) - b)) >> (sizeof(s) * 8 - b + a - 1))
#define BIT(s, n) ((s >> (n)) & 1)

#define POS(i) ((~(i)) >> 31)
#define NEG(i) ((i) >> 31)

// Use ARM NEON intrinsics for ARM64 platforms
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

// Unsigned sum of absolute difference
inline u8 ARMul_UnsignedAbsoluteDifference(u8 left, u8 right) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    // Use ARM NEON intrinsics for absolute difference
    uint8x8_t v_left = vdup_n_u8(left);
    uint8x8_t v_right = vdup_n_u8(right);
    uint8x8_t result = vabd_u8(v_left, v_right);
    return vget_lane_u8(result, 0);
#else
    // Fallback for non-ARM platforms
    if (left > right)
        return left - right;

    return right - left;
#endif
}

// Add with carry, indicates if a carry-out or signed overflow occurred.
inline u32 AddWithCarry(u32 left, u32 right, u32 carry_in, bool* carry_out_occurred,
                 bool* overflow_occurred) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    u32 result;
    u32 carry_out = 0;
    u32 overflow = 0;
    
    // Use direct ARM64 assembly for the most efficient implementation
    // This uses the adds/adcs instructions which set the carry and overflow flags
    __asm__ volatile(
        "mov w3, %w[carry_in]\n"    // Move carry_in to w3 register
        "cmp w3, #0\n"             // Compare carry_in with 0
        "cset w3, ne\n"            // Set w3 to 1 if carry_in != 0, otherwise 0
        
        "adds %w[result], %w[left], %w[right]\n"  // result = left + right, set flags
        "cset w4, cs\n"            // Set w4 to 1 if carry set (CS), otherwise 0
        "cset w5, vs\n"            // Set w5 to 1 if overflow set (VS), otherwise 0
        
        "cmp w3, #0\n"             // Check if we need to add carry_in
        "beq 1f\n"                 // Skip adding carry if w3 == 0
        
        "adds %w[result], %w[result], #1\n"  // result += 1, set flags
        "cset w3, cs\n"            // Set w3 to 1 if carry set (CS), otherwise 0
        "orr w4, w4, w3\n"         // Combine carries: w4 |= w3
        "cset w3, vs\n"            // Set w3 to 1 if overflow set (VS), otherwise 0
        "orr w5, w5, w3\n"         // Combine overflows: w5 |= w3
        
        "1:\n"                     // Label for skipping carry addition
        "mov %w[carry_out], w4\n"   // Store final carry result
        "mov %w[overflow], w5\n"    // Store final overflow result
        
        : [result] "=r" (result), [carry_out] "=r" (carry_out), [overflow] "=r" (overflow)
        : [left] "r" (left), [right] "r" (right), [carry_in] "r" (carry_in)
        : "w3", "w4", "w5", "cc"  // Clobbered registers and condition codes
    );
    
    if (carry_out_occurred)
        *carry_out_occurred = (carry_out != 0);
    
    if (overflow_occurred)
        *overflow_occurred = (overflow != 0);
    
    return result;
#else
    // Fallback for non-ARM platforms
    u64 unsigned_sum = (u64)left + (u64)right + (u64)carry_in;
    s64 signed_sum = (s64)(s32)left + (s64)(s32)right + (s64)carry_in;
    u64 result = (unsigned_sum & 0xFFFFFFFF);

    if (carry_out_occurred)
        *carry_out_occurred = (result != unsigned_sum);

    if (overflow_occurred)
        *overflow_occurred = ((s64)(s32)result != signed_sum);

    return (u32)result;
#endif
}

// Compute whether an addition of A and B, giving RESULT, overflowed.
inline bool AddOverflow(u32 a, u32 b, u32 result) {
    return ((NEG(a) && NEG(b) && POS(result)) || (POS(a) && POS(b) && NEG(result)));
}

// Compute whether a subtraction of A and B, giving RESULT, overflowed.
inline bool SubOverflow(u32 a, u32 b, u32 result) {
    return ((NEG(a) && POS(b) && POS(result)) || (POS(a) && NEG(b) && NEG(result)));
}

// Returns true if the Q flag should be set as a result of overflow.
inline bool ARMul_AddOverflowQ(u32 a, u32 b) {
    u32 result = a + b;
    if (((result ^ a) & (u32)0x80000000) && ((a ^ b) & (u32)0x80000000) == 0)
        return true;

    return false;
}

// 8-bit signed saturated addition
inline u8 ARMul_SignedSaturatedAdd8(u8 left, u8 right) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    // Use ARM NEON intrinsics for signed saturated addition
    // We use NEON intrinsics as they're more portable across ARM compilers
    int8x8_t v_left = vdup_n_s8((s8)left);
    int8x8_t v_right = vdup_n_s8((s8)right);
    int8x8_t result = vqadd_s8(v_left, v_right);
    return (u8)vget_lane_s8(result, 0);
#else
    // Fallback for non-ARM platforms
    u8 result = left + right;

    if (((result ^ left) & 0x80) && ((left ^ right) & 0x80) == 0) {
        if (left & 0x80)
            result = 0x80;
        else
            result = 0x7F;
    }

    return result;
#endif
}

// 8-bit signed saturated subtraction
inline u8 ARMul_SignedSaturatedSub8(u8 left, u8 right) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    // Use ARM NEON intrinsics for signed saturated subtraction
    // We use NEON intrinsics as they're more portable across ARM compilers
    int8x8_t v_left = vdup_n_s8((s8)left);
    int8x8_t v_right = vdup_n_s8((s8)right);
    int8x8_t result = vqsub_s8(v_left, v_right);
    return (u8)vget_lane_s8(result, 0);
#else
    // Fallback for non-ARM platforms
    u8 result = left - right;

    if (((result ^ left) & 0x80) && ((left ^ right) & 0x80) != 0) {
        if (left & 0x80)
            result = 0x80;
        else
            result = 0x7F;
    }

    return result;
#endif
}

// 16-bit signed saturated addition
inline u16 ARMul_SignedSaturatedAdd16(u16 left, u16 right) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    // Use ARM NEON intrinsics for signed saturated addition
    // We use NEON intrinsics as they're more portable across ARM compilers
    int16x4_t v_left = vdup_n_s16((s16)left);
    int16x4_t v_right = vdup_n_s16((s16)right);
    int16x4_t result = vqadd_s16(v_left, v_right);
    return (u16)vget_lane_s16(result, 0);
#else
    // Fallback for non-ARM platforms
    u16 result = left + right;

    if (((result ^ left) & 0x8000) && ((left ^ right) & 0x8000) == 0) {
        if (left & 0x8000)
            result = 0x8000;
        else
            result = 0x7FFF;
    }

    return result;
#endif
}

// 16-bit signed saturated subtraction
inline u16 ARMul_SignedSaturatedSub16(u16 left, u16 right) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    // Use ARM NEON intrinsics for signed saturated subtraction
    // We use NEON intrinsics as they're more portable across ARM compilers
    int16x4_t v_left = vdup_n_s16((s16)left);
    int16x4_t v_right = vdup_n_s16((s16)right);
    int16x4_t result = vqsub_s16(v_left, v_right);
    return (u16)vget_lane_s16(result, 0);
#else
    // Fallback for non-ARM platforms
    u16 result = left - right;

    if (((result ^ left) & 0x8000) && ((left ^ right) & 0x8000) != 0) {
        if (left & 0x8000)
            result = 0x8000;
        else
            result = 0x7FFF;
    }

    return result;
#endif
}

// 8-bit unsigned saturated addition
inline u8 ARMul_UnsignedSaturatedAdd8(u8 left, u8 right) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    // Use ARM NEON intrinsics for unsigned saturated addition
    // We use NEON intrinsics as they're more portable across ARM compilers
    uint8x8_t v_left = vdup_n_u8(left);
    uint8x8_t v_right = vdup_n_u8(right);
    uint8x8_t result = vqadd_u8(v_left, v_right);
    return vget_lane_u8(result, 0);
#else
    // Fallback for non-ARM platforms
    u8 result = left + right;

    if (result < left)
        result = 0xFF;

    return result;
#endif
}

// 16-bit unsigned saturated addition
inline u16 ARMul_UnsignedSaturatedAdd16(u16 left, u16 right) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    // Use ARM NEON intrinsics for unsigned saturated addition
    // We use NEON intrinsics as they're more portable across ARM compilers
    uint16x4_t v_left = vdup_n_u16(left);
    uint16x4_t v_right = vdup_n_u16(right);
    uint16x4_t result = vqadd_u16(v_left, v_right);
    return vget_lane_u16(result, 0);
#else
    // Fallback for non-ARM platforms
    u16 result = left + right;

    if (result < left)
        result = 0xFFFF;

    return result;
#endif
}

// 8-bit unsigned saturated subtraction
inline u8 ARMul_UnsignedSaturatedSub8(u8 left, u8 right) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    // Use ARM NEON intrinsics for unsigned saturated subtraction
    // We use NEON intrinsics as they're more portable across ARM compilers
    uint8x8_t v_left = vdup_n_u8(left);
    uint8x8_t v_right = vdup_n_u8(right);
    uint8x8_t result = vqsub_u8(v_left, v_right);
    return vget_lane_u8(result, 0);
#else
    // Fallback for non-ARM platforms
    if (left <= right)
        return 0;

    return left - right;
#endif
}

// 16-bit unsigned saturated subtraction
inline u16 ARMul_UnsignedSaturatedSub16(u16 left, u16 right) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    // Use ARM NEON intrinsics for unsigned saturated subtraction
    // We use NEON intrinsics as they're more portable across ARM compilers
    uint16x4_t v_left = vdup_n_u16(left);
    uint16x4_t v_right = vdup_n_u16(right);
    uint16x4_t result = vqsub_u16(v_left, v_right);
    return vget_lane_u16(result, 0);
#else
    // Fallback for non-ARM platforms
    if (left <= right)
        return 0;

    return left - right;
#endif
}

// Signed saturation.
inline u32 ARMul_SignedSatQ(s32 value, u8 shift, bool* saturation_occurred) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    // Use a more portable approach with NEON intrinsics
    const u32 max = (1 << shift) - 1;
    const s32 min = -(1 << shift);
    s32 result;
    bool sat = false;
    
    if (value > (s32)max) {
        result = max;
        sat = true;
    } else if (value < min) {
        result = min;
        sat = true;
    } else {
        result = value;
    }
    
    if (saturation_occurred)
        *saturation_occurred = sat;
    
    return (u32)result;
#else
    // Fallback for non-ARM platforms
    const u32 max = (1 << shift) - 1;
    const s32 top = (value >> shift);

    if (top > 0) {
        *saturation_occurred = true;
        return max;
    } else if (top < -1) {
        *saturation_occurred = true;
        return ~max;
    }

    *saturation_occurred = false;
    return (u32)value;
#endif
}

// Unsigned saturation
inline u32 ARMul_UnsignedSatQ(s32 value, u8 shift, bool* saturation_occurred) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    // Use a more portable approach
    const u32 max = (1 << shift) - 1;
    u32 result;
    bool sat = false;
    
    if (value < 0) {
        result = 0;
        sat = true;
    } else if ((u32)value > max) {
        result = max;
        sat = true;
    } else {
        result = (u32)value;
    }
    
    if (saturation_occurred)
        *saturation_occurred = sat;
    
    return result;
#else
    // Fallback for non-ARM platforms
    const u32 max = (1 << shift) - 1;

    if (value < 0) {
        *saturation_occurred = true;
        return 0;
    } else if ((u32)value > max) {
        *saturation_occurred = true;
        return max;
    }

    *saturation_occurred = false;
    return (u32)value;
#endif
}

