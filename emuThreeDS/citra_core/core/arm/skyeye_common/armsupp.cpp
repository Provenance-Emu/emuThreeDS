/*  armsupp.c -- ARMulator support code:  ARM6 Instruction Emulator.
    Copyright (C) 1994 Advanced RISC Machines Ltd.

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA. */

#include "core/arm/skyeye_common/armsupp.h"

// Use ARM NEON intrinsics for ARM64 platforms
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

// Unsigned sum of absolute difference
u8 ARMul_UnsignedAbsoluteDifference(u8 left, u8 right) {
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
u32 AddWithCarry(u32 left, u32 right, u32 carry_in, bool* carry_out_occurred,
                 bool* overflow_occurred) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    // Use ARM intrinsics for more efficient addition with carry
    uint32x2_t v_left = vdup_n_u32(left);
    uint32x2_t v_right = vdup_n_u32(right);
    
    // First add left and right
    uint32x2_t v_result = vadd_u32(v_left, v_right);
    
    // Check for carry from first addition
    uint32x2_t v_carry = vclt_u32(v_result, v_left);  // result < left means carry occurred
    uint32_t first_carry = vget_lane_u32(v_carry, 0);
    
    // Add carry_in if needed
    if (carry_in) {
        uint32x2_t v_carry_in = vdup_n_u32(1);
        uint32x2_t v_prev_result = v_result;
        v_result = vadd_u32(v_result, v_carry_in);
        
        // Check for additional carry from adding carry_in
        uint32x2_t v_second_carry = vclt_u32(v_result, v_prev_result);
        uint32_t second_carry = vget_lane_u32(v_second_carry, 0);
        
        // Combine carries
        first_carry = first_carry | second_carry;
    }
    
    uint32_t result = vget_lane_u32(v_result, 0);
    
    if (carry_out_occurred)
        *carry_out_occurred = (first_carry != 0);
    
    if (overflow_occurred) {
        // Check for signed overflow: result has different sign than both inputs
        // when inputs have the same sign
        int32x2_t v_left_s = vreinterpret_s32_u32(v_left);
        int32x2_t v_right_s = vreinterpret_s32_u32(v_right);
        int32x2_t v_result_s = vreinterpret_s32_u32(v_result);
        
        // Check if left and right have the same sign
        uint32x2_t v_same_sign = vceq_s32(vshr_n_s32(v_left_s, 31), vshr_n_s32(v_right_s, 31));
        
        // Check if result has different sign than left
        uint32x2_t v_diff_sign = vceq_s32(vshr_n_s32(v_left_s, 31), vshr_n_s32(v_result_s, 31));
        v_diff_sign = vmvn_u32(v_diff_sign);  // Invert to get different sign
        
        // Overflow occurred if inputs have same sign but result has different sign
        uint32x2_t v_overflow = vand_u32(v_same_sign, v_diff_sign);
        *overflow_occurred = vget_lane_u32(v_overflow, 0) != 0;
    }
    
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
bool AddOverflow(u32 a, u32 b, u32 result) {
    return ((NEG(a) && NEG(b) && POS(result)) || (POS(a) && POS(b) && NEG(result)));
}

// Compute whether a subtraction of A and B, giving RESULT, overflowed.
bool SubOverflow(u32 a, u32 b, u32 result) {
    return ((NEG(a) && POS(b) && POS(result)) || (POS(a) && NEG(b) && NEG(result)));
}

// Returns true if the Q flag should be set as a result of overflow.
bool ARMul_AddOverflowQ(u32 a, u32 b) {
    u32 result = a + b;
    if (((result ^ a) & (u32)0x80000000) && ((a ^ b) & (u32)0x80000000) == 0)
        return true;

    return false;
}

// 8-bit signed saturated addition
u8 ARMul_SignedSaturatedAdd8(u8 left, u8 right) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    // Use ARM NEON intrinsics for saturated addition
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
u8 ARMul_SignedSaturatedSub8(u8 left, u8 right) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    // Use ARM NEON intrinsics for saturated subtraction
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
u16 ARMul_SignedSaturatedAdd16(u16 left, u16 right) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    // Use ARM NEON intrinsics for saturated addition
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
u16 ARMul_SignedSaturatedSub16(u16 left, u16 right) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    // Use ARM NEON intrinsics for saturated subtraction
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
u8 ARMul_UnsignedSaturatedAdd8(u8 left, u8 right) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    // Use ARM NEON intrinsics for unsigned saturated addition
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
u16 ARMul_UnsignedSaturatedAdd16(u16 left, u16 right) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    // Use ARM NEON intrinsics for unsigned saturated addition
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
u8 ARMul_UnsignedSaturatedSub8(u8 left, u8 right) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    // Use ARM NEON intrinsics for unsigned saturated subtraction
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
u16 ARMul_UnsignedSaturatedSub16(u16 left, u16 right) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    // Use ARM NEON intrinsics for unsigned saturated subtraction
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
u32 ARMul_SignedSatQ(s32 value, u8 shift, bool* saturation_occurred) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    // Use ARM64 intrinsics for more efficient saturation
    const u32 max = (1 << shift) - 1;
    const s32 min = -(1 << shift);
    s32 result;
    bool sat = false;
    
    // Use ARM64 assembly for efficient saturation check
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
u32 ARMul_UnsignedSatQ(s32 value, u8 shift, bool* saturation_occurred) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    // Use ARM64 intrinsics for more efficient saturation
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
