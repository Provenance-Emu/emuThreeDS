// Copyright 2025 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <cstddef>
#include <cstdint>

// Only use ARM NEON intrinsics on ARM64 platforms
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#include "common/vector_math.h"
#include "video_core/pica_types.h"
#include "video_core/regs_framebuffer.h"

namespace Pica {
namespace ColorBlending {
namespace NEON {

/**
 * Optimized alpha blending using NEON intrinsics
 * @param src Source color (RGBA8)
 * @param dst Destination color (RGBA8)
 * @return Blended color (RGBA8)
 */
inline Common::Vec4<u8> AlphaBlend(const Common::Vec4<u8>& src, const Common::Vec4<u8>& dst) {
    // Load source and destination colors
    uint8x8_t src_vec = vdup_n_u8(0);
    uint8x8_t dst_vec = vdup_n_u8(0);
    
    src_vec = vset_lane_u8(src.r(), src_vec, 0);
    src_vec = vset_lane_u8(src.g(), src_vec, 1);
    src_vec = vset_lane_u8(src.b(), src_vec, 2);
    src_vec = vset_lane_u8(src.a(), src_vec, 3);
    
    dst_vec = vset_lane_u8(dst.r(), dst_vec, 0);
    dst_vec = vset_lane_u8(dst.g(), dst_vec, 1);
    dst_vec = vset_lane_u8(dst.b(), dst_vec, 2);
    dst_vec = vset_lane_u8(dst.a(), dst_vec, 3);
    
    // Extract alpha values
    uint16x8_t src_alpha = vmovl_u8(vdup_n_u8(src.a()));
    uint16x8_t dst_alpha = vmovl_u8(vdup_n_u8(dst.a()));
    uint16x8_t inv_src_alpha = vdupq_n_u16(255) - src_alpha;
    
    // Convert colors to 16-bit
    uint16x8_t src_wide = vmovl_u8(src_vec);
    uint16x8_t dst_wide = vmovl_u8(dst_vec);
    
    // Blend formula: src * src_alpha + dst * (1 - src_alpha)
    uint16x8_t src_contrib = vmulq_u16(src_wide, src_alpha);
    uint16x8_t dst_contrib = vmulq_u16(dst_wide, inv_src_alpha);
    
    // Sum contributions and divide by 255
    uint16x8_t sum = vaddq_u16(src_contrib, dst_contrib);
    uint16x8_t result = vshrq_n_u16(sum, 8);
    
    // Convert back to 8-bit
    uint8x8_t result_narrow = vmovn_u16(result);
    
    // Extract components
    Common::Vec4<u8> blended;
    blended.r() = vget_lane_u8(result_narrow, 0);
    blended.g() = vget_lane_u8(result_narrow, 1);
    blended.b() = vget_lane_u8(result_narrow, 2);
    blended.a() = vget_lane_u8(result_narrow, 3);
    
    return blended;
}

/**
 * Optimized batch alpha blending using NEON intrinsics
 * @param dst Destination buffer (RGBA8)
 * @param src Source buffer (RGBA8)
 * @param count Number of pixels to blend
 */
inline void AlphaBlendBatch(Common::Vec4<u8>* dst, const Common::Vec4<u8>* src, size_t count) {
    // Process 4 pixels at a time when possible
    size_t i = 0;
    for (; i + 3 < count; i += 4) {
        // Load 4 source pixels
        uint8x16x4_t src_pixels;
        src_pixels.val[0] = vld1q_u8(reinterpret_cast<const uint8_t*>(&src[i]));
        src_pixels.val[1] = vld1q_u8(reinterpret_cast<const uint8_t*>(&src[i+1]));
        src_pixels.val[2] = vld1q_u8(reinterpret_cast<const uint8_t*>(&src[i+2]));
        src_pixels.val[3] = vld1q_u8(reinterpret_cast<const uint8_t*>(&src[i+3]));
        
        // Load 4 destination pixels
        uint8x16x4_t dst_pixels;
        dst_pixels.val[0] = vld1q_u8(reinterpret_cast<const uint8_t*>(&dst[i]));
        dst_pixels.val[1] = vld1q_u8(reinterpret_cast<const uint8_t*>(&dst[i+1]));
        dst_pixels.val[2] = vld1q_u8(reinterpret_cast<const uint8_t*>(&dst[i+2]));
        dst_pixels.val[3] = vld1q_u8(reinterpret_cast<const uint8_t*>(&dst[i+3]));
        
        // Process each pixel
        for (int j = 0; j < 4; j++) {
            // Extract alpha values
            uint8x16_t src_alpha = vdupq_lane_u8(vget_high_u8(vreinterpretq_u8_u32(src_pixels.val[j])), 3);
            uint8x16_t inv_src_alpha = vsubq_u8(vdupq_n_u8(255), src_alpha);
            
            // Blend formula: src * src_alpha + dst * (1 - src_alpha)
            uint16x8_t src_low = vmovl_u8(vget_low_u8(src_pixels.val[j]));
            uint16x8_t src_high = vmovl_u8(vget_high_u8(src_pixels.val[j]));
            uint16x8_t dst_low = vmovl_u8(vget_low_u8(dst_pixels.val[j]));
            uint16x8_t dst_high = vmovl_u8(vget_high_u8(dst_pixels.val[j]));
            
            uint16x8_t src_alpha_low = vmovl_u8(vget_low_u8(src_alpha));
            uint16x8_t src_alpha_high = vmovl_u8(vget_high_u8(src_alpha));
            uint16x8_t inv_src_alpha_low = vmovl_u8(vget_low_u8(inv_src_alpha));
            uint16x8_t inv_src_alpha_high = vmovl_u8(vget_high_u8(inv_src_alpha));
            
            uint16x8_t src_contrib_low = vmulq_u16(src_low, src_alpha_low);
            uint16x8_t src_contrib_high = vmulq_u16(src_high, src_alpha_high);
            uint16x8_t dst_contrib_low = vmulq_u16(dst_low, inv_src_alpha_low);
            uint16x8_t dst_contrib_high = vmulq_u16(dst_high, inv_src_alpha_high);
            
            uint16x8_t sum_low = vaddq_u16(src_contrib_low, dst_contrib_low);
            uint16x8_t sum_high = vaddq_u16(src_contrib_high, dst_contrib_high);
            
            uint16x8_t result_low = vshrq_n_u16(sum_low, 8);
            uint16x8_t result_high = vshrq_n_u16(sum_high, 8);
            
            uint8x8_t result_low_narrow = vmovn_u16(result_low);
            uint8x8_t result_high_narrow = vmovn_u16(result_high);
            
            uint8x16_t result = vcombine_u8(result_low_narrow, result_high_narrow);
            
            // Store result
            vst1q_u8(reinterpret_cast<uint8_t*>(&dst[i+j]), result);
        }
    }
    
    // Handle remaining pixels
    for (; i < count; i++) {
        dst[i] = AlphaBlend(src[i], dst[i]);
    }
}

/**
 * Optimized color blending with custom blend equation using NEON intrinsics
 * @param src Source color (RGBA8)
 * @param dst Destination color (RGBA8)
 * @param src_factor Source blend factor
 * @param dst_factor Destination blend factor
 * @param blend_op Blend operation
 * @return Blended color (RGBA8)
 */
inline Common::Vec4<u8> BlendCustom(const Common::Vec4<u8>& src, const Common::Vec4<u8>& dst,
                                    Pica::FramebufferRegs::BlendFactor src_factor, Pica::FramebufferRegs::BlendFactor dst_factor,
                                    FramebufferRegs::BlendEquation blend_op) {
    // Load source and destination colors
    uint8x8_t src_vec = vdup_n_u8(0);
    uint8x8_t dst_vec = vdup_n_u8(0);
    
    src_vec = vset_lane_u8(src.r(), src_vec, 0);
    src_vec = vset_lane_u8(src.g(), src_vec, 1);
    src_vec = vset_lane_u8(src.b(), src_vec, 2);
    src_vec = vset_lane_u8(src.a(), src_vec, 3);
    
    dst_vec = vset_lane_u8(dst.r(), dst_vec, 0);
    dst_vec = vset_lane_u8(dst.g(), dst_vec, 1);
    dst_vec = vset_lane_u8(dst.b(), dst_vec, 2);
    dst_vec = vset_lane_u8(dst.a(), dst_vec, 3);
    
    // Convert colors to 16-bit
    uint16x8_t src_wide = vmovl_u8(src_vec);
    uint16x8_t dst_wide = vmovl_u8(dst_vec);
    
    // Calculate source factor
    uint16x8_t src_factor_vec;
    switch (src_factor) {
        case FramebufferRegs::BlendFactor::Zero:
        src_factor_vec = vdupq_n_u16(0);
        break;
    case FramebufferRegs::BlendFactor::One:
        src_factor_vec = vdupq_n_u16(255);
        break;
    case FramebufferRegs::BlendFactor::SourceAlpha:
        src_factor_vec = vdupq_n_u16(src.a());
        break;
    case FramebufferRegs::BlendFactor::OneMinusSourceAlpha:
        src_factor_vec = vdupq_n_u16(255 - src.a());
        break;
    case FramebufferRegs::BlendFactor::DestAlpha:
        src_factor_vec = vdupq_n_u16(dst.a());
        break;
    case FramebufferRegs::BlendFactor::OneMinusDestAlpha:
        src_factor_vec = vdupq_n_u16(255 - dst.a());
        break;
    default:
        src_factor_vec = vdupq_n_u16(255);
        break;
    }
    
    // Calculate destination factor
    uint16x8_t dst_factor_vec;
    switch (dst_factor) {
    case FramebufferRegs::BlendFactor::Zero:
        dst_factor_vec = vdupq_n_u16(0);
        break;
    case FramebufferRegs::BlendFactor::One:
        dst_factor_vec = vdupq_n_u16(255);
        break;
    case FramebufferRegs::BlendFactor::SourceAlpha:
        dst_factor_vec = vdupq_n_u16(src.a());
        break;
    case FramebufferRegs::BlendFactor::OneMinusSourceAlpha:
        dst_factor_vec = vdupq_n_u16(255 - src.a());
        break;
    case FramebufferRegs::BlendFactor::DestAlpha:
        dst_factor_vec = vdupq_n_u16(dst.a());
        break;
    case FramebufferRegs::BlendFactor::OneMinusDestAlpha:
        dst_factor_vec = vdupq_n_u16(255 - dst.a());
        break;
    default:
        dst_factor_vec = vdupq_n_u16(255);
        break;
    }
    
    // Apply factors
    uint16x8_t src_contrib = vmulq_u16(src_wide, src_factor_vec);
    uint16x8_t dst_contrib = vmulq_u16(dst_wide, dst_factor_vec);
    
    // Apply blend operation
    uint16x8_t result;
    switch (blend_op) {
    case FramebufferRegs::BlendEquation::Add:
        result = vaddq_u16(src_contrib, dst_contrib);
        break;
    case FramebufferRegs::BlendEquation::Subtract:
        result = vsubq_u16(src_contrib, dst_contrib);
        break;
    case FramebufferRegs::BlendEquation::ReverseSubtract:
        result = vsubq_u16(dst_contrib, src_contrib);
        break;
    case FramebufferRegs::BlendEquation::Min:
        result = vminq_u16(src_contrib, dst_contrib);
        break;
    case FramebufferRegs::BlendEquation::Max:
        result = vmaxq_u16(src_contrib, dst_contrib);
        break;
    default:
        result = vaddq_u16(src_contrib, dst_contrib);
        break;
    }
    
    // Normalize by dividing by 255
    result = vshrq_n_u16(result, 8);
    
    // Clamp to [0, 255]
    result = vminq_u16(result, vdupq_n_u16(255));
    
    // Convert back to 8-bit
    uint8x8_t result_narrow = vmovn_u16(result);
    
    // Extract components
    Common::Vec4<u8> blended;
    blended.r() = vget_lane_u8(result_narrow, 0);
    blended.g() = vget_lane_u8(result_narrow, 1);
    blended.b() = vget_lane_u8(result_narrow, 2);
    blended.a() = vget_lane_u8(result_narrow, 3);
    
    return blended;
}

/**
 * Optimized color format conversion from RGBA8 to RGB565 using NEON intrinsics
 * @param dst Destination buffer (RGB565)
 * @param src Source buffer (RGBA8)
 * @param count Number of pixels to convert
 */
inline void ConvertRGBA8ToRGB565(u16* dst, const Common::Vec4<u8>* src, size_t count) {
    // Process 8 pixels at a time when possible
    size_t i = 0;
    for (; i + 7 < count; i += 8) {
        // Load 8 RGBA8 pixels
        uint8x16x4_t rgba;
        rgba.val[0] = vld1q_u8(reinterpret_cast<const uint8_t*>(&src[i]));
        rgba.val[1] = vld1q_u8(reinterpret_cast<const uint8_t*>(&src[i+2]));
        rgba.val[2] = vld1q_u8(reinterpret_cast<const uint8_t*>(&src[i+4]));
        rgba.val[3] = vld1q_u8(reinterpret_cast<const uint8_t*>(&src[i+6]));
        
        // Extract R, G, B components for each pixel
        uint8x8x4_t r, g, b;
        for (int j = 0; j < 4; j++) {
            uint8x16_t rgba_vec = rgba.val[j];
            uint8x16x4_t rgba_deinterleaved = vld4q_u8(reinterpret_cast<const uint8_t*>(&rgba_vec));
            
            r.val[j] = vget_low_u8(rgba_deinterleaved.val[0]);
            g.val[j] = vget_low_u8(rgba_deinterleaved.val[1]);
            b.val[j] = vget_low_u8(rgba_deinterleaved.val[2]);
        }
        
        // Convert to RGB565 format
        uint16x8_t r565 = vshlq_n_u16(vmovl_u8(r.val[0]), 11);
        uint16x8_t g565 = vshlq_n_u16(vmovl_u8(g.val[0]), 5);
        uint16x8_t b565 = vmovl_u8(b.val[0]);
        
        // Combine components
        uint16x8_t rgb565 = vorrq_u16(vorrq_u16(r565, g565), b565);
        
        // Store result
        vst1q_u16(dst + i, rgb565);
    }
    
    // Handle remaining pixels
    for (; i < count; i++) {
        const Common::Vec4<u8>& rgba = src[i];
        u16 r = (rgba.r() >> 3) & 0x1F;
        u16 g = (rgba.g() >> 2) & 0x3F;
        u16 b = (rgba.b() >> 3) & 0x1F;
        dst[i] = (r << 11) | (g << 5) | b;
    }
}

/**
 * Optimized color format conversion from RGB565 to RGBA8 using NEON intrinsics
 * @param dst Destination buffer (RGBA8)
 * @param src Source buffer (RGB565)
 * @param count Number of pixels to convert
 */
inline void ConvertRGB565ToRGBA8(Common::Vec4<u8>* dst, const u16* src, size_t count) {
    // Process 8 pixels at a time when possible
    size_t i = 0;
    for (; i + 7 < count; i += 8) {
        // Load 8 RGB565 pixels
        uint16x8_t rgb565 = vld1q_u16(src + i);
        
        // Extract R, G, B components
        uint16x8_t r = vshrq_n_u16(vandq_u16(rgb565, vdupq_n_u16(0xF800)), 11);
        uint16x8_t g = vshrq_n_u16(vandq_u16(rgb565, vdupq_n_u16(0x07E0)), 5);
        uint16x8_t b = vandq_u16(rgb565, vdupq_n_u16(0x001F));
        
        // Convert to 8-bit per channel (with bit replication for precision)
        uint8x8_t r8 = vmovn_u16(vshlq_n_u16(vorrq_u16(r, vshlq_n_u16(r, 5)), 3));
        uint8x8_t g8 = vmovn_u16(vshlq_n_u16(vorrq_u16(g, vshlq_n_u16(g, 6)), 2));
        uint8x8_t b8 = vmovn_u16(vshlq_n_u16(vorrq_u16(b, vshlq_n_u16(b, 5)), 3));
        uint8x8_t a8 = vdup_n_u8(255);
        
        // Interleave components
        uint8x8x4_t rgba;
        rgba.val[0] = r8;
        rgba.val[1] = g8;
        rgba.val[2] = b8;
        rgba.val[3] = a8;
        
        // Store result
        vst4_u8(reinterpret_cast<uint8_t*>(dst + i), rgba);
    }
    
    // Handle remaining pixels
    for (; i < count; i++) {
        u16 rgb565 = src[i];
        Common::Vec4<u8>& rgba = dst[i];
        
        rgba.r() = ((rgb565 >> 11) & 0x1F) << 3;
        rgba.g() = ((rgb565 >> 5) & 0x3F) << 2;
        rgba.b() = (rgb565 & 0x1F) << 3;
        rgba.a() = 255;
        
        // Replicate bits for better precision
        rgba.r() |= rgba.r() >> 5;
        rgba.g() |= rgba.g() >> 6;
        rgba.b() |= rgba.b() >> 5;
    }
}

} // namespace NEON
} // namespace ColorBlending
} // namespace Pica

#endif // defined(__ARM_NEON) || defined(__aarch64__)
