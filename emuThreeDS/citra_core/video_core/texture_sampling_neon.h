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
#include "common/vector_math_neon.h"
#include "video_core/pica_types.h"

namespace Pica {
namespace TextureSampling {
namespace NEON {

/**
 * Optimized bilinear texture sampling using NEON intrinsics
 * @param src Source texture data
 * @param x X coordinate (fixed-point)
 * @param y Y coordinate (fixed-point)
 * @param width Texture width
 * @param height Texture height
 * @param stride Texture stride in bytes
 * @return Sampled RGBA8 color
 */
inline Common::Vec4<u8> SampleBilinear(const u8* src, u32 x, u32 y, u32 width, u32 height, u32 stride) {
    // Extract integer and fractional parts
    const u32 x_int = x >> 8;
    const u32 y_int = y >> 8;
    const u32 x_frac = x & 0xFF;
    const u32 y_frac = y & 0xFF;
    
    // Calculate weights for bilinear interpolation
    const uint8x8_t x_weight = vdup_n_u8(x_frac);
    const uint8x8_t y_weight = vdup_n_u8(y_frac);
    const uint8x8_t inv_x_weight = vdup_n_u8(255 - x_frac);
    const uint8x8_t inv_y_weight = vdup_n_u8(255 - y_frac);
    
    // Calculate pixel addresses with bounds checking
    const u32 x0 = std::min(x_int, width - 1);
    const u32 y0 = std::min(y_int, height - 1);
    const u32 x1 = std::min(x_int + 1, width - 1);
    const u32 y1 = std::min(y_int + 1, height - 1);
    
    // Get pointers to the four pixels
    const u8* p00 = src + y0 * stride + x0 * 4;
    const u8* p01 = src + y0 * stride + x1 * 4;
    const u8* p10 = src + y1 * stride + x0 * 4;
    const u8* p11 = src + y1 * stride + x1 * 4;
    
    // Load pixel values (RGBA)
    uint8x8_t p00_vec = vld1_u8(p00);
    uint8x8_t p01_vec = vld1_u8(p01);
    uint8x8_t p10_vec = vld1_u8(p10);
    uint8x8_t p11_vec = vld1_u8(p11);
    
    // First interpolate horizontally
    // top row: p00 * (1-x_frac) + p01 * x_frac
    uint16x8_t top_row = vmull_u8(p00_vec, inv_x_weight);
    top_row = vmlal_u8(top_row, p01_vec, x_weight);
    
    // bottom row: p10 * (1-x_frac) + p11 * x_frac
    uint16x8_t bottom_row = vmull_u8(p10_vec, inv_x_weight);
    bottom_row = vmlal_u8(bottom_row, p11_vec, x_weight);
    
    // Then interpolate vertically
    // result = top_row * (1-y_frac) + bottom_row * y_frac
    uint16x8_t result = vmulq_n_u16(top_row, 255 - y_frac);
    result = vmlaq_n_u16(result, bottom_row, y_frac);
    
    // Shift right by 16 (8 bits for each weight) and narrow to 8-bit
    uint8x8_t final_color = vshrn_n_u16(result, 16);
    
    // Extract components
    Common::Vec4<u8> color;
    color.r = vget_lane_u8(final_color, 0);
    color.g = vget_lane_u8(final_color, 1);
    color.b = vget_lane_u8(final_color, 2);
    color.a = vget_lane_u8(final_color, 3);
    
    return color;
}

/**
 * Optimized nearest-neighbor texture sampling using NEON intrinsics
 * @param src Source texture data
 * @param x X coordinate (fixed-point)
 * @param y Y coordinate (fixed-point)
 * @param width Texture width
 * @param height Texture height
 * @param stride Texture stride in bytes
 * @return Sampled RGBA8 color
 */
inline Common::Vec4<u8> SampleNearest(const u8* src, u32 x, u32 y, u32 width, u32 height, u32 stride) {
    // Extract integer parts (with rounding)
    const u32 x_int = (x + 128) >> 8;
    const u32 y_int = (y + 128) >> 8;
    
    // Apply bounds checking
    const u32 x_bounded = std::min(x_int, width - 1);
    const u32 y_bounded = std::min(y_int, height - 1);
    
    // Get pointer to the pixel
    const u8* pixel = src + y_bounded * stride + x_bounded * 4;
    
    // Load pixel value (RGBA)
    Common::Vec4<u8> color;
    color.r = pixel[0];
    color.g = pixel[1];
    color.b = pixel[2];
    color.a = pixel[3];
    
    return color;
}

/**
 * Optimized batch texture sampling for multiple coordinates using NEON intrinsics
 * @param dest Destination buffer for sampled colors
 * @param src Source texture data
 * @param coords Array of (x,y) coordinates in fixed-point format
 * @param count Number of coordinates to sample
 * @param width Texture width
 * @param height Texture height
 * @param stride Texture stride in bytes
 * @param bilinear Whether to use bilinear filtering
 */
inline void SampleBatch(Common::Vec4<u8>* dest, const u8* src, const Common::Vec2<u32>* coords, 
                        size_t count, u32 width, u32 height, u32 stride, bool bilinear) {
    for (size_t i = 0; i < count; i++) {
        const u32 x = coords[i].x;
        const u32 y = coords[i].y;
        
        if (bilinear) {
            dest[i] = SampleBilinear(src, x, y, width, height, stride);
        } else {
            dest[i] = SampleNearest(src, x, y, width, height, stride);
        }
    }
}

/**
 * Optimized texture format conversion from various formats to RGBA8 using NEON
 * @param dest Destination RGBA8 buffer
 * @param src Source texture data
 * @param format Source texture format
 * @param width Texture width
 * @param height Texture height
 * @param stride Source texture stride in bytes
 */
inline void ConvertFormatToRGBA8(u8* dest, const u8* src, Pica::TextureFormat format, 
                                u32 width, u32 height, u32 stride) {
    const u32 pixel_count = width * height;
    
    switch (format) {
    case Pica::TextureFormat::RGBA8: {
        // Direct copy for RGBA8 format
        Common::Memory::NEON::FastCopy(dest, src, pixel_count * 4);
        break;
    }
    
    case Pica::TextureFormat::RGB8: {
        // Convert RGB8 to RGBA8 (set alpha to 255)
        for (u32 i = 0; i < pixel_count; i++) {
            const u8* src_pixel = src + i * 3;
            u8* dest_pixel = dest + i * 4;
            
            // Load RGB components
            uint8x8_t rgb = vld1_u8(src_pixel);
            
            // Store RGB components and set alpha to 255
            dest_pixel[0] = vget_lane_u8(rgb, 0); // R
            dest_pixel[1] = vget_lane_u8(rgb, 1); // G
            dest_pixel[2] = vget_lane_u8(rgb, 2); // B
            dest_pixel[3] = 255;                  // A
        }
        break;
    }
    
    case Pica::TextureFormat::RGB565: {
        // Convert RGB565 to RGBA8
        for (u32 i = 0; i < pixel_count; i++) {
            const u16 rgb565 = *reinterpret_cast<const u16*>(src + i * 2);
            u8* dest_pixel = dest + i * 4;
            
            // Extract RGB components
            const u8 r = (rgb565 >> 11) & 0x1F;
            const u8 g = (rgb565 >> 5) & 0x3F;
            const u8 b = rgb565 & 0x1F;
            
            // Convert to 8-bit per channel (with bit replication for precision)
            dest_pixel[0] = (r << 3) | (r >> 2);  // R
            dest_pixel[1] = (g << 2) | (g >> 4);  // G
            dest_pixel[2] = (b << 3) | (b >> 2);  // B
            dest_pixel[3] = 255;                  // A
        }
        break;
    }
    
    case Pica::TextureFormat::RGBA4: {
        // Convert RGBA4 to RGBA8
        for (u32 i = 0; i < pixel_count; i++) {
            const u16 rgba4 = *reinterpret_cast<const u16*>(src + i * 2);
            u8* dest_pixel = dest + i * 4;
            
            // Extract RGBA components
            const u8 r = (rgba4 >> 12) & 0xF;
            const u8 g = (rgba4 >> 8) & 0xF;
            const u8 b = (rgba4 >> 4) & 0xF;
            const u8 a = rgba4 & 0xF;
            
            // Convert to 8-bit per channel (with bit replication for precision)
            dest_pixel[0] = (r << 4) | r;  // R
            dest_pixel[1] = (g << 4) | g;  // G
            dest_pixel[2] = (b << 4) | b;  // B
            dest_pixel[3] = (a << 4) | a;  // A
        }
        break;
    }
    
    case Pica::TextureFormat::IA8: {
        // Convert IA8 (Intensity/Alpha) to RGBA8
        for (u32 i = 0; i < pixel_count; i++) {
            const u16 ia = *reinterpret_cast<const u16*>(src + i * 2);
            u8* dest_pixel = dest + i * 4;
            
            // Extract intensity and alpha
            const u8 i_val = (ia >> 8) & 0xFF;
            const u8 a_val = ia & 0xFF;
            
            // Set RGB to intensity, A to alpha
            dest_pixel[0] = i_val;  // R = I
            dest_pixel[1] = i_val;  // G = I
            dest_pixel[2] = i_val;  // B = I
            dest_pixel[3] = a_val;  // A
        }
        break;
    }
    
    default:
        // Fallback for unsupported formats
        std::memset(dest, 255, pixel_count * 4);  // White with alpha 255
        break;
    }
}

/**
 * Optimized texture mipmap generation using NEON intrinsics
 * @param dest Destination mipmap level
 * @param src Source (higher resolution) mipmap level
 * @param width Width of the source texture
 * @param height Height of the source texture
 * @param format Texture format (must be RGBA8)
 */
inline void GenerateMipmap(u8* dest, const u8* src, u32 width, u32 height) {
    const u32 dest_width = width / 2;
    const u32 dest_height = height / 2;
    
    for (u32 y = 0; y < dest_height; y++) {
        for (u32 x = 0; x < dest_width; x++) {
            // Calculate source pixel positions
            const u32 src_x = x * 2;
            const u32 src_y = y * 2;
            
            // Calculate pixel addresses
            const u8* p00 = src + (src_y * width + src_x) * 4;
            const u8* p01 = src + (src_y * width + src_x + 1) * 4;
            const u8* p10 = src + ((src_y + 1) * width + src_x) * 4;
            const u8* p11 = src + ((src_y + 1) * width + src_x + 1) * 4;
            
            // Load 4 pixels (RGBA)
            uint8x8_t p00_vec = vld1_u8(p00);
            uint8x8_t p01_vec = vld1_u8(p01);
            uint8x8_t p10_vec = vld1_u8(p10);
            uint8x8_t p11_vec = vld1_u8(p11);
            
            // Average horizontally first
            uint16x8_t top_row = vaddl_u8(p00_vec, p01_vec);    // p00 + p01
            uint16x8_t bottom_row = vaddl_u8(p10_vec, p11_vec); // p10 + p11
            
            // Then average vertically
            uint16x8_t sum = vaddq_u16(top_row, bottom_row);    // top_row + bottom_row
            
            // Divide by 4 (shift right by 2)
            uint8x8_t avg = vshrn_n_u16(sum, 2);
            
            // Store the result
            u8* dest_pixel = dest + (y * dest_width + x) * 4;
            vst1_u8(dest_pixel, avg);
        }
    }
}

} // namespace NEON
} // namespace TextureSampling
} // namespace Pica

#endif // defined(__ARM_NEON) || defined(__aarch64__)
