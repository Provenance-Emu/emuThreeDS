// Copyright 2023 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/rasterizer_cache/surface_params.h"
#include "video_core/rasterizer_cache/texture_codec.h"
#include "video_core/rasterizer_cache/utils.h"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace VideoCore {

u32 MipLevels(u32 width, u32 height, u32 max_level) {
    u32 levels = 1;
    while (width > 8 && height > 8) {
        levels++;
        width >>= 1;
        height >>= 1;
    }

    return std::min(levels, max_level + 1);
}

void EncodeTexture(const SurfaceParams& surface_info, PAddr start_addr, PAddr end_addr,
                   std::span<u8> source, std::span<u8> dest, bool convert) {
    const PixelFormat format = surface_info.pixel_format;
    const u32 func_index = static_cast<u32>(format);

    /// Fast path for non-tiled textures using NEON
    if (!surface_info.is_tiled) {
        const LinearFunc LinearEncodeImpl =
            (convert ? LINEAR_ENCODE_TABLE_CONVERTED : LINEAR_ENCODE_TABLE)[func_index];
        if (LinearEncodeImpl) {
            /// Check if we have enough data to use SIMD
            const size_t size = dest.size();
            if (size >= 16) {  // At least one NEON vector
                const size_t vector_size = 16;
                const size_t aligned_size = size & ~(vector_size - 1);

                /// Process 16 bytes at a time using NEON
                for (size_t i = 0; i < aligned_size; i += vector_size) {
                    uint8x16_t pixels = vld1q_u8(source.data() + i);
                    vst1q_u8(dest.data() + i, pixels);
                }

                /// Handle remaining bytes with original implementation
                if (aligned_size < size) {
                    LinearEncodeImpl(source.subspan(aligned_size), dest.subspan(aligned_size));
                }
                return;
            }
            LinearEncodeImpl(source, dest);
            return;
        }
    } else {
        const MortonFunc SwizzleImpl =
            (convert ? SWIZZLE_TABLE_CONVERTED : SWIZZLE_TABLE)[func_index];
        if (SwizzleImpl) {
            SwizzleImpl(surface_info.width, surface_info.height, start_addr - surface_info.addr,
                        end_addr - surface_info.addr, source, dest);
            return;
        }
    }

    LOG_ERROR(HW_GPU, "Unimplemented texture encode function for pixel format = {}, tiled = {}",
              func_index, surface_info.is_tiled);
    UNIMPLEMENTED();
}

void DecodeTexture(const SurfaceParams& surface_info, PAddr start_addr, PAddr end_addr,
                   std::span<u8> source, std::span<u8> dest, bool convert) {
    const PixelFormat format = surface_info.pixel_format;
    const u32 func_index = static_cast<u32>(format);

    if (surface_info.is_tiled) {
        const MortonFunc UnswizzleImpl =
            (convert ? UNSWIZZLE_TABLE_CONVERTED : UNSWIZZLE_TABLE)[func_index];
        if (UnswizzleImpl) {
            UnswizzleImpl(surface_info.width, surface_info.height, start_addr - surface_info.addr,
                          end_addr - surface_info.addr, dest, source);
            return;
        }
    } else {
        const LinearFunc LinearDecodeImpl =
            (convert ? LINEAR_DECODE_TABLE_CONVERTED : LINEAR_DECODE_TABLE)[func_index];
        if (LinearDecodeImpl) {
            LinearDecodeImpl(source, dest);
            return;
        }
    }

    LOG_ERROR(HW_GPU, "Unimplemented texture decode function for pixel format = {}, tiled = {}",
              func_index, surface_info.is_tiled);
    UNIMPLEMENTED();
}

} // namespace VideoCore
