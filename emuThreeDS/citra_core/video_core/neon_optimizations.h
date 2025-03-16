// Copyright 2025 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

// This header serves as a central point for including all NEON-optimized implementations
// It automatically selects the appropriate implementation based on the target architecture

// Only include NEON optimizations on ARM64 platforms
#if defined(__ARM_NEON) || defined(__aarch64__)

// Include all NEON-optimized headers
#include "video_core/rasterizer_neon.h"
#include "video_core/swizzle_neon.h"
#include "video_core/vertex_processor_neon.h"
#include "video_core/shader_processing_neon.h"
#include "video_core/color_blending_neon.h"
#include "common/vector_math_neon.h"
#include "common/memory_ops_neon.h"

// Define a macro to indicate that NEON optimizations are available
#define CITRA_NEON_OPTIMIZATIONS_ENABLED 1

#else

// Define a macro to indicate that NEON optimizations are not available
#define CITRA_NEON_OPTIMIZATIONS_ENABLED 0

#endif // defined(__ARM_NEON) || defined(__aarch64__)
