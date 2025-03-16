// Copyright 2025 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

// This header serves as a central point for including all NEON-optimized implementations for audio processing
// It automatically selects the appropriate implementation based on the target architecture

// Only include NEON optimizations on ARM64 platforms
#if defined(__ARM_NEON) || defined(__aarch64__)

// Include all NEON-optimized headers for audio processing
#include "audio_core/hle/mixer_neon.h"
#include "audio_core/interpolate_neon.h"
#include "audio_core/time_stretch_neon.h"
#include "audio_core/dsp_interface_neon.h"

// Define a macro to indicate that NEON optimizations are available
#define CITRA_AUDIO_NEON_OPTIMIZATIONS_ENABLED 1

#else

// Define a macro to indicate that NEON optimizations are not available
#define CITRA_AUDIO_NEON_OPTIMIZATIONS_ENABLED 0

#endif // defined(__ARM_NEON) || defined(__aarch64__)
