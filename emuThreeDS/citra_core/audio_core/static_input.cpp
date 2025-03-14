// Copyright 2019 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <array>
#include <random>
#include <ctime>
#include "audio_core/input.h"
#include "audio_core/static_input.h"
#include "common/logging/log.h"

namespace AudioCore {

StaticInput::StaticInput() {
    // Initialize the random number generator
    std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr)));
    
    // For 8-bit unsigned static noise (centered around 128)
    std::uniform_int_distribution<> dist_8bit(80, 176); // Reduced range for less harsh noise
    
    // For 16-bit signed static noise (centered around 0)
    std::uniform_int_distribution<> dist_16bit(-8192, 8191); // Reduced range for less harsh noise
    
    // Generate 8-bit static (1/4 second at 48000Hz mono)
    const size_t sample_count_8bit = 12000;
    CACHE_8_BIT.resize(sample_count_8bit);
    for (size_t i = 0; i < sample_count_8bit; i++) {
        CACHE_8_BIT[i] = static_cast<u8>(dist_8bit(rng));
    }
    
    // Generate 16-bit static (1/4 second at 48000Hz mono)
    const size_t sample_count_16bit = 12000;
    CACHE_16_BIT.resize(sample_count_16bit * 2); // 2 bytes per sample
    
    for (size_t i = 0; i < sample_count_16bit; i++) {
        s16 sample = static_cast<s16>(dist_16bit(rng));
        // Store in little-endian format
        CACHE_16_BIT[i*2] = sample & 0xFF;
        CACHE_16_BIT[i*2+1] = (sample >> 8) & 0xFF;
    }
    
    LOG_INFO(Audio, "Generated static noise: {} bytes for 8-bit, {} bytes for 16-bit",
             CACHE_8_BIT.size(), CACHE_16_BIT.size());
}

} // namespace AudioCore
