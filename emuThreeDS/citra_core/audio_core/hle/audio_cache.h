#pragma once

#include <array>
#include <memory>
#include <unordered_map>
#include <vector>
#include "audio_core/audio_types.h"
#include "audio_core/hle/common.h"
#include "common/common_types.h"

namespace AudioCore::HLE {

/**
 * Audio sample cache to avoid redundant decoding of frequently used audio samples.
 * This is particularly useful for looping sounds and frequently played sound effects.
 */
class AudioSampleCache {
public:
    AudioSampleCache() = default;
    ~AudioSampleCache() = default;

    /**
     * Get a cached buffer for the given buffer ID.
     * @param buffer_id The unique ID of the buffer to retrieve.
     * @return Pointer to the cached buffer, or nullptr if not found.
     */
    std::shared_ptr<StereoBuffer16> GetBuffer(u32 buffer_id) const {
        auto it = cache.find(buffer_id);
        if (it != cache.end()) {
            return it->second;
        }
        return std::shared_ptr<StereoBuffer16>(nullptr);
    }

    /**
     * Cache a buffer with the given ID.
     * @param buffer_id The unique ID to associate with the buffer.
     * @param buffer The buffer to cache.
     */
    void CacheBuffer(u32 buffer_id, const StereoBuffer16& buffer) {
        // Don't cache very small buffers as the overhead isn't worth it
        if (buffer.size() < 16) {
            return;
        }

        // If the cache is getting too large, remove the oldest entries
        if (cache.size() >= max_cache_size) {
            PruneCache();
        }

        auto shared_buffer = std::make_shared<StereoBuffer16>(buffer);
        cache[buffer_id] = shared_buffer;
        access_order.push_back(buffer_id);
    }

    /**
     * Clear the entire cache.
     */
    void Clear() {
        cache.clear();
        access_order.clear();
    }

    /**
     * Update the access time for a buffer to keep it in the cache longer.
     * @param buffer_id The ID of the buffer that was accessed.
     */
    void UpdateAccessTime(u32 buffer_id) {
        // Remove the ID from its current position
        auto it = std::find(access_order.begin(), access_order.end(), buffer_id);
        if (it != access_order.end()) {
            access_order.erase(it);
        }
        // Add it to the end (most recently used)
        access_order.push_back(buffer_id);
    }

private:
    /**
     * Remove the least recently used entries when the cache gets too large.
     */
    void PruneCache() {
        // Remove 1/4 of the cache (the oldest entries)
        size_t entries_to_remove = max_cache_size / 4;
        for (size_t i = 0; i < entries_to_remove && !access_order.empty(); i++) {
            u32 oldest_id = access_order.front();
            access_order.erase(access_order.begin());
            cache.erase(oldest_id);
        }
    }

    // Maximum number of buffers to keep in the cache
    static constexpr size_t max_cache_size = 128;

    // The actual cache storage
    std::unordered_map<u32, std::shared_ptr<StereoBuffer16>> cache;

    // Tracks the order of buffer access (front = oldest, back = newest)
    std::vector<u32> access_order;
};

// Global audio sample cache instance
extern AudioSampleCache g_audio_sample_cache;

} // namespace AudioCore::HLE
