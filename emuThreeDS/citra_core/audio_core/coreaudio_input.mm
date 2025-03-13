//
//  coreaudio_input.mm
//  emuThreeDS
//
//  Created for Citra Emulator Project
//  Licensed under GPLv2 or any later version
//  Refer to the license.txt file included.
//

#include <AudioToolbox/AudioToolbox.h>
#include <AVFoundation/AVFoundation.h>
#include <utility>
#include <vector>
#include <array>
#include <atomic>
#include "audio_core/input.h"
#include "audio_core/coreaudio_input.h"
#include "common/logging/log.h"

#define USE_MUTEX 1

namespace AudioCore {

// Static noise samples to use as fallback
constexpr std::array<u8, 16> NOISE_SAMPLE_8_BIT = {0xFC, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xF5, 0xFF, 0xFF, 0xFF, 0xFF, 0x8E, 0xFF};

constexpr std::array<u8, 32> NOISE_SAMPLE_16_BIT = {
    0x64, 0x61, 0x74, 0x61, 0x56, 0xD7, 0x00, 0x00, 0x48, 0xF7, 0x86, 0x05, 0x77, 0x1A, 0xF4, 0x1F,
    0x28, 0x0F, 0x6B, 0xEB, 0x1C, 0xC0, 0xCB, 0x9D, 0x46, 0x90, 0xDF, 0x98, 0xEA, 0xAE, 0xB5, 0xC4};

// Simple fixed-size ring buffer for iOS to avoid memory allocation issues
class RingBuffer {
public:
    // Constructor with fixed size
    explicit RingBuffer(size_t capacity) : 
        capacity_(capacity),
        buffer_(new u8[capacity]),
        read_pos_(0),
        write_pos_(0),
        size_(0) {
        LOG_INFO(Audio, "Created ring buffer with capacity: {} bytes", capacity);
    }
    
    // Destructor
    ~RingBuffer() {
        delete[] buffer_;
        LOG_INFO(Audio, "Ring buffer destroyed");
    }
    
    // Write data to the buffer
    size_t Write(const u8* data, size_t length) {
        if (!buffer_ || length == 0) return 0;
        
        // Limit to available space
        const size_t available = capacity_ - size_;
        const size_t to_write = std::min(length, available);
        
        if (to_write == 0) return 0;
        
        // First chunk (from write_pos to end of buffer)
        const size_t first_chunk = std::min(to_write, capacity_ - write_pos_);
        std::memcpy(buffer_ + write_pos_, data, first_chunk);
        
        // Second chunk (wrap around to beginning of buffer if needed)
        if (first_chunk < to_write) {
            std::memcpy(buffer_, data + first_chunk, to_write - first_chunk);
        }
        
        // Update write position and size
        write_pos_ = (write_pos_ + to_write) % capacity_;
        size_ += to_write;
        
        return to_write;
    }
    
    // Read data from the buffer
    size_t Read(u8* data, size_t length) {
        if (!buffer_ || length == 0 || size_ == 0) return 0;
        
        // Limit to available data
        const size_t to_read = std::min(length, size_);
        
        // First chunk (from read_pos to end of buffer)
        const size_t first_chunk = std::min(to_read, capacity_ - read_pos_);
        std::memcpy(data, buffer_ + read_pos_, first_chunk);
        
        // Second chunk (wrap around to beginning of buffer if needed)
        if (first_chunk < to_read) {
            std::memcpy(data + first_chunk, buffer_, to_read - first_chunk);
        }
        
        // Update read position and size
        read_pos_ = (read_pos_ + to_read) % capacity_;
        size_ -= to_read;
        
        return to_read;
    }
    
    // Get a copy of all data in the buffer
    std::vector<u8> GetAll() {
        std::vector<u8> result;
        if (size_ == 0) return result;
        
        try {
            result.resize(size_);
            
            // First chunk (from read_pos to end of buffer)
            const size_t first_chunk = std::min(size_, capacity_ - read_pos_);
            std::memcpy(result.data(), buffer_ + read_pos_, first_chunk);
            
            // Second chunk (wrap around to beginning of buffer if needed)
            if (first_chunk < size_) {
                std::memcpy(result.data() + first_chunk, buffer_, size_ - first_chunk);
            }
        } catch (const std::exception& e) {
            LOG_ERROR(Audio, "Exception in RingBuffer::GetAll: {}", e.what());
            result.clear();
        }
        
        return result;
    }
    
    // Clear the buffer
    void Clear() {
        read_pos_ = 0;
        write_pos_ = 0;
        size_ = 0;
    }
    
    // Get current size
    size_t Size() const { return size_; }
    
    // Get capacity
    size_t Capacity() const { return capacity_; }
    
    // Check if buffer contains non-zero data
    bool HasNonZeroData() const {
        if (size_ == 0) return false;
        
        // Check a sample of the buffer (every 16th byte)
        const size_t step = 16;
        size_t pos = read_pos_;
        
        for (size_t i = 0; i < size_; i += step) {
            if (buffer_[pos] != 0) return true;
            pos = (pos + step) % capacity_;
            if (pos < read_pos_) {
                // We've wrapped around, adjust for the remaining count
                i += capacity_ - read_pos_;
                pos = 0;
            }
        }
        
        return false;
    }
    
private:
    const size_t capacity_;
    u8* buffer_;
    size_t read_pos_;
    size_t write_pos_;
    size_t size_;
};

struct CoreAudioInput::Impl {
    AudioUnit audio_unit = nullptr;
    AudioStreamBasicDescription format_desc;
    std::unique_ptr<RingBuffer> ring_buffer;
    bool is_recording = false;
    u8 sample_size_in_bytes = 0;
    
    // Static noise samples for fallback
    std::vector<u8> noise_8bit{NOISE_SAMPLE_8_BIT.begin(), NOISE_SAMPLE_8_BIT.end()};
    std::vector<u8> noise_16bit{NOISE_SAMPLE_16_BIT.begin(), NOISE_SAMPLE_16_BIT.end()};
    std::atomic<bool> has_real_data{false};
    
    // Track if we've shown the permission warning
    bool shown_permission_warning = false;
    
    // Counter for audio data warnings
    std::atomic<int> warning_count{0};
    
    // Fixed buffer sizes for iOS
    static constexpr size_t kDefaultBufferSize = 32 * 1024;  // 32KB default
    static constexpr size_t kMinBufferSize = 4 * 1024;       // 4KB minimum
    
    // Pre-allocate the ring buffer with a fixed size
    Impl() {
        LOG_INFO(Audio, "CoreAudioInput::Impl constructor - creating fixed-size ring buffer");
        try {
            ring_buffer = std::make_unique<RingBuffer>(kDefaultBufferSize);
            LOG_INFO(Audio, "Successfully created ring buffer with capacity: {} bytes", kDefaultBufferSize);
        } catch (const std::exception& e) {
            LOG_ERROR(Audio, "Failed to create default ring buffer: {}, trying smaller size", e.what());
            try {
                ring_buffer = std::make_unique<RingBuffer>(kMinBufferSize);
                LOG_INFO(Audio, "Successfully created fallback ring buffer with capacity: {} bytes", kMinBufferSize);
            } catch (const std::exception& e) {
                LOG_CRITICAL(Audio, "Failed to create even minimal ring buffer: {}", e.what());
                // We'll handle the null ring_buffer case in the code
            }
        }
    }
    
    // Mutex for thread safety when accessing the buffer
#if USE_MUTEX
    std::mutex buffer_mutex;
#endif
};


CoreAudioInput::CoreAudioInput(std::string device_id)
: impl(std::make_unique<Impl>()), device_id(std::move(device_id)) {
    LOG_INFO(Audio, "CoreAudioInput initialized with device: {}", this->device_id);
}

CoreAudioInput::~CoreAudioInput() {
    LOG_INFO(Audio, "CoreAudioInput being destroyed");
    try {
        StopSampling();
    } catch (const std::exception& e) {
        LOG_ERROR(Audio, "Exception during CoreAudioInput destruction: {}", e.what());
        // Don't rethrow in destructor
    } catch (...) {
        LOG_ERROR(Audio, "Unknown exception during CoreAudioInput destruction");
        // Don't rethrow in destructor
    }
}

void CoreAudioInput::StartSampling(const InputParameters& params) {
    LOG_INFO(Audio, "CoreAudioInput starting sampling: sample_rate={}, sample_size={}, buffer_size={}",
             params.sample_rate, params.sample_size, params.buffer_size);
    if (IsSampling()) {
        LOG_INFO(Audio, "CoreAudioInput already sampling, ignoring StartSampling call");
        return;
    }
    
    parameters = params;
    impl->sample_size_in_bytes = params.sample_size / 8;
    is_sampling = true;
    
    // Make sure we have a valid ring buffer
    if (!impl->ring_buffer) {
        LOG_WARNING(Audio, "Ring buffer not initialized, creating a new one");
        try {
            impl->ring_buffer = std::make_unique<RingBuffer>(impl->kDefaultBufferSize);
            LOG_INFO(Audio, "Created new ring buffer with capacity: {} bytes", impl->kDefaultBufferSize);
        } catch (const std::exception& e) {
            LOG_ERROR(Audio, "Failed to create ring buffer: {}, trying minimal size", e.what());
            try {
                impl->ring_buffer = std::make_unique<RingBuffer>(impl->kMinBufferSize);
                LOG_INFO(Audio, "Created minimal ring buffer with capacity: {} bytes", impl->kMinBufferSize);
            } catch (const std::exception& e) {
                LOG_CRITICAL(Audio, "Failed to create even minimal ring buffer: {}, audio input will not work", e.what());
                StopSampling();
                return;
            }
        }
    } else {
        // Clear any existing data in the ring buffer
        impl->ring_buffer->Clear();
        LOG_INFO(Audio, "Cleared existing ring buffer");
    }
    
    OSStatus err;
    AudioComponentDescription acdesc = {
        .componentType = kAudioUnitType_Output,
        .componentSubType = kAudioUnitSubType_RemoteIO,
        .componentManufacturer = kAudioUnitManufacturer_Apple,
        .componentFlags = 0,
        .componentFlagsMask = 0,
    };
    
    LOG_INFO(Audio, "Finding AudioComponent for RemoteIO");
    AudioComponent component = AudioComponentFindNext(nullptr, &acdesc);
    if (component == nullptr) {
        LOG_CRITICAL(Audio, "Failed to find Input AudioComponent");
        StopSampling();
        return;
    }
    LOG_INFO(Audio, "AudioComponent found successfully");
    
    err = AudioComponentInstanceNew(component, &impl->audio_unit);
    if (err != noErr) {
        LOG_CRITICAL(Audio, "Failed to create new AudioComponent instance: {}", err);
        StopSampling();
        return;
    }
    
    // Enable input on the input scope of bus 1
    UInt32 enable_io = 1;
    err = AudioUnitSetProperty(impl->audio_unit, kAudioOutputUnitProperty_EnableIO,
                               kAudioUnitScope_Input, 1, &enable_io, sizeof(enable_io));
    if (err != noErr) {
        LOG_CRITICAL(Audio, "Failed to enable input: {}", err);
        StopSampling();
        return;
    }
    
    // Disable output on the output scope of bus 0 (we only want input)
    enable_io = 0;
    err = AudioUnitSetProperty(impl->audio_unit, kAudioOutputUnitProperty_EnableIO,
                               kAudioUnitScope_Output, 0, &enable_io, sizeof(enable_io));
    if (err != noErr) {
        LOG_CRITICAL(Audio, "Failed to disable output: {}", err);
        StopSampling();
        return;
    }
    
    // Configure the audio session for input
    LOG_INFO(Audio, "Configuring AVAudioSession for recording");
    NSError* error = nil;
    AVAudioSession* session = [AVAudioSession sharedInstance];
    
    // Check if microphone permission is granted and request if needed
    AVAudioSessionRecordPermission permission = [AVAudioSession sharedInstance].recordPermission;
    
    if (permission == AVAudioSessionRecordPermissionUndetermined) {
        LOG_WARNING(Audio, "Microphone permission not determined, requesting permission");
        
        // Create a dispatch semaphore to wait for the permission callback
        dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
        
        // Request permission synchronously
        [[AVAudioSession sharedInstance] requestRecordPermission:^(BOOL granted) {
            if (!granted) {
                LOG_WARNING(Audio, "Microphone permission denied, will use static noise fallback");
                impl->shown_permission_warning = true;
            } else {
                LOG_INFO(Audio, "Microphone permission granted");
            }
            dispatch_semaphore_signal(semaphore);
        }];
        
        // Wait for up to 1 second for the permission result
        dispatch_semaphore_wait(semaphore, dispatch_time(DISPATCH_TIME_NOW, 1 * NSEC_PER_SEC));
    } else if (permission == AVAudioSessionRecordPermissionDenied && !impl->shown_permission_warning) {
        LOG_WARNING(Audio, "Microphone permission denied, will use static noise fallback");
        impl->shown_permission_warning = true;
    }
    
    // Use PlayAndRecord with additional options to improve microphone capture
    [session setCategory:AVAudioSessionCategoryPlayAndRecord
             withOptions:AVAudioSessionCategoryOptionDefaultToSpeaker |
     AVAudioSessionCategoryOptionMixWithOthers |
     AVAudioSessionCategoryOptionAllowBluetooth
                   error:&error];
    if (error) {
        LOG_CRITICAL(Audio, "Failed to set audio session category: {}", [[error localizedDescription] UTF8String]);
        StopSampling();
        return;
    }
    LOG_INFO(Audio, "AVAudioSession category set to PlayAndRecord with additional options");
    
    // Set the preferred input to the built-in microphone
    AVAudioSessionPortDescription* preferredInput = nil;
    for (AVAudioSessionPortDescription* port in [[AVAudioSession sharedInstance] availableInputs]) {
        if ([port.portType isEqualToString:AVAudioSessionPortBuiltInMic]) {
            preferredInput = port;
            break;
        }
    }
    
    if (preferredInput) {
        [[AVAudioSession sharedInstance] setPreferredInput:preferredInput error:&error];
        if (error) {
            LOG_WARNING(Audio, "Failed to set preferred input: {}", [[error localizedDescription] UTF8String]);
            // Continue anyway
            error = nil;
        } else {
            LOG_INFO(Audio, "Set preferred input to built-in microphone");
        }
    }
    
    // Set audio session active
    [session setActive:YES error:&error];
    if (error) {
        LOG_CRITICAL(Audio, "Failed to activate audio session: {}", [[error localizedDescription] UTF8String]);
        StopSampling();
        return;
    }
    LOG_INFO(Audio, "AVAudioSession activated successfully");
    
    // Set up the audio format
    FillOutASBDForLPCM(impl->format_desc,
                       static_cast<Float64>(params.sample_rate),  // Sample rate
                       1,                                         // Channels (mono)
                       params.sample_size,                        // Bits per channel
                       params.sample_size,                        // Bits per frame
                       false,                                     // Non-interleaved
                       params.sign == Signedness::Signed);        // Signed or unsigned
    
    err = AudioUnitSetProperty(impl->audio_unit, kAudioUnitProperty_StreamFormat,
                               kAudioUnitScope_Output, 1, &impl->format_desc,
                               sizeof(impl->format_desc));
    if (err != noErr) {
        LOG_CRITICAL(Audio, "Failed to set stream format: {}", err);
        StopSampling();
        return;
    }
    
    // Set up the callback
    AURenderCallbackStruct callback = {
        .inputProc = AudioInputCallback,
        .inputProcRefCon = this,
    };
    
    err = AudioUnitSetProperty(impl->audio_unit, kAudioOutputUnitProperty_SetInputCallback,
                               kAudioUnitScope_Global, 0, &callback, sizeof(callback));
    if (err != noErr) {
        LOG_CRITICAL(Audio, "Failed to set input callback: {}", err);
        StopSampling();
        return;
    }
    
    // Initialize and start the audio unit
    LOG_INFO(Audio, "Initializing AudioUnit");
    err = AudioUnitInitialize(impl->audio_unit);
    if (err != noErr) {
        LOG_CRITICAL(Audio, "Failed to initialize AudioUnit: {}", err);
        StopSampling();
        return;
    }
    LOG_INFO(Audio, "AudioUnit initialized successfully");
    
    LOG_INFO(Audio, "Starting AudioUnit for recording");
    err = AudioOutputUnitStart(impl->audio_unit);
    if (err != noErr) {
        LOG_CRITICAL(Audio, "Failed to start AudioUnit: {}", err);
        StopSampling();
        return;
    }
    
    impl->is_recording = true;
    LOG_INFO(Audio, "CoreAudioInput now recording audio");
}

void CoreAudioInput::StopSampling() {
    LOG_INFO(Audio, "CoreAudioInput stopping sampling");
    if (impl->audio_unit) {
        if (impl->is_recording) {
            LOG_INFO(Audio, "Stopping AudioUnit recording");
            AudioOutputUnitStop(impl->audio_unit);
            impl->is_recording = false;
        }
        
        LOG_INFO(Audio, "Uninitializing and disposing AudioUnit");
        AudioUnitUninitialize(impl->audio_unit);
        AudioComponentInstanceDispose(impl->audio_unit);
        impl->audio_unit = nullptr;
    }
    
    // Clear the ring buffer to free memory
    if (impl->ring_buffer) {
        impl->ring_buffer->Clear();
        LOG_INFO(Audio, "Cleared ring buffer");
    }
    
    is_sampling = false;
    LOG_INFO(Audio, "CoreAudioInput sampling stopped");
}

bool CoreAudioInput::IsSampling() {
    bool result = is_sampling && impl->is_recording;
    LOG_TRACE(Audio, "CoreAudioInput::IsSampling() = {}", result);
    return result;
}

void CoreAudioInput::AdjustSampleRate(u32 sample_rate) {
    LOG_INFO(Audio, "CoreAudioInput adjusting sample rate to: {}", sample_rate);
    if (!IsSampling()) {
        LOG_INFO(Audio, "Not currently sampling, ignoring sample rate adjustment");
        return;
    }
    
    auto new_params = parameters;
    new_params.sample_rate = sample_rate;
    StopSampling();
    StartSampling(new_params);
}

Samples CoreAudioInput::Read() {
    LOG_TRACE(Audio, "CoreAudioInput::Read() called");
    try {
        if (!IsSampling()) {
            LOG_TRACE(Audio, "Not sampling, returning empty buffer");
            return {};
        }
        
        // If we don't have a valid ring buffer, return static noise
        if (!impl->ring_buffer) {
            LOG_WARNING(Audio, "Read: No valid ring buffer, returning static noise");
            return (parameters.sample_size == 8) ? impl->noise_8bit : impl->noise_16bit;
        }
        
#if USE_MUTEX
        // Lock the buffer mutex to safely access the buffer
        std::lock_guard<std::mutex> lock(impl->buffer_mutex);
#endif
        
        // If we don't have real audio data, return static noise instead
        if (!impl->has_real_data) {
            LOG_INFO(Audio, "Read: No real audio data, returning static noise (sample size: {})", parameters.sample_size);
            return (parameters.sample_size == 8) ? impl->noise_8bit : impl->noise_16bit;
        }
        
        try {
            // Get all data from the ring buffer
            std::vector<u8> data = impl->ring_buffer->GetAll();
            LOG_INFO(Audio, "Read: Got {} bytes of audio data from ring buffer", data.size());
            
            // If we got no data, return static noise
            if (data.empty()) {
                LOG_INFO(Audio, "Read: Ring buffer returned empty data, using static noise");
                return (parameters.sample_size == 8) ? impl->noise_8bit : impl->noise_16bit;
            }
            
            return data;
        } catch (const std::exception& e) {
            // If getting data from the ring buffer fails, return static noise instead
            LOG_ERROR(Audio, "Read: Exception when getting data from ring buffer: {}, using static noise", e.what());
            return (parameters.sample_size == 8) ? impl->noise_8bit : impl->noise_16bit;
        } catch (...) {
            LOG_ERROR(Audio, "Read: Unknown exception when getting data from ring buffer, using static noise");
            return (parameters.sample_size == 8) ? impl->noise_8bit : impl->noise_16bit;
        }
    } catch (const std::exception& e) {
        LOG_ERROR(Audio, "Read: Outer exception: {}, returning static noise", e.what());
        return (parameters.sample_size == 8) ? impl->noise_8bit : impl->noise_16bit;
    } catch (...) {
        LOG_ERROR(Audio, "Read: Unknown outer exception, returning static noise");
        return (parameters.sample_size == 8) ? impl->noise_8bit : impl->noise_16bit;
    }
}

OSStatus CoreAudioInput::AudioInputCallback(void* inRefCon,
                                             AudioUnitRenderActionFlags* ioActionFlags,
                                             const AudioTimeStamp* inTimeStamp,
                                             UInt32 inBusNumber,
                                             UInt32 inNumberFrames,
                                             AudioBufferList* ioData) {
    LOG_TRACE(Audio, "CoreAudioInput callback: bus={}, frames={}", inBusNumber, inNumberFrames);
    CoreAudioInput* input = static_cast<CoreAudioInput*>(inRefCon);
    
    if (!input || !input->IsSampling()) {
        LOG_TRACE(Audio, "Input invalid or not sampling, skipping callback");
        return noErr;
    }
    
    // Make sure we have a valid ring buffer
    if (!input->impl->ring_buffer) {
        LOG_ERROR(Audio, "Callback: No valid ring buffer available");
        input->impl->has_real_data = false;
        return -1;
    }
    
    // Set up the AudioBufferList to receive the input data
    AudioBufferList buffer_list;
    buffer_list.mNumberBuffers = 1;
    buffer_list.mBuffers[0].mNumberChannels = 1;  // Mono
    buffer_list.mBuffers[0].mDataByteSize = inNumberFrames * input->impl->sample_size_in_bytes;
    
    // Sanity check the requested buffer size
    size_t requested_size = buffer_list.mBuffers[0].mDataByteSize;
    
    if (requested_size == 0) {
        // Only log every 100th warning to reduce spam
        if (input->impl->warning_count.fetch_add(1) % 100 == 0) {
            LOG_WARNING(Audio, "Callback: Zero-sized buffer requested, using minimum size");
        }
        requested_size = 64; // Minimum buffer size
        buffer_list.mBuffers[0].mDataByteSize = requested_size;
    }
    
    LOG_INFO(Audio, "Callback: Allocating temporary buffer of {} bytes", requested_size);
    
    // Use a temporary buffer for audio capture to avoid memory issues
    std::unique_ptr<u8[]> temp_buffer;
    try {
        temp_buffer = std::make_unique<u8[]>(requested_size);
        buffer_list.mBuffers[0].mData = temp_buffer.get();
    } catch (const std::exception& e) {
        LOG_ERROR(Audio, "Callback: Failed to allocate temporary buffer: {}", e.what());
        input->impl->has_real_data = false;
        return -1;
    }
    
#if USE_MUTEX
    // Lock the buffer mutex for thread safety
    std::lock_guard<std::mutex> lock(input->impl->buffer_mutex);
#endif
    
    // Render the audio data
    LOG_TRACE(Audio, "Rendering audio data from input unit");
    OSStatus err = AudioUnitRender(input->impl->audio_unit,
                                   ioActionFlags,
                                   inTimeStamp,
                                   inBusNumber,
                                   inNumberFrames,
                                   &buffer_list);
    
    if (err != noErr) {
        LOG_WARNING(Audio, "Callback: AudioUnitRender failed: {}", err);
        input->impl->has_real_data = false;
        return err;
    }
    
    LOG_TRACE(Audio, "Successfully captured {} bytes of audio data", buffer_list.mBuffers[0].mDataByteSize);
    
    // Check if we have actual audio data
    bool permission_granted = false;
    bool has_data = false;
    
    try {
        permission_granted = [AVAudioSession sharedInstance].recordPermission == AVAudioSessionRecordPermissionGranted;
    } catch (...) {
        permission_granted = false;
    }
    
    // Only check for non-zero data if we have permission
    if (permission_granted) {
        // Check a sample of the buffer to see if we have non-zero data
        const u8* data = static_cast<const u8*>(buffer_list.mBuffers[0].mData);
        const size_t check_size = std::min(static_cast<size_t>(buffer_list.mBuffers[0].mDataByteSize), static_cast<size_t>(64));
        
        for (size_t i = 0; i < check_size; i += 4) {
            if (data[i] != 0) {
                has_data = true;
                break;
            }
        }
        
        // If we still don't have data but have permission, add some minimal non-zero data
        if (!has_data) {
            // Only log every 100th time to reduce spam
            if (input->impl->warning_count.fetch_add(1) % 100 == 0) {
                LOG_WARNING(Audio, "Callback: No actual audio data captured, adding minimal non-zero data");
            }
            
            // Add a small non-zero value to create minimal audio data
            if (buffer_list.mBuffers[0].mDataByteSize > 0) {
                u8* mutable_buffer = static_cast<u8*>(buffer_list.mBuffers[0].mData);
                mutable_buffer[0] = 1;
                has_data = true;
            }
        }
    }
    
    // Update the real data flag
    input->impl->has_real_data = has_data;
    
    // Write the captured data to our ring buffer
    if (has_data) {
        size_t bytes_written = input->impl->ring_buffer->Write(
            static_cast<const u8*>(buffer_list.mBuffers[0].mData),
            buffer_list.mBuffers[0].mDataByteSize);
        
        LOG_INFO(Audio, "Callback: Wrote {} bytes to ring buffer", bytes_written);
        
        if (bytes_written < buffer_list.mBuffers[0].mDataByteSize) {
            // Only log every 100th warning to reduce spam
            if (input->impl->warning_count.fetch_add(1) % 100 == 0) {
                LOG_WARNING(Audio, "Callback: Ring buffer full, only wrote {} of {} bytes",
                            bytes_written, buffer_list.mBuffers[0].mDataByteSize);
            }
        }
    } else {
        // Only log every 100th warning to reduce spam
        if (input->impl->warning_count.fetch_add(1) % 100 == 0) {
            LOG_WARNING(Audio, "Callback: No valid audio data to write to ring buffer (permission: {})",
                        permission_granted ? "granted" : "denied");
        }
    }
    
    return noErr;
}

std::vector<std::string> ListCoreAudioInputDevices() {
    LOG_INFO(Audio, "Listing CoreAudio input devices");
    std::vector<std::string> device_list;
    
    // On iOS, we only have the default microphone
    device_list.push_back("auto");
    device_list.push_back("Built-in Microphone");
    
    LOG_INFO(Audio, "Found {} CoreAudio input devices", device_list.size());
    return device_list;
}

} // namespace AudioCore
