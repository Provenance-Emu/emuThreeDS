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

namespace AudioCore {

struct CoreAudioInput::Impl {
    AudioUnit audio_unit = nullptr;
    AudioStreamBasicDescription format_desc;
    u8 sample_size_in_bytes = 0;
    
    // Most recent audio buffer captured from the microphone
    std::vector<u8> current_buffer;
    std::atomic<bool> has_audio_data{false};
    
    // Track if we've shown the permission warning
    bool shown_permission_warning = false;
    
    // Counter for audio data warnings
    std::atomic<int> warning_count{0};
    
    // Mutex for thread safety when accessing the buffer
    std::mutex buffer_mutex;
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
    
    // Initialize the current buffer
    impl->current_buffer.clear();
    impl->has_audio_data = false;
    
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
    
    LOG_INFO(Audio, "CoreAudioInput now recording audio");
}

void CoreAudioInput::StopSampling() {
    LOG_INFO(Audio, "CoreAudioInput stopping sampling");
    if (impl->audio_unit) {
        LOG_INFO(Audio, "Stopping AudioUnit recording");
        AudioOutputUnitStop(impl->audio_unit);
        
        LOG_INFO(Audio, "Uninitializing and disposing AudioUnit");
        AudioUnitUninitialize(impl->audio_unit);
        AudioComponentInstanceDispose(impl->audio_unit);
        impl->audio_unit = nullptr;
    }
    
    // Clear the current buffer
    impl->current_buffer.clear();
    impl->has_audio_data = false;
    
    is_sampling = false;
    LOG_INFO(Audio, "CoreAudioInput sampling stopped");
}

bool CoreAudioInput::IsSampling() {
    LOG_TRACE(Audio, "CoreAudioInput::IsSampling() = {}", is_sampling);
    return is_sampling;
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
        
        // Lock the buffer mutex for thread safety
        std::lock_guard<std::mutex> lock(impl->buffer_mutex);
        
        // Return the current buffer if we have audio data
        if (impl->has_audio_data && !impl->current_buffer.empty()) {
            LOG_INFO(Audio, "Read: Returning {} bytes of audio data", impl->current_buffer.size());
            return impl->current_buffer;
        }
        
        // Return silence instead of a tone when no audio is available
        std::vector<u8> silence;
        const int buffer_size = 16 * impl->sample_size_in_bytes; // 16 samples
        silence.resize(buffer_size, 0);
        
        // For unsigned 8-bit audio, we need to use 128 (middle value) instead of 0
        if (parameters.sample_size == 8 && parameters.sign == Signedness::Unsigned) {
            std::fill(silence.begin(), silence.end(), 128);
        }
        
        LOG_INFO(Audio, "Read: Returning {} bytes of silence", silence.size());
        return silence;
    } catch (const std::exception& e) {
        LOG_ERROR(Audio, "Exception in Read: {}", e.what());
        return {};
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
    
    // Set up the AudioBufferList to receive the input data
    AudioBufferList buffer_list;
    buffer_list.mNumberBuffers = 1;
    buffer_list.mBuffers[0].mNumberChannels = 1;  // Mono
    buffer_list.mBuffers[0].mDataByteSize = inNumberFrames * input->impl->sample_size_in_bytes;
    
    // Sanity check the requested buffer size
    size_t requested_size = buffer_list.mBuffers[0].mDataByteSize;
    
    if (requested_size == 0) {
        if (input->impl->warning_count.fetch_add(1) % 100 == 0) {
            LOG_WARNING(Audio, "Callback: Zero-sized buffer requested, using minimum size");
        }
        requested_size = 64; // Minimum buffer size
        buffer_list.mBuffers[0].mDataByteSize = requested_size;
    }
    
    // Use a temporary buffer for audio capture
    std::vector<u8> temp_buffer(requested_size);
    buffer_list.mBuffers[0].mData = temp_buffer.data();
    
    // Lock the buffer mutex for thread safety
    std::lock_guard<std::mutex> lock(input->impl->buffer_mutex);
    
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
        input->impl->has_audio_data = false;
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
        // Check if we have non-zero data
        const u8* data = static_cast<const u8*>(buffer_list.mBuffers[0].mData);
        const size_t check_size = std::min(static_cast<size_t>(buffer_list.mBuffers[0].mDataByteSize), static_cast<size_t>(64));
        
        for (size_t i = 0; i < check_size; i += 4) {
            if (data[i] != 0) {
                has_data = true;
                break;
            }
        }
    }
    
    // Store the captured data directly
    if (has_data) {
        // Copy the data to our buffer
        input->impl->current_buffer.resize(buffer_list.mBuffers[0].mDataByteSize);
        
        // Get the raw audio data
        const u8* src_data = static_cast<const u8*>(buffer_list.mBuffers[0].mData);
        
        // Process the audio data to ensure correct format
        if (input->parameters.sample_size == 16) {
            // For 16-bit audio, ensure proper endianness
            // CoreAudio gives us native endian, but we need little endian for the emulator
            const int16_t* src_samples = reinterpret_cast<const int16_t*>(src_data);
            const size_t num_samples = buffer_list.mBuffers[0].mDataByteSize / sizeof(int16_t);
            
            for (size_t i = 0; i < num_samples; i++) {
                int16_t sample = src_samples[i];
                // Convert to little endian if needed
                input->impl->current_buffer[i*2] = sample & 0xFF;
                input->impl->current_buffer[i*2+1] = (sample >> 8) & 0xFF;
            }
        } else {
            // For 8-bit audio, just copy directly
            std::memcpy(input->impl->current_buffer.data(), src_data, buffer_list.mBuffers[0].mDataByteSize);
            
            // If unsigned is required, convert from signed to unsigned
            if (input->parameters.sign == Signedness::Unsigned) {
                for (size_t i = 0; i < input->impl->current_buffer.size(); i++) {
                    input->impl->current_buffer[i] = static_cast<u8>(static_cast<int8_t>(input->impl->current_buffer[i]) + 128);
                }
            }
        }
        
        input->impl->has_audio_data = true;
        LOG_INFO(Audio, "Callback: Captured and processed {} bytes of audio data", input->impl->current_buffer.size());
    } else {
        if (input->impl->warning_count.fetch_add(1) % 100 == 0) {
            LOG_WARNING(Audio, "Callback: No valid audio data captured (permission: {})",
                       permission_granted ? "granted" : "denied");
        }
        input->impl->has_audio_data = false;
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
