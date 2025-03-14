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
#include "common/threadsafe_queue.h"

namespace AudioCore {

struct CoreAudioInput::Impl {
    AudioUnit audio_unit = nullptr;
    AudioStreamBasicDescription format_desc;
    u8 sample_size_in_bytes = 0;

    // Queue for audio samples
    Common::SPSCQueue<Samples> sample_queue{};

    // Track if we've shown the permission warning
    bool shown_permission_warning = false;

    // Counter for audio data warnings
    std::atomic<int> warning_count{0};

    // Sample rate conversion info
    u32 actual_sample_rate = 48000; // iOS typically uses 48kHz
    u32 target_sample_rate = 0;     // Target sample rate from parameters

    // Audio session configuration state
    bool audio_session_initialized = false;

    // Signedness of the audio format
    Signedness sign = Signedness::Signed;
};

// Simple linear interpolation for sample rate conversion
template <typename T>
std::vector<T> ResampleAudio(const std::vector<T>& input, u32 input_rate, u32 output_rate) {
    if (input_rate == output_rate || input.empty()) {
        return input;
    }

    const double ratio = static_cast<double>(input_rate) / output_rate;
    const size_t output_size = static_cast<size_t>(std::ceil(input.size() / ratio));
    std::vector<T> output(output_size);

    for (size_t i = 0; i < output_size; i++) {
        const double input_idx = i * ratio;
        const size_t input_idx_floor = static_cast<size_t>(input_idx);
        const size_t input_idx_ceil = std::min(input_idx_floor + 1, input.size() - 1);
        const double frac = input_idx - input_idx_floor;

        // Linear interpolation
        if constexpr (std::is_integral_v<T>) {
            // For integer types (like u8 or s16)
            output[i] = static_cast<T>(
                input[input_idx_floor] * (1.0 - frac) +
                input[input_idx_ceil] * frac);
        } else {
            // For floating point types
            output[i] = static_cast<T>(
                static_cast<double>(input[input_idx_floor]) * (1.0 - frac) +
                static_cast<double>(input[input_idx_ceil]) * frac);
        }
    }

    return output;
}


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

    // Clear any existing samples in the queue
    Samples dummy;
    while (impl->sample_queue.Pop(dummy)) {
        // Just drain the queue
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

    // Configure the audio session for input, but only if not already initialized
    if (!impl->audio_session_initialized) {
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
                    LOG_WARNING(Audio, "Microphone permission denied, will use silence fallback");
                    impl->shown_permission_warning = true;
                } else {
                    LOG_INFO(Audio, "Microphone permission granted");
                }
                dispatch_semaphore_signal(semaphore);
            }];

            // Wait for up to 1 second for the permission result
            dispatch_semaphore_wait(semaphore, dispatch_time(DISPATCH_TIME_NOW, 1 * NSEC_PER_SEC));
        } else if (permission == AVAudioSessionRecordPermissionDenied && !impl->shown_permission_warning) {
            LOG_WARNING(Audio, "Microphone permission denied, will use silence fallback");
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

        // Mark the audio session as initialized to avoid redundant calls
        impl->audio_session_initialized = true;
    } else {
        LOG_INFO(Audio, "AVAudioSession already initialized, skipping configuration");
    }

    // Store the target sample rate and signedness from parameters
    impl->target_sample_rate = params.sample_rate;
    impl->sign = params.sign;

    // Set up the audio format based on 3DS requirements
    // The 3DS uses unsigned 8-bit PCM or signed 16-bit PCM
    // Always use iOS's native 48kHz sample rate for capture and resample later

    // For 3DS: PCM8 = Unsigned 8-bit, PCM16 = Unsigned 16-bit
    //          PCM8Signed = Signed 8-bit, PCM16Signed = Signed 16-bit
    bool is_signed = params.sign == Signedness::Signed;

    // CoreAudio on iOS works best with signed 16-bit PCM
    // We'll convert to the requested format later if needed
    FillOutASBDForLPCM(impl->format_desc,
                       static_cast<Float64>(impl->actual_sample_rate),  // Use actual iOS sample rate (48kHz)
                       1,                                               // Channels (mono)
                       16,                                              // Always use 16-bit for capture
                       16,                                              // Bits per frame
                       false,                                           // Non-interleaved
                       true);                                           // Always signed for capture

    LOG_INFO(Audio, "Using actual sample rate {} Hz (will resample to {} Hz), format: {}-bit {}",
             impl->actual_sample_rate, impl->target_sample_rate,
             params.sample_size, is_signed ? "signed" : "unsigned");

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

    // Clear any samples in the queue
    Samples dummy;
    while (impl->sample_queue.Pop(dummy)) {
        // Just drain the queue
    }

    is_sampling = false;

    // Note: We intentionally don't reset audio_session_initialized here
    // to avoid unnecessary reconfiguration of the AVAudioSession
    LOG_INFO(Audio, "CoreAudioInput sampling stopped (keeping AVAudioSession configuration)");
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

        // Collect all available samples from the queue
        Samples all_samples;
        Samples queue_chunk;

        // Drain the queue and combine all samples
        while (impl->sample_queue.Pop(queue_chunk)) {
            all_samples.insert(all_samples.end(), queue_chunk.begin(), queue_chunk.end());
        }

        // If we have samples, process them
        if (!all_samples.empty()) {
            LOG_INFO(Audio, "Read: Collected {} bytes of audio data from queue", all_samples.size());
            return all_samples;
        }

        // Return silence when no audio is available
        // Create a properly sized buffer of zeros based on the expected format
        const size_t expected_sample_count = impl->target_sample_rate / 100; // ~10ms of audio
        const size_t silence_size = expected_sample_count * impl->sample_size_in_bytes;

        LOG_INFO(Audio, "Read: No audio data available, returning silence buffer of {} bytes", silence_size);

        Samples silence_buffer(silence_size, 0); // Create a zero-filled buffer
        return silence_buffer;
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
    // Properly allocate AudioBufferList with the correct size for 1 buffer
    AudioBufferList buffer_list;
    memset(&buffer_list, 0, sizeof(buffer_list));
    buffer_list.mNumberBuffers = 1;
    buffer_list.mBuffers[0].mNumberChannels = 1;  // Mono
    buffer_list.mBuffers[0].mDataByteSize = inNumberFrames * 2; // Always 16-bit (2 bytes) for capture

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
    // Make sure the buffer stays in scope until after AudioUnitRender completes
    std::vector<u8> temp_buffer(requested_size, 0); // Initialize with zeros
    buffer_list.mBuffers[0].mData = temp_buffer.data();
    
    // Ensure the buffer is properly aligned
    if (buffer_list.mBuffers[0].mData == nullptr) {
        LOG_ERROR(Audio, "Failed to allocate audio buffer");
        return kAudioUnitErr_CannotDoInCurrentContext;
    }

    // No need for mutex with queue-based approach

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

    // Process the captured data
    Samples output_samples;
    
    if (has_data) {
        // Reset warning counter on successful capture
        input->impl->warning_count.store(0);
        
        // Get the raw audio data (always 16-bit signed PCM from capture)
        const u8* src_data = static_cast<const u8*>(buffer_list.mBuffers[0].mData);
        const size_t data_size = buffer_list.mBuffers[0].mDataByteSize;
        
        // Convert to the target format if needed
        if (input->parameters.sample_size == 8) {
            // Convert 16-bit to 8-bit
            const size_t num_samples = data_size / 2;
            output_samples.resize(num_samples);
            
            // Process each 16-bit sample
            for (size_t i = 0; i < num_samples; i++) {
                // Get the 16-bit sample (little-endian)
                s16 sample_16bit = static_cast<s16>((src_data[i*2+1] << 8) | src_data[i*2]);
                
                // Convert to 8-bit based on signedness
                if (input->impl->sign == Signedness::Signed) {
                    // Convert to signed 8-bit (scale down and keep sign)
                    output_samples[i] = static_cast<u8>((sample_16bit >> 8) + 128);
                } else {
                    // Convert to unsigned 8-bit (scale down and shift)
                    output_samples[i] = static_cast<u8>((sample_16bit >> 8) + 128);
                }
            }
        } else {
            // Keep as 16-bit but handle signedness conversion if needed
            output_samples.resize(data_size);
            
            if (input->impl->sign == Signedness::Signed) {
                // Keep as signed 16-bit
                std::memcpy(output_samples.data(), src_data, data_size);
            } else {
                // Convert from signed to unsigned 16-bit
                const size_t num_samples = data_size / 2;
                for (size_t i = 0; i < num_samples; i++) {
                    // Get the signed 16-bit sample
                    s16 sample_16bit = static_cast<s16>((src_data[i*2+1] << 8) | src_data[i*2]);
                    
                    // Convert to unsigned 16-bit
                    u16 unsigned_sample = static_cast<u16>(sample_16bit + 32768);
                    
                    // Store in little-endian format
                    output_samples[i*2] = unsigned_sample & 0xFF;
                    output_samples[i*2+1] = (unsigned_sample >> 8) & 0xFF;
                }
            }
        }
        
        LOG_INFO(Audio, "Callback: Processed {} bytes of audio data to {} bytes in target format", 
                 data_size, output_samples.size());
    } else {
        // For silence, create a buffer of zeros with the appropriate size
        const size_t silence_size = input->impl->sample_size_in_bytes * inNumberFrames;
        output_samples.resize(silence_size, 0);
        
        if (input->impl->warning_count.fetch_add(1) % 100 == 0) {
            LOG_WARNING(Audio, "Callback: No valid audio data captured, using silence buffer of {} bytes (permission: {})",
                       silence_size, permission_granted ? "granted" : "denied");
        }
    }
    
    // Resample if needed
    if (input->impl->actual_sample_rate != input->impl->target_sample_rate) {
        Samples resampled_output;
        
        if (input->parameters.sample_size == 8) {
            // Resample 8-bit data
            resampled_output = ResampleAudio<u8>(
                output_samples,
                input->impl->actual_sample_rate,
                input->impl->target_sample_rate);
        } else {
            // For 16-bit, we need to convert to s16 for resampling
            const size_t num_samples = output_samples.size() / 2;
            std::vector<s16> samples_16bit(num_samples);
            
            // Convert byte array to s16 array
            for (size_t i = 0; i < num_samples; i++) {
                if (input->impl->sign == Signedness::Signed) {
                    // Already signed, just convert from bytes to s16
                    samples_16bit[i] = static_cast<s16>((output_samples[i*2+1] << 8) | output_samples[i*2]);
                } else {
                    // Convert from unsigned to signed for processing
                    u16 unsigned_sample = (output_samples[i*2+1] << 8) | output_samples[i*2];
                    samples_16bit[i] = static_cast<s16>(unsigned_sample - 32768);
                }
            }
            
            // Resample the s16 data
            auto resampled_16bit = ResampleAudio<s16>(
                samples_16bit, 
                input->impl->actual_sample_rate, 
                input->impl->target_sample_rate);
            
            // Convert back to byte array
            resampled_output.resize(resampled_16bit.size() * 2);
            for (size_t i = 0; i < resampled_16bit.size(); i++) {
                if (input->impl->sign == Signedness::Signed) {
                    // Keep as signed
                    resampled_output[i*2] = resampled_16bit[i] & 0xFF;
                    resampled_output[i*2+1] = (resampled_16bit[i] >> 8) & 0xFF;
                } else {
                    // Convert back to unsigned
                    u16 unsigned_sample = static_cast<u16>(resampled_16bit[i] + 32768);
                    resampled_output[i*2] = unsigned_sample & 0xFF;
                    resampled_output[i*2+1] = (unsigned_sample >> 8) & 0xFF;
                }
            }
        }
        
        LOG_INFO(Audio, "Callback: Resampled from {} Hz to {} Hz, size changed from {} to {} bytes",
                 input->impl->actual_sample_rate, input->impl->target_sample_rate,
                 output_samples.size(), resampled_output.size());
        
        // Use the resampled data
        output_samples = std::move(resampled_output);
    }
    
    // Push the processed samples to the queue
    input->impl->sample_queue.Push(output_samples);

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
