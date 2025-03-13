//
//  coreaudio_input.cpp
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
#include "audio_core/input.h"
#include "audio_core/coreaudio_input.h"
#include "common/logging/log.h"

#define USE_MUTEX 1

namespace AudioCore {

struct CoreAudioInput::Impl {
    AudioUnit audio_unit = nullptr;
    AudioBufferList* buffer_list = nullptr;
    AudioStreamBasicDescription format_desc;
    Samples buffer;
    bool is_recording = false;
    u8 sample_size_in_bytes = 0;
    
    // Mutex for thread safety when accessing the buffer
#if USE_MUTEX
    std::mutex buffer_mutex;
#endif
};


CoreAudioInput::CoreAudioInput(std::string device_id)
    : impl(std::make_unique<Impl>()), device_id(std::move(device_id)) {}

CoreAudioInput::~CoreAudioInput() {
    StopSampling();
}

void CoreAudioInput::StartSampling(const InputParameters& params) {
    if (IsSampling()) {
        return;
    }

    parameters = params;
    impl->sample_size_in_bytes = params.sample_size / 8;
    is_sampling = true;

    OSStatus err;
    AudioComponentDescription acdesc = {
        .componentType = kAudioUnitType_Output,
        .componentSubType = kAudioUnitSubType_RemoteIO,
        .componentManufacturer = kAudioUnitManufacturer_Apple,
        .componentFlags = 0,
        .componentFlagsMask = 0,
    };

    AudioComponent component = AudioComponentFindNext(nullptr, &acdesc);
    if (component == nullptr) {
        LOG_CRITICAL(Audio, "Failed to find Input AudioComponent");
        StopSampling();
        return;
    }

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
    NSError* error = nil;
    AVAudioSession* session = [AVAudioSession sharedInstance];
    [session setCategory:AVAudioSessionCategoryRecord error:&error];
    if (error) {
        LOG_CRITICAL(Audio, "Failed to set audio session category");
        StopSampling();
        return;
    }
    
    [session setActive:YES error:&error];
    if (error) {
        LOG_CRITICAL(Audio, "Failed to activate audio session");
        StopSampling();
        return;
    }

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

    // Allocate buffer for audio data
    impl->buffer.resize(params.buffer_size);

    // Initialize and start the audio unit
    err = AudioUnitInitialize(impl->audio_unit);
    if (err != noErr) {
        LOG_CRITICAL(Audio, "Failed to initialize AudioUnit: {}", err);
        StopSampling();
        return;
    }

    err = AudioOutputUnitStart(impl->audio_unit);
    if (err != noErr) {
        LOG_CRITICAL(Audio, "Failed to start AudioUnit: {}", err);
        StopSampling();
        return;
    }

    impl->is_recording = true;
}

void CoreAudioInput::StopSampling() {
    if (impl->audio_unit) {
        if (impl->is_recording) {
            AudioOutputUnitStop(impl->audio_unit);
            impl->is_recording = false;
        }
        
        AudioUnitUninitialize(impl->audio_unit);
        AudioComponentInstanceDispose(impl->audio_unit);
        impl->audio_unit = nullptr;
    }
    
    is_sampling = false;
}

bool CoreAudioInput::IsSampling() {
    return is_sampling && impl->is_recording;
}

void CoreAudioInput::AdjustSampleRate(u32 sample_rate) {
    if (!IsSampling()) {
        return;
    }

    auto new_params = parameters;
    new_params.sample_rate = sample_rate;
    StopSampling();
    StartSampling(new_params);
}

Samples CoreAudioInput::Read() {
    if (!IsSampling()) {
        return {};
    }
    
#if USE_MUTEX
    // Lock the buffer mutex to safely access the buffer
    std::lock_guard<std::mutex> lock(impl->buffer_mutex);
#endif
    
    // Return a copy of the current buffer
    return impl->buffer;
}

OSStatus CoreAudioInput::AudioInputCallback(void* inRefCon, 
                                           AudioUnitRenderActionFlags* ioActionFlags,
                                           const AudioTimeStamp* inTimeStamp,
                                           UInt32 inBusNumber,
                                           UInt32 inNumberFrames,
                                           AudioBufferList* ioData) {
    CoreAudioInput* input = static_cast<CoreAudioInput*>(inRefCon);
    
    if (!input || !input->IsSampling()) {
        return noErr;
    }
    
    // Set up the AudioBufferList to receive the input data
    AudioBufferList buffer_list;
    buffer_list.mNumberBuffers = 1;
    buffer_list.mBuffers[0].mNumberChannels = 1;  // Mono
    buffer_list.mBuffers[0].mDataByteSize = inNumberFrames * input->impl->sample_size_in_bytes;
    
#if USE_MUTEX
    // Lock the buffer mutex to safely access/modify the buffer
    std::lock_guard<std::mutex> lock(input->impl->buffer_mutex);
#endif
    
    // Make sure our buffer is large enough
    if (input->impl->buffer.size() < buffer_list.mBuffers[0].mDataByteSize) {
        input->impl->buffer.resize(buffer_list.mBuffers[0].mDataByteSize);
    }
    
    buffer_list.mBuffers[0].mData = input->impl->buffer.data();
    
    // Render the audio data
    OSStatus err = AudioUnitRender(input->impl->audio_unit,
                                  ioActionFlags,
                                  inTimeStamp,
                                  inBusNumber,
                                  inNumberFrames,
                                  &buffer_list);
    
    if (err != noErr) {
        LOG_WARNING(Audio, "AudioUnitRender failed: {}", err);
        return err;
    }
    
    return noErr;
}

std::vector<std::string> ListCoreAudioInputDevices() {
    std::vector<std::string> device_list;
    
    // On iOS, we only have the default microphone
    device_list.push_back("auto");
    device_list.push_back("Built-in Microphone");
    
    return device_list;
}

} // namespace AudioCore
