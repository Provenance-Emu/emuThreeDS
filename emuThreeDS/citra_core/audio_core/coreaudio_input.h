// Copyright 2023 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <AudioUnit/AudioUnit.h>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "audio_core/input.h"

namespace AudioCore {

class CoreAudioInput final : public Input {
public:
    explicit CoreAudioInput(std::string device_id);
    ~CoreAudioInput() override;

    void StartSampling(const InputParameters& params) override;
    void StopSampling() override;
    bool IsSampling() override;
    void AdjustSampleRate(u32 sample_rate) override;
    Samples Read() override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
    std::string device_id;

    static OSStatus AudioInputCallback(void* inRefCon, 
                                      AudioUnitRenderActionFlags* ioActionFlags,
                                      const AudioTimeStamp* inTimeStamp,
                                      UInt32 inBusNumber,
                                      UInt32 inNumberFrames,
                                      AudioBufferList* ioData);
};

std::vector<std::string> ListCoreAudioInputDevices();

} // namespace AudioCore
