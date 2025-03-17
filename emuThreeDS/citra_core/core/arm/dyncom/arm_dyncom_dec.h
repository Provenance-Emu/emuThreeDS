// Copyright 2012 Michael Kang, 2015 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <unordered_map>
#include "common/common_types.h"

enum class ARMDecodeStatus { SUCCESS, FAILURE };

// Cached instruction decode result
struct ARMInstructionInfo {
    int instruction_index;
    ARMDecodeStatus status;
};

// Decode an ARM instruction with caching
ARMDecodeStatus DecodeARMInstruction(u32 instr, int* idx);

// Clear the instruction decode cache
void ClearARMInstructionCache();
