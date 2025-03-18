// Copyright 2012 Michael Kang, 2015 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "common/common_types.h"

#ifndef USE_NEON
#define USE_NEON 1
#endif

enum class ARMDecodeStatus { SUCCESS, FAILURE };

/**
 * Decode a single ARM instruction and find its matching index in the instruction table.
 * @param instr The instruction to decode
 * @param idx Pointer to store the index of the matched instruction
 * @return SUCCESS if a matching instruction was found, FAILURE otherwise
 */
ARMDecodeStatus DecodeARMInstruction(u32 instr, int* idx);

/**
 * Decode multiple ARM instructions in parallel using SIMD operations.
 * This function processes instructions in batches of 4 (when possible) using SIMD.
 * On platforms without SIMD support, it falls back to sequential decoding.
 * 
 * @param instrs Array of instructions to decode
 * @param indices Array to store the indices of matched instructions
 * @param count Number of instructions to decode
 * @return Number of successfully decoded instructions
 */
int BatchDecodeARMInstructions(const u32* instrs, int* indices, int count);
