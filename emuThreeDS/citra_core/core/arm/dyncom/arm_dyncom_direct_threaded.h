#pragma once

#include "common/common_types.h"
#include "core/arm/skyeye_common/armstate.h"

// Forward declarations
class ARMSpecializedCache;

// Structure to hold instruction data for direct threading
struct ThreadedInstruction {
    void (*handler)(ARMul_State*, void*);
    void* inst_base;  // This will be cast to arm_inst* when used
    u32 address;
};

// Entry point for the direct-threaded interpreter
unsigned DirectThreadedMainLoop(ARMul_State* cpu);

// Execute a block of instructions using the threaded interpreter
unsigned ExecuteThreadedBlock(ARMul_State* cpu, u32 start_address, unsigned max_instructions);

// Memory validation function
bool IsValidMemoryAddress(u32 address);

// Get the global specialized cache instance
ARMSpecializedCache& GetGlobalSpecializedCache();

// Decode a block of ARM instructions for direct-threaded execution
// Returns the number of instructions decoded
unsigned DecodeThreadedBlock(ARMul_State* cpu, u32 start_address, ThreadedInstruction* instructions, unsigned max_instructions);

// Execute a pre-decoded block of instructions
// Returns the number of instructions executed
unsigned ExecuteThreadedBlock(ARMul_State* cpu, ThreadedInstruction* instructions, int count);

// Batch decode multiple ARM instructions at once using SIMD when available
// This is an optimization for ARM64/NEON platforms
unsigned BatchDecodeThreadedBlock(ARMul_State* cpu, u32 start_address, ThreadedInstruction* instructions, unsigned max_instructions);
