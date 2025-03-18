#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "common/common_types.h"
#include "core/arm/dyncom/arm_dyncom.h"
#include "core/arm/dyncom/arm_dyncom_direct_threaded.h"
#include "core/arm/dyncom/arm_dyncom_interpreter.h"
#include "core/arm/dyncom/arm_dyncom_profiler.h"
#include "core/arm/dyncom/arm_dyncom_test.h"
#include "core/core.h"
#include "core/memory.h"

// Simple benchmark program to test the performance of the different interpreter implementations
int main(int argc, char* argv[]) {
    std::cout << "ARM Dyncom Benchmark" << std::endl;
    std::cout << "===================" << std::endl;
    
    // Create a Core instance
    Core::System system;
    
    // Create an ARM_DynCom instance
    auto arm_cpu = std::make_unique<ARM_DynCom>(system);
    
    // Initialize the CPU
    arm_cpu->SetPC(0x00100000);
    
    // Load some test code into memory
    // In a real benchmark, we would load actual ARM code here
    // For now, we'll just use some dummy values
    for (u32 addr = 0x00100000; addr < 0x00110000; addr += 4) {
        // Create some simple ARM instructions (ADD, MOV, etc.)
        u32 instr = 0xE0800001;  // ADD R0, R0, R1
        if ((addr & 0xFF) == 0) {
            instr = 0xE1A00001;  // MOV R0, R1
        } else if ((addr & 0xFF) == 4) {
            instr = 0xE1A01000;  // MOV R1, R0
        } else if ((addr & 0xFF) == 8) {
            instr = 0xE0400001;  // SUB R0, R0, R1
        } else if ((addr & 0xFF) == 12) {
            instr = 0xE5801000;  // STR R1, [R0]
        } else if ((addr & 0xFF) == 16) {
            instr = 0xE5900000;  // LDR R0, [R0]
        }
        
        // Write the instruction to memory
        arm_cpu->memory.Write32(addr, instr);
    }
    
    // Run the benchmark
    TestInterpreterPerformance(arm_cpu.get(), 10, 1000000);
    
    return 0;
}
