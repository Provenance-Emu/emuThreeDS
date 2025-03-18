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
#include "core/core.h"
#include "core/memory.h"

// Test function to compare the performance of different interpreter implementations
void TestInterpreterPerformance(ARM_DynCom* arm_cpu, int num_iterations = 10, u64 num_instructions = 1000000) {
    // Reset the profiler
    arm_cpu->ResetProfiler();
    
    // Test the original interpreter
    std::cout << "Testing original interpreter..." << std::endl;
    arm_cpu->SetUseDirectThreadedInterpreter(false);
    arm_cpu->EnableSpecializedCache(false);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; i++) {
        arm_cpu->ExecuteInstructions(num_instructions);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto original_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    std::cout << "Original interpreter: " << original_duration << " ms for " 
              << num_iterations << " iterations of " << num_instructions << " instructions" << std::endl;
    
    // Reset the profiler
    arm_cpu->ResetProfiler();
    
    // Test the direct-threaded interpreter
    std::cout << "Testing direct-threaded interpreter..." << std::endl;
    arm_cpu->SetUseDirectThreadedInterpreter(true);
    arm_cpu->EnableSpecializedCache(false);
    
    start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; i++) {
        arm_cpu->ExecuteInstructions(num_instructions);
    }
    
    end_time = std::chrono::high_resolution_clock::now();
    auto threaded_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    std::cout << "Direct-threaded interpreter: " << threaded_duration << " ms for " 
              << num_iterations << " iterations of " << num_instructions << " instructions" << std::endl;
    
    // Calculate speedup over original
    double threaded_speedup = static_cast<double>(original_duration) / threaded_duration;
    std::cout << "Speedup: " << std::fixed << std::setprecision(2) << threaded_speedup << "x" << std::endl;
    
    // Reset the profiler
    arm_cpu->ResetProfiler();
    arm_cpu->ClearSpecializedCache();
    
    // Test the direct-threaded interpreter with specialized cache
    std::cout << "\nTesting direct-threaded interpreter with specialized cache..." << std::endl;
    arm_cpu->SetUseDirectThreadedInterpreter(true);
    arm_cpu->EnableSpecializedCache(true);
    
    start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; i++) {
        arm_cpu->ExecuteInstructions(num_instructions);
    }
    
    end_time = std::chrono::high_resolution_clock::now();
    auto specialized_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    std::cout << "Direct-threaded with specialized cache: " << specialized_duration << " ms for " 
              << num_iterations << " iterations of " << num_instructions << " instructions" << std::endl;
    
    // Calculate speedups
    double specialized_speedup = static_cast<double>(original_duration) / specialized_duration;
    double specialized_vs_threaded = static_cast<double>(threaded_duration) / specialized_duration;
    
    std::cout << "Speedup vs original: " << std::fixed << std::setprecision(2) << specialized_speedup << "x" << std::endl;
    std::cout << "Speedup vs threaded: " << std::fixed << std::setprecision(2) << specialized_vs_threaded << "x" << std::endl;
    
    // Print profiler statistics
    arm_cpu->PrintProfilerStats();
}

// Example usage:
// ARM_DynCom* arm_cpu = GetARMCPU();
// TestInterpreterPerformance(arm_cpu);
