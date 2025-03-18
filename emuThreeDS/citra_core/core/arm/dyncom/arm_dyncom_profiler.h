#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include "common/common_types.h"

// Simple profiler for measuring CPU emulation performance
class ARMDyncomProfiler {
public:
    static ARMDyncomProfiler& GetInstance() {
        static ARMDyncomProfiler instance;
        return instance;
    }
    
    // Start timing a section
    void StartTiming(const std::string& section_name) {
        section_starts[section_name] = std::chrono::high_resolution_clock::now();
    }
    
    // End timing a section and accumulate the time
    void EndTiming(const std::string& section_name) {
        auto now = std::chrono::high_resolution_clock::now();
        auto start = section_starts[section_name];
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - start).count();
        
        section_times[section_name] += duration;
        section_counts[section_name]++;
    }
    
    // Record an instruction execution
    void RecordInstructionExecution(u32 address) {
        instruction_counts[address]++;
        total_instructions++;
    }
    
    // Record a block execution
    void RecordBlockExecution(u32 address, int size) {
        block_counts[address]++;
        total_blocks++;
        total_block_size += size;
    }
    
    // Get the average time for a section in microseconds
    double GetAverageTime(const std::string& section_name) {
        if (section_counts[section_name] == 0) return 0.0;
        return static_cast<double>(section_times[section_name]) / section_counts[section_name];
    }
    
    // Get the total time for a section in microseconds
    u64 GetTotalTime(const std::string& section_name) {
        return section_times[section_name];
    }
    
    // Get the count for a section
    u64 GetCount(const std::string& section_name) {
        return section_counts[section_name];
    }
    
    // Get the total number of instructions executed
    u64 GetTotalInstructions() {
        return total_instructions;
    }
    
    // Get the total number of blocks executed
    u64 GetTotalBlocks() {
        return total_blocks;
    }
    
    // Get the average block size
    double GetAverageBlockSize() {
        if (total_blocks == 0) return 0.0;
        return static_cast<double>(total_block_size) / total_blocks;
    }
    
    // Reset all counters
    void Reset() {
        section_times.clear();
        section_counts.clear();
        instruction_counts.clear();
        block_counts.clear();
        total_instructions = 0;
        total_blocks = 0;
        total_block_size = 0;
    }
    
    // Print profiling statistics
    void PrintStats() {
        printf("\n===== ARM Dyncom Profiler Statistics =====\n");
        
        // Print section timings
        printf("\nSection Timings:\n");
        printf("%-25s %-15s %-15s %-15s\n", "Section", "Count", "Total (us)", "Average (us)");
        printf("--------------------------------------------------------------------\n");
        
        for (const auto& pair : section_times) {
            const std::string& section = pair.first;
            u64 count = section_counts[section];
            u64 total = pair.second;
            double avg = GetAverageTime(section);
            
            printf("%-25s %-15llu %-15llu %-15.2f\n", 
                   section.c_str(), count, total, avg);
        }
        
        // Print instruction statistics
        printf("\nInstruction Statistics:\n");
        printf("Total Instructions Executed: %llu\n", total_instructions);
        
        // Print block statistics
        printf("\nBlock Statistics:\n");
        printf("Total Blocks Executed: %llu\n", total_blocks);
        printf("Average Block Size: %.2f instructions\n", GetAverageBlockSize());
        
        // Print top instructions
        printf("\nTop 10 Most Executed Instructions:\n");
        printf("%-12s %-15s\n", "Address", "Count");
        printf("---------------------------\n");
        
        auto top_instrs = GetTopInstructions(10);
        for (const auto& pair : top_instrs) {
            printf("0x%08X %-15llu\n", pair.first, pair.second);
        }
        
        // Print top blocks
        printf("\nTop 10 Most Executed Blocks:\n");
        printf("%-12s %-15s\n", "Address", "Count");
        printf("---------------------------\n");
        
        auto top_blocks = GetTopBlocks(10);
        for (const auto& pair : top_blocks) {
            printf("0x%08X %-15llu\n", pair.first, pair.second);
        }
        
        printf("\n=========================================\n");
    }
    
    // Get the most frequently executed instructions
    std::vector<std::pair<u32, u64>> GetTopInstructions(int count) {
        std::vector<std::pair<u32, u64>> result;
        for (const auto& pair : instruction_counts) {
            result.push_back(pair);
        }
        
        std::sort(result.begin(), result.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        if (result.size() > count) {
            result.resize(count);
        }
        
        return result;
    }
    
    // Get the most frequently executed blocks
    std::vector<std::pair<u32, u64>> GetTopBlocks(int count) {
        std::vector<std::pair<u32, u64>> result;
        for (const auto& pair : block_counts) {
            result.push_back(pair);
        }
        
        std::sort(result.begin(), result.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        if (result.size() > count) {
            result.resize(count);
        }
        
        return result;
    }
    
private:
    ARMDyncomProfiler() = default;
    ~ARMDyncomProfiler() = default;
    
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> section_starts;
    std::unordered_map<std::string, u64> section_times;
    std::unordered_map<std::string, u64> section_counts;
    std::unordered_map<u32, u64> instruction_counts;
    std::unordered_map<u32, u64> block_counts;
    
    u64 total_instructions = 0;
    u64 total_blocks = 0;
    u64 total_block_size = 0;
};
