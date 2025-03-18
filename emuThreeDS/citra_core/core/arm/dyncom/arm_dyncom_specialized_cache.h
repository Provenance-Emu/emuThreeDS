#pragma once

#include <array>
#include <unordered_map>
#include <vector>
#include "common/common_types.h"
#include "core/arm/skyeye_common/armstate.h"

// Specialized instruction cache for ARM instructions
// This cache stores pre-decoded instructions in a more efficient format
// to reduce the overhead of instruction decoding
class ARMSpecializedCache {
public:
    // Maximum number of instructions in a specialized block
    static constexpr int MAX_SPECIALIZED_BLOCK_SIZE = 32;
    
    // Types of specialized blocks
    enum class BlockType {
        UNKNOWN = 0,
        MEMORY_COPY,    // Memory copy operations (LDR/STR sequences)
        MEMORY_FILL,    // Memory fill operations (MOV/STR sequences)
        ARITHMETIC,     // Arithmetic operations (ADD/SUB/MUL sequences)
        BRANCH_HEAVY,   // Blocks with many branches
        SEQUENTIAL      // Simple sequential blocks
    };
    
    // Specialized block structure
    struct SpecializedBlock {
        BlockType type;
        u32 start_address;
        u32 end_address;
        int instruction_count;
        std::vector<u32> instruction_addresses;
        
        // Block-specific data
        union {
            struct {
                u32 src_reg;
                u32 dst_reg;
                u32 count_reg;
                bool increment;
            } memory_copy;
            
            struct {
                u32 value_reg;
                u32 dst_reg;
                u32 count_reg;
                bool increment;
            } memory_fill;
            
            struct {
                u32 result_reg;
                u32 operand1_reg;
                u32 operand2_reg;
                u32 operation; // 0=ADD, 1=SUB, 2=MUL, etc.
            } arithmetic;
        } data;
        
        // Execute the specialized block
        unsigned Execute(ARMul_State* cpu);
    };
    
    ARMSpecializedCache() = default;
    ~ARMSpecializedCache() = default;
    
    // Find or create a specialized block for the given address
    SpecializedBlock* FindOrCreateBlock(ARMul_State* cpu, u32 address);
    
    // Execute a specialized block
    unsigned ExecuteBlock(ARMul_State* cpu, u32 address);
    
    // Clear the cache
    void Clear();
    
    // Get statistics
    size_t GetBlockCount() const { return blocks.size(); }
    size_t GetHitCount() const { return hit_count; }
    size_t GetMissCount() const { return miss_count; }
    
private:
    // Analyze a sequence of instructions to determine if it can be specialized
    BlockType AnalyzeInstructions(ARMul_State* cpu, u32 start_address, 
                                 SpecializedBlock& block);
    
    // Create specialized handlers for different block types
    void CreateMemoryCopyHandler(ARMul_State* cpu, SpecializedBlock& block);
    void CreateMemoryFillHandler(ARMul_State* cpu, SpecializedBlock& block);
    void CreateArithmeticHandler(ARMul_State* cpu, SpecializedBlock& block);
    
    // Cache of specialized blocks
    std::unordered_map<u32, SpecializedBlock> blocks;
    
    // Statistics
    size_t hit_count = 0;
    size_t miss_count = 0;
};
