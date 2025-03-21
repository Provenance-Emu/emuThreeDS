// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <vector>
#include <unordered_map>
#include <list>
#include "common/common_types.h"

struct arm_inst;
struct ARMul_State;

namespace Core {

/**
 * Represents an optimized block of ARM instructions that can be executed together.
 * This is used for Block-Based Optimization to improve emulation performance.
 */
struct OptimizedBlock {
    /// Starting address of the block
    u32 start_address;
    
    /// Ending address of the block
    u32 end_address;
    
    /// Instructions in this block
    std::vector<arm_inst*> instructions;
    
    /// Whether this block ends with a branch instruction
    bool has_branch_exit;
    
    /// Target address of the branch (if known statically)
    u32 branch_target;
    
    /// Location in the optimized code cache
    size_t cache_index;
    
    /// Last time this block was executed (for LRU eviction)
    u64 last_used;
    
    /// Number of times this block has been executed
    u32 execution_count;
    
    /// Constructor
    OptimizedBlock(u32 start_addr) 
        : start_address(start_addr), end_address(0), has_branch_exit(false),
          branch_target(0), cache_index(0), last_used(0), execution_count(0) {}
};

/**
 * Manages a cache of optimized code blocks for the ARM interpreter.
 * Provides block lookup, creation, and invalidation functionality.
 */
class BlockCache {
public:
    BlockCache();
    ~BlockCache();
    
    /**
     * Finds an optimized block starting at the given address.
     * @param address The address to look up
     * @return Pointer to the block if found, nullptr otherwise
     */
    OptimizedBlock* FindBlock(u32 address);
    
    /**
     * Creates a new optimized block starting at the given address.
     * @param address The starting address of the block
     * @return Pointer to the newly created block
     */
    OptimizedBlock* CreateBlock(u32 address);
    
    /**
     * Invalidates any blocks that contain the given address range.
     * @param start_address Start of the address range to invalidate
     * @param size Size of the address range to invalidate
     */
    void InvalidateRange(u32 start_address, u32 size);
    
    /**
     * Clears all cached blocks.
     */
    void Clear();
    
    /**
     * Updates the LRU information for a block that was just used.
     * @param block The block that was used
     */
    void UpdateBlockUsage(OptimizedBlock* block);
    
private:
    /// Maximum number of blocks to keep in the cache
    static const size_t MAX_BLOCK_CACHE_SIZE = 8192;
    
    /// Map from start address to optimized block
    std::unordered_map<u32, OptimizedBlock> blocks;
    
    /// List of block addresses in LRU order
    std::list<u32> lru_list;
    
    /// Current execution count (incremented for each block execution)
    u64 current_timestamp;
    
    /**
     * Evicts the least recently used block if the cache is full.
     */
    void EvictBlockIfNeeded();
};

/**
 * Detects and forms a basic block of ARM instructions.
 * @param cpu The ARM CPU state
 * @param start_address The address to start block formation from
 * @param block The block to populate
 * @return True if block formation was successful, false otherwise
 */
bool DetectBlock(ARMul_State* cpu, u32 start_address, OptimizedBlock* block);

/**
 * Executes an optimized block of ARM instructions.
 * @param cpu The ARM CPU state
 * @param block The block to execute
 * @return Number of instructions executed
 */
unsigned ExecuteBlock(ARMul_State* cpu, OptimizedBlock* block);

} // namespace Core
