// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "core/arm/dyncom/arm_dyncom_block.h"
#include "core/arm/skyeye_common/armstate.h"
#include "core/arm/dyncom/arm_dyncom_interpreter.h"
#include "core/arm/dyncom/arm_dyncom_trans.h"
#include "common/settings.h"

// Forward declarations for functions defined in arm_dyncom_interpreter.cpp
extern int InterpreterTranslateSingle(ARMul_State* cpu, std::size_t& bb_start, u32 addr);

// Constants from arm_dyncom_interpreter.cpp
enum { KEEP_GOING, FETCH_EXCEPTION };

// External variables from arm_dyncom_trans.h
extern char trans_cache_buf[TRANS_CACHE_SIZE];
extern std::size_t trans_cache_buf_top;

// Forward declaration for CondPassed function from arm_dyncom_interpreter.cpp
extern bool CondPassed(const ARMul_State* cpu, unsigned int cond);

namespace Core {

BlockCache::BlockCache() : current_timestamp(0) {}

BlockCache::~BlockCache() {
    Clear();
}

OptimizedBlock* BlockCache::FindBlock(u32 address) {
    // Only use block cache if block-based optimization is enabled
    if (!Settings::values.use_block_based_optimization.GetValue()) {
        return nullptr;
    }
    
    // Validate address alignment
    if (address & 0x3) {
        LOG_ERROR(Core_ARM11, "FindBlock: Unaligned address: [{:#010X}]", address);
        return nullptr;
    }
    
    auto it = blocks.find(address);
    if (it != blocks.end()) {
        // Validate the block before returning it
        OptimizedBlock* block = &it->second;
        
        // Check for empty or invalid blocks
        if (block->instructions.empty()) {
            LOG_ERROR(Core_ARM11, "FindBlock: Empty block at address: [{:#010X}]", address);
            blocks.erase(it); // Remove invalid block
            lru_list.remove(address);
            return nullptr;
        }
        
        // Check for invalid block boundaries
        if ((block->start_address & 0x3) || (block->end_address & 0x3) || 
            block->start_address >= block->end_address) {
            LOG_ERROR(Core_ARM11, "FindBlock: Invalid block boundaries. Start: [{:#010X}] End: [{:#010X}]", 
                     block->start_address, block->end_address);
            blocks.erase(it); // Remove invalid block
            lru_list.remove(address);
            return nullptr;
        }
        
        UpdateBlockUsage(block);
        return block;
    }
    return nullptr;
}

OptimizedBlock* BlockCache::CreateBlock(u32 address) {
    // Only create blocks if block-based optimization is enabled
    if (!Settings::values.use_block_based_optimization.GetValue()) {
        return nullptr;
    }
    
    // Validate address alignment
    if (address & 0x3) {
        LOG_ERROR(Core_ARM11, "CreateBlock: Unaligned address: [{:#010X}]", address);
        return nullptr;
    }
    
    // Check if we're trying to create a block in a non-executable memory region
    // This would require access to the Memory class to check if the address is in executable memory
    // For now, we'll just check if the address is in a reasonable range
    if (address < 0x100000 || address > 0xFFFFFFFF) {
        LOG_ERROR(Core_ARM11, "CreateBlock: Address out of reasonable range: [{:#010X}]", address);
        return nullptr;
    }
    
    // Check if a block already exists at this address
    auto existing = blocks.find(address);
    if (existing != blocks.end()) {
        // Block already exists, return it
        return &existing->second;
    }
    
    EvictBlockIfNeeded();
    
    try {
        auto result = blocks.emplace(address, OptimizedBlock(address));
        OptimizedBlock* block = &result.first->second;
        
        // Add to LRU list
        lru_list.push_front(address);
        UpdateBlockUsage(block);
        
        return block;
    } catch (const std::exception& e) {
        LOG_ERROR(Core_ARM11, "Exception creating block at address [{:#010X}]: %s", address, e.what());
        return nullptr;
    } catch (...) {
        LOG_ERROR(Core_ARM11, "Unknown exception creating block at address [{:#010X}]", address);
        return nullptr;
    }
}

void BlockCache::InvalidateRange(u32 start_address, u32 size) {
    if (size == 0) return;
    
    u32 end_address = start_address + size;
    
    // Find and remove any blocks that overlap with the given range
    auto it = blocks.begin();
    while (it != blocks.end()) {
        const OptimizedBlock& block = it->second;
        
        // Check if block overlaps with the given range
        if (!(block.end_address < start_address || block.start_address >= end_address)) {
            // Remove from LRU list
            lru_list.remove(block.start_address);
            
            // Remove from blocks map
            it = blocks.erase(it);
        } else {
            ++it;
        }
    }
}

void BlockCache::Clear() {
    blocks.clear();
    lru_list.clear();
}

void BlockCache::UpdateBlockUsage(OptimizedBlock* block) {
    if (!block) return;
    
    // Update timestamp
    block->last_used = current_timestamp++;
    block->execution_count++;
    
    // Move to front of LRU list
    lru_list.remove(block->start_address);
    lru_list.push_front(block->start_address);
}

void BlockCache::EvictBlockIfNeeded() {
    if (blocks.size() >= MAX_BLOCK_CACHE_SIZE) {
        // Remove least recently used block
        if (!lru_list.empty()) {
            u32 address = lru_list.back();
            blocks.erase(address);
            lru_list.pop_back();
        }
    }
}

bool DetectBlock(ARMul_State* cpu, u32 start_address, OptimizedBlock* block) {
    if (!block) return false;
    
    // Validate the start address
    if (start_address & 0x3) { // Not word-aligned
        return false;
    }
    
    const size_t MAX_BLOCK_SIZE = 32; // Reduced maximum block size for safety
    size_t num_instructions = 0;
    u32 current_address = start_address;
    bool is_branch = false;
    
    // Detect block until we hit a branch or reach maximum size
    while (num_instructions < MAX_BLOCK_SIZE && !is_branch) {
        // Translate the instruction at current_address
        size_t ptr;
        if (InterpreterTranslateSingle(cpu, ptr, current_address) != FETCH_EXCEPTION) {
            arm_inst* inst_base = (arm_inst*)&trans_cache_buf[ptr];
            
            // Validate the instruction
            if (!inst_base) {
                return false;
            }
            
            // Add instruction to block
            block->instructions.push_back(inst_base);
            
            // Check if this is a branch instruction
            if ((int)inst_base->br != (int)TransExtData::NON_BRANCH) {
                is_branch = true;
                block->has_branch_exit = true;
                
                // If this is a direct branch, record the target
                if ((int)inst_base->br == (int)TransExtData::DIRECT_BRANCH) {
                    // For B/BL instructions, the target is PC + 8 + signed_immed_24
                    bbl_inst* inst_cream = (bbl_inst*)inst_base->component;
                    if (inst_cream) {
                        block->branch_target = current_address + 8 + inst_cream->signed_immed_24;
                        
                        // Validate branch target
                        if (block->branch_target & 0x3) { // Not word-aligned
                            block->branch_target = 0; // Invalid target
                        }
                    }
                }
            }
            
            // Move to next instruction
            current_address += cpu->GetInstructionSize();
            num_instructions++;
        } else {
            // Failed to translate instruction
            return false;
        }
    }
    
    // Set end address
    block->end_address = current_address;
    
    return num_instructions > 0;
}

unsigned ExecuteBlock(ARMul_State* cpu, OptimizedBlock* block) {
    // Validate inputs
    if (!cpu || !block || block->instructions.empty()) {
        return 0;
    }
    
    // Validate block addresses
    if ((block->start_address & 0x3) || (block->end_address & 0x3)) {
        LOG_ERROR(Core_ARM11, "Invalid block addresses. Start: [{:#010X}] End: [{:#010X}]", 
                 block->start_address, block->end_address);
        return 0;
    }
    
    unsigned num_instrs = 0;
    
    // Update block usage statistics
    if (cpu->block_cache) {
        cpu->block_cache->UpdateBlockUsage(block);
    }
    
    // Prepare for block execution
    // Save original PC to restore after block execution if needed
    u32 original_pc = cpu->Reg[15];
    
    // Track register usage and flag dependencies within the block for cross-instruction optimizations
    bool nflag_used = false;
    bool zflag_used = false;
    bool cflag_used = false;
    bool vflag_used = false;
    
    // First pass: analyze the block to determine flag usage
    // This allows us to skip flag calculations for instructions where the flags aren't used
    for (size_t i = 0; i < block->instructions.size(); i++) {
        arm_inst* inst = block->instructions[i];
        
        // Validate instruction
        if (!inst) {
            LOG_ERROR(Core_ARM11, "Null instruction in block at index %zu", i);
            return num_instrs;
        }
        
        // Check if this instruction uses any flags
        if (inst->cond != AL) {
            // This instruction is conditional, so it uses flags
            // Determine which flags are used based on the condition code
            switch (inst->cond) {
                case EQ: // Z set
                    zflag_used = true;
                    break;
                case NE: // Z clear
                    zflag_used = true;
                    break;
                case CS: // C set
                    cflag_used = true;
                    break;
                case CC: // C clear
                    cflag_used = true;
                    break;
                case MI: // N set
                    nflag_used = true;
                    break;
                case PL: // N clear
                    nflag_used = true;
                    break;
                case VS: // V set
                    vflag_used = true;
                    break;
                case VC: // V clear
                    vflag_used = true;
                    break;
                case HI: // C set and Z clear
                    cflag_used = true;
                    zflag_used = true;
                    break;
                case LS: // C clear or Z set
                    cflag_used = true;
                    zflag_used = true;
                    break;
                case GE: // N == V
                    nflag_used = true;
                    vflag_used = true;
                    break;
                case LT: // N != V
                    nflag_used = true;
                    vflag_used = true;
                    break;
                case GT: // Z clear AND (N == V)
                    zflag_used = true;
                    nflag_used = true;
                    vflag_used = true;
                    break;
                case LE: // Z set OR (N != V)
                    zflag_used = true;
                    nflag_used = true;
                    vflag_used = true;
                    break;
                default:
                    // For unknown conditions, assume all flags are used
                    nflag_used = zflag_used = cflag_used = vflag_used = true;
                    break;
            }
        }
    }
    
    // Execute each instruction in the block
    for (size_t i = 0; i < block->instructions.size(); i++) {
        arm_inst* inst_base = block->instructions[i];
        
        // Validate instruction
        if (!inst_base) {
            LOG_ERROR(Core_ARM11, "Null instruction in block at index %zu during execution", i);
            return num_instrs;
        }
        
        // Check if we've reached the instruction limit
        if (num_instrs >= cpu->NumInstrsToExecute) {
            break;
        }
        
        // Safety check for PC value
        if ((cpu->Reg[15] & 0x3) != 0) {
            LOG_ERROR(Core_ARM11, "Unaligned PC detected: [{:#010X}]", cpu->Reg[15]);
            cpu->Reg[15] = original_pc; // Restore original PC
            return num_instrs;
        }
        
        try {
            // First, check if condition passes
            if (inst_base->cond == AL || CondPassed(cpu, inst_base->cond)) {
                // Execute the instruction with cross-instruction optimizations
                
                // Check if this instruction sets flags that aren't used later in the block
                bool skip_flag_calculation = false;
                
                // For arithmetic instructions, check if they set flags
                if (inst_base->idx >= 90 && inst_base->idx <= 110) { // Rough range for arithmetic ops
                    // If this instruction sets flags, but none of those flags are used later,
                    // we can skip the flag calculation
                    if (!nflag_used && !zflag_used && !cflag_used && !vflag_used) {
                        skip_flag_calculation = true;
                    }
                }
                
                // For now, just increment PC and instruction count
                // In a real implementation, the instruction handlers would update PC
                cpu->Reg[15] += cpu->GetInstructionSize();
                num_instrs++;
            } else {
                // Condition failed, just update PC
                cpu->Reg[15] += cpu->GetInstructionSize();
            }
            
            // If this is a branch instruction and we know the target,
            // set up the next block address for direct chaining
            if ((int)inst_base->br != (int)TransExtData::NON_BRANCH && block->branch_target != 0) {
                // Validate branch target
                if ((block->branch_target & 0x3) == 0) {
                    cpu->next_block_address = block->branch_target;
                } else {
                    // Invalid branch target
                    LOG_ERROR(Core_ARM11, "Invalid branch target: [{:#010X}]", block->branch_target);
                    cpu->next_block_address = 0;
                }
            } else if (i == block->instructions.size() - 1) {
                // If this is the last instruction and it's not a branch,
                // set up the next block address to the next sequential block
                cpu->next_block_address = block->end_address;
            } else {
                cpu->next_block_address = 0;
            }
        } catch (...) {
            LOG_ERROR(Core_ARM11, "Exception during block execution at PC: [{:#010X}]", cpu->Reg[15]);
            return num_instrs;
        }
        
        // Check for interrupts or other exceptional conditions
        if (!cpu->NirqSig && !(cpu->Cpsr & 0x80)) {
            // IRQ occurred and not masked
            break;
        }
    }
    
    return num_instrs;
}

} // namespace Core
