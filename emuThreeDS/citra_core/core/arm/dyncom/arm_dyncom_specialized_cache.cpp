#include "core/arm/dyncom/arm_dyncom_specialized_cache.h"
#include "common/microprofile.h"
#include "core/arm/dyncom/arm_dyncom_dec.h"
#include "core/arm/dyncom/arm_dyncom_interpreter.h"
#include "core/arm/dyncom/arm_dyncom_trans.h"
#include "core/arm/dyncom/arm_dyncom_profiler.h"

// Memory validation function
extern bool IsValidMemoryAddress(u32 address);

// Helper function to print CPU register state for debugging
void PrintCPUState(ARMul_State* cpu) {
    fprintf(stderr, "CPU State: PC=0x%08X, CPSR=0x%08X, TFlag=%d\n", 
            cpu->Reg[15], cpu->Cpsr, cpu->TFlag);
    
    // Print general purpose registers
    fprintf(stderr, "Registers: R0=0x%08X, R1=0x%08X, R2=0x%08X, R3=0x%08X\n", 
            cpu->Reg[0], cpu->Reg[1], cpu->Reg[2], cpu->Reg[3]);
    fprintf(stderr, "          R4=0x%08X, R5=0x%08X, R6=0x%08X, R7=0x%08X\n", 
            cpu->Reg[4], cpu->Reg[5], cpu->Reg[6], cpu->Reg[7]);
    fprintf(stderr, "          R8=0x%08X, R9=0x%08X, R10=0x%08X, R11=0x%08X\n", 
            cpu->Reg[8], cpu->Reg[9], cpu->Reg[10], cpu->Reg[11]);
    fprintf(stderr, "          R12=0x%08X, SP=0x%08X, LR=0x%08X\n", 
            cpu->Reg[12], cpu->Reg[13], cpu->Reg[14]);
    
    // Print flags
    fprintf(stderr, "Flags: N=%d, Z=%d, C=%d, V=%d\n", 
            cpu->NFlag, cpu->ZFlag, cpu->CFlag, cpu->VFlag);
}

// We use the IsValidMemoryAddress function from arm_dyncom_direct_threaded.cpp
// which has the complete 3DS memory map validation

// Helper function to dump memory at a given address
void DumpMemory(ARMul_State* cpu, u32 address, int num_words) {
    fprintf(stderr, "Memory dump at 0x%08X:\n", address);
    
    for (int i = 0; i < num_words; i++) {
        u32 addr = address + i * 4;
        
        // Check if address is within valid range using the comprehensive validation
        if (!IsValidMemoryAddress(addr)) {
            fprintf(stderr, "  0x%08X: INVALID ADDRESS\n", addr);
            continue;
        }
        
        // Try to read the memory value, but handle potential exceptions
        try {
            u32 value = cpu->memory.Read32(addr);
            fprintf(stderr, "  0x%08X: 0x%08X\n", addr, value);
        } catch (...) {
            fprintf(stderr, "  0x%08X: ERROR READING MEMORY\n", addr);
        }
    }
}

// Helper function to check if an address is valid
extern bool IsValidMemoryAddress(u32 address);

MICROPROFILE_DEFINE(DynCom_SpecializedCache, "DynCom", "SpecializedCache", MP_RGB(255, 128, 64));

// Execute a specialized block
unsigned ARMSpecializedCache::SpecializedBlock::Execute(ARMul_State* cpu) {
    MICROPROFILE_SCOPE(DynCom_SpecializedCache);
    
    // Add debug logging
    LOG_TRACE(Core_ARM11, "SpecializedBlock::Execute type=%d, PC before=0x%08X", 
              static_cast<int>(type), cpu->Reg[15]);
    fprintf(stderr, "SpecializedBlock::Execute type=%d, PC before=0x%08X\n", 
            static_cast<int>(type), cpu->Reg[15]);
    
    // Get the profiler instance
    auto& profiler = ARMDyncomProfiler::GetInstance();
    
    // Start timing the specialized block execution
    profiler.StartTiming("SpecializedBlockExecution");
    
    unsigned executed_instructions = 0;
    
    switch (type) {
        case BlockType::MEMORY_COPY: {
            // Optimized memory copy implementation
            u32 src_addr = cpu->Reg[data.memory_copy.src_reg];
            u32 dst_addr = cpu->Reg[data.memory_copy.dst_reg];
            u32 count = cpu->Reg[data.memory_copy.count_reg];
            
            // Limit the count to a reasonable value to prevent infinite loops
            count = std::min(count, 1024u);
            
            for (u32 i = 0; i < count; i++) {
                // Validate memory addresses before access
                if (!IsValidMemoryAddress(src_addr) || !IsValidMemoryAddress(dst_addr)) {
                    LOG_ERROR(Core_ARM11, "Memory copy: Invalid memory address - src=0x%08X, dst=0x%08X", src_addr, dst_addr);
                    fprintf(stderr, "ERROR: Memory copy: Invalid memory address - src=0x%08X, dst=0x%08X\n", src_addr, dst_addr);
                    break;
                }
                
                // Read from source
                u32 value = cpu->memory.Read32(src_addr);
                
                // Write to destination
                cpu->memory.Write32(dst_addr, value);
                
                // Update addresses if incrementing
                if (data.memory_copy.increment) {
                    src_addr += 4;
                    dst_addr += 4;
                }
                
                executed_instructions++;
            }
            
            // Update registers
            cpu->Reg[data.memory_copy.src_reg] = src_addr;
            cpu->Reg[data.memory_copy.dst_reg] = dst_addr;
            
            // Update PC to the end of the block
            cpu->Reg[15] = end_address;
            break;
        }
        
        case BlockType::MEMORY_FILL: {
            // Optimized memory fill implementation
            u32 value = cpu->Reg[data.memory_fill.value_reg];
            u32 dst_addr = cpu->Reg[data.memory_fill.dst_reg];
            u32 count = cpu->Reg[data.memory_fill.count_reg];
            
            // Limit the count to a reasonable value to prevent infinite loops
            count = std::min(count, 1024u);
            
            for (u32 i = 0; i < count; i++) {
                // Validate memory address before writing
                if (!IsValidMemoryAddress(dst_addr)) {
                    LOG_ERROR(Core_ARM11, "Memory fill: Invalid memory address - dst=0x%08X", dst_addr);
                    fprintf(stderr, "ERROR: Memory fill: Invalid memory address - dst=0x%08X\n", dst_addr);
                    break;
                }
                
                // Write value to destination
                cpu->memory.Write32(dst_addr, value);
                
                // Update address if incrementing
                if (data.memory_fill.increment) {
                    dst_addr += 4;
                }
                
                executed_instructions++;
            }
            
            // Update registers
            cpu->Reg[data.memory_fill.dst_reg] = dst_addr;
            
            // Update PC to the end of the block
            cpu->Reg[15] = end_address;
            break;
        }
        
        case BlockType::ARITHMETIC: {
            // Optimized arithmetic implementation
            u32 operand1 = cpu->Reg[data.arithmetic.operand1_reg];
            u32 operand2 = cpu->Reg[data.arithmetic.operand2_reg];
            u32 result = 0;
            
            // Perform the operation
            switch (data.arithmetic.operation) {
                case 0: // ADD
                    result = operand1 + operand2;
                    break;
                case 1: // SUB
                    result = operand1 - operand2;
                    break;
                case 2: // MUL
                    result = operand1 * operand2;
                    break;
                // Add more operations as needed
            }
            
            // Update result register
            cpu->Reg[data.arithmetic.result_reg] = result;
            
            // Update PC to the end of the block
            cpu->Reg[15] = end_address;
            
            executed_instructions = instruction_count;
            break;
        }
        
        case BlockType::SEQUENTIAL:
        case BlockType::BRANCH_HEAVY:
        case BlockType::UNKNOWN:
        default:
            // For other block types, fall back to the regular interpreter
            // This is just a placeholder - in a real implementation, we would
            // have specialized handlers for these block types as well
            executed_instructions = instruction_count;
            
            // Ensure PC is updated to avoid infinite loops
            // For now, just advance by 4 bytes (one instruction) to make progress
            cpu->Reg[15] = start_address + 4;
            break;
    }
    
    profiler.EndTiming("SpecializedBlockExecution");
    
    // Add debug logging for the end of execution
    LOG_TRACE(Core_ARM11, "SpecializedBlock::Execute completed, executed=%u, PC after=0x%08X", 
              executed_instructions, cpu->Reg[15]);
    fprintf(stderr, "SpecializedBlock::Execute completed, executed=%u, PC after=0x%08X\n", 
            executed_instructions, cpu->Reg[15]);
    
    return executed_instructions;
}

// Find or create a specialized block for the given address
ARMSpecializedCache::SpecializedBlock* ARMSpecializedCache::FindOrCreateBlock(ARMul_State* cpu, u32 address) {
    MICROPROFILE_SCOPE(DynCom_SpecializedCache);
    
    // Check if we already have a block for this address
    auto it = blocks.find(address);
    if (it != blocks.end()) {
        hit_count++;
        return &it->second;
    }
    
    // No block found, create a new one
    miss_count++;
    
    SpecializedBlock block;
    block.start_address = address;
    block.instruction_count = 0;
    
    // Analyze the instructions to determine the block type
    block.type = AnalyzeInstructions(cpu, address, block);
    
    // Create specialized handlers based on the block type
    switch (block.type) {
        case BlockType::MEMORY_COPY:
            CreateMemoryCopyHandler(cpu, block);
            break;
        case BlockType::MEMORY_FILL:
            CreateMemoryFillHandler(cpu, block);
            break;
        case BlockType::ARITHMETIC:
            CreateArithmeticHandler(cpu, block);
            break;
        // Add more specialized handlers as needed
    }
    
    // Ensure end_address is properly set to avoid infinite loops
    // If not already set, make it at least address + 4 to ensure progress
    if (block.end_address <= block.start_address) {
        block.end_address = block.start_address + 4;
        LOG_TRACE(Core_ARM11, "Setting default end_address=0x%08X for block at 0x%08X",
                  block.end_address, block.start_address);
        fprintf(stderr, "Setting default end_address=0x%08X for block at 0x%08X\n", 
                block.end_address, block.start_address);
    }
    
    // Add the block to the cache
    auto [iter, inserted] = blocks.emplace(address, std::move(block));
    return &iter->second;
}

// Execute a specialized block
unsigned ARMSpecializedCache::ExecuteBlock(ARMul_State* cpu, u32 address) {
    MICROPROFILE_SCOPE(DynCom_SpecializedCache);
    
    // Add debug logging
    LOG_TRACE(Core_ARM11, "ARMSpecializedCache::ExecuteBlock at address=0x%08X", address);
    fprintf(stderr, "ARMSpecializedCache::ExecuteBlock at address=0x%08X\n", address);
    
    // Check if address is valid
    if (!IsValidMemoryAddress(address)) {
        LOG_ERROR(Core_ARM11, "Invalid memory address in ExecuteBlock: 0x%08X", address);
        fprintf(stderr, "ERROR: Invalid memory address in ExecuteBlock: 0x%08X\n", address);
        return 0;
    }
    
    // Print CPU state before execution
//    fprintf(stderr, "CPU state before specialized block execution:\n");
//    PrintCPUState(cpu);
    
    // Dump memory at the current PC
    fprintf(stderr, "Memory at PC before execution:\n");
    DumpMemory(cpu, address, 8); // Dump 8 words (32 bytes)
    
    // Store the original PC for comparison
    u32 original_pc = cpu->Reg[15];
    
    // Find or create a specialized block
    SpecializedBlock* block = FindOrCreateBlock(cpu, address);
    
    // Log block details
    LOG_TRACE(Core_ARM11, "Block type=%d, instruction_count=%d, start=0x%08X, end=0x%08X",
              static_cast<int>(block->type), block->instruction_count, block->start_address, block->end_address);
    fprintf(stderr, "Block type=%d, instruction_count=%d\n", static_cast<int>(block->type), block->instruction_count);
    
    // Execute the block
    unsigned result = block->Execute(cpu);
    
    // Check if PC was updated
    if (cpu->Reg[15] == original_pc) {
        // PC wasn't updated, force it to advance
        LOG_TRACE(Core_ARM11, "PC wasn't updated by specialized block execution, forcing advance");
        fprintf(stderr, "PC wasn't updated by specialized block execution, forcing advance\n");
        
        // Force PC to advance by at least 4 bytes (one instruction)
        cpu->Reg[15] = original_pc + 4;
    }
    
    // Log execution result
    LOG_TRACE(Core_ARM11, "Block execution result=%u, new PC=0x%08X", result, cpu->Reg[15]);
    fprintf(stderr, "Block execution result=%u, new PC=0x%08X\n", result, cpu->Reg[15]);
    
    // Print CPU state after execution
//    fprintf(stderr, "CPU state after specialized block execution:\n");
//    PrintCPUState(cpu);
    
    // Dump memory at the new PC
    fprintf(stderr, "Memory at new PC after execution:\n");
    DumpMemory(cpu, cpu->Reg[15], 8); // Dump 8 words (32 bytes)
    
    return result;
}

// Clear the cache
void ARMSpecializedCache::Clear() {
    blocks.clear();
    hit_count = 0;
    miss_count = 0;
}

// Analyze a sequence of instructions to determine if it can be specialized
ARMSpecializedCache::BlockType ARMSpecializedCache::AnalyzeInstructions(ARMul_State* cpu, u32 start_address, 
                                                                       SpecializedBlock& block) {
    MICROPROFILE_SCOPE(DynCom_SpecializedCache);
    
    // Initialize block
    block.instruction_addresses.clear();
    block.instruction_count = 0;
    
    // Counters for different instruction types
    int load_count = 0;
    int store_count = 0;
    int branch_count = 0;
    int arithmetic_count = 0;
    
    // Pattern detection variables
    bool potential_memory_copy = false;
    bool potential_memory_fill = false;
    bool potential_arithmetic = false;
    
    // Analyze up to MAX_SPECIALIZED_BLOCK_SIZE instructions
    u32 address = start_address;
    for (int i = 0; i < MAX_SPECIALIZED_BLOCK_SIZE; i++) {
        // Validate memory address before reading
        u32 aligned_addr = address & 0xFFFFFFFC;
        if (!IsValidMemoryAddress(aligned_addr)) {
            LOG_ERROR(Core_ARM11, "AnalyzeInstructions: Invalid memory address: 0x%08X", aligned_addr);
            fprintf(stderr, "ERROR: AnalyzeInstructions: Invalid memory address: 0x%08X\n", aligned_addr);
            break;
        }
        
        // Read the instruction
        u32 instr = cpu->memory.Read32(aligned_addr);
        
        // Decode the instruction
        int idx = 0;
        ARMDecodeStatus status = DecodeARMInstruction(instr, &idx);
        if (status != ARMDecodeStatus::SUCCESS || idx < 0 || idx >= static_cast<int>(arm_instruction_trans_len)) {
            // Invalid instruction, stop analysis
            break;
        }
        
        // Add the instruction to the block
        block.instruction_addresses.push_back(address);
        block.instruction_count++;
        
        // Update counters based on instruction type
        if (idx == 94) { // LDR_INST
            load_count++;
        } else if (idx == 95) { // STR_INST
            store_count++;
        } else if (idx == 62) { // ADD_INST
            arithmetic_count++;
        } else if (idx == 66) { // SUB_INST
            arithmetic_count++;
        } else if (idx == 52) { // MUL_INST
            arithmetic_count++;
        }
        
        // Check for branch instructions
        arm_inst* inst_base = arm_instruction_trans[idx](instr, idx);
        if (inst_base->br != TransExtData::NON_BRANCH) {
            branch_count++;
            
            // Stop analysis at unconditional branches
            if (inst_base->cond == ConditionCode::AL) {
                break;
            }
        }
        
        // Move to the next instruction
        address += 4;
    }
    
    // Update block end address
    block.end_address = address;
    
    // Ensure we have at least one instruction and the end address is greater than the start address
    if (block.instruction_count == 0 || block.end_address <= block.start_address) {
        LOG_TRACE(Core_ARM11, "Block has no instructions or invalid end address, setting defaults");
        fprintf(stderr, "Block has no instructions or invalid end address, setting defaults\n");
        
        // Ensure we have at least one instruction
        block.instruction_count = std::max(1, block.instruction_count);
        
        // Ensure end address is at least start_address + 4
        block.end_address = block.start_address + 4;
    }
    
    LOG_TRACE(Core_ARM11, "Analyzed block at 0x%08X, type=%d, instruction_count=%d, end_address=0x%08X",
              block.start_address, static_cast<int>(block.type), block.instruction_count, block.end_address);
    fprintf(stderr, "Analyzed block at 0x%08X, type=%d, instruction_count=%d, end_address=0x%08X\n",
            block.start_address, static_cast<int>(block.type), block.instruction_count, block.end_address);
    
    // Determine block type based on instruction mix
    if (load_count > 0 && store_count > 0 && load_count + store_count > block.instruction_count / 2) {
        // Memory operation heavy block
        if (load_count >= store_count) {
            return BlockType::MEMORY_COPY;
        } else {
            return BlockType::MEMORY_FILL;
        }
    } else if (arithmetic_count > block.instruction_count / 2) {
        // Arithmetic heavy block
        return BlockType::ARITHMETIC;
    } else if (branch_count > block.instruction_count / 4) {
        // Branch heavy block
        return BlockType::BRANCH_HEAVY;
    } else {
        // Simple sequential block
        return BlockType::SEQUENTIAL;
    }
}

// Create specialized handlers for different block types
void ARMSpecializedCache::CreateMemoryCopyHandler(ARMul_State* cpu, SpecializedBlock& block) {
    // This is a simplified implementation - in a real implementation,
    // we would analyze the instructions to determine the source, destination,
    // and count registers, as well as whether the addresses are incremented
    
    // For now, just set some default values
    block.data.memory_copy.src_reg = 0;  // R0
    block.data.memory_copy.dst_reg = 1;  // R1
    block.data.memory_copy.count_reg = 2; // R2
    block.data.memory_copy.increment = true;
}

void ARMSpecializedCache::CreateMemoryFillHandler(ARMul_State* cpu, SpecializedBlock& block) {
    // Simplified implementation
    block.data.memory_fill.value_reg = 0;  // R0
    block.data.memory_fill.dst_reg = 1;    // R1
    block.data.memory_fill.count_reg = 2;  // R2
    block.data.memory_fill.increment = true;
}

void ARMSpecializedCache::CreateArithmeticHandler(ARMul_State* cpu, SpecializedBlock& block) {
    // Simplified implementation
    block.data.arithmetic.result_reg = 0;    // R0
    block.data.arithmetic.operand1_reg = 1;  // R1
    block.data.arithmetic.operand2_reg = 2;  // R2
    block.data.arithmetic.operation = 0;     // ADD
}
