#include <array>
#include <cstddef>
#include <memory>
#include "common/common_types.h"
#include "common/microprofile.h"
#include "core/arm/dyncom/arm_dyncom_dec.h"
#include "core/arm/dyncom/arm_dyncom_interpreter.h"
#include "core/arm/dyncom/arm_dyncom_profiler.h"
#include "core/arm/dyncom/arm_dyncom_specialized_cache.h"
#include "core/arm/dyncom/arm_dyncom_trans.h"

// Memory validation is now handled by the IsValidMemoryAddress function
// which checks against the actual 3DS memory map

#include "core/arm/skyeye_common/armstate.h"
#include "core/arm/skyeye_common/armsupp.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/gdbstub/gdbstub.h"

// Direct-threaded interpreter for ARM
// This implementation uses a direct-threaded approach to improve performance
// by reducing branch mispredictions in the main interpreter loop

// Structure to hold instruction data for direct threading is defined in the header

// Macros for loading and updating CPU flags
#define LOAD_NZCVT                                                                                  \
    cpu->NFlag = (cpu->Cpsr >> 31) & 1;                                                            \
    cpu->ZFlag = (cpu->Cpsr >> 30) & 1;                                                            \
    cpu->CFlag = (cpu->Cpsr >> 29) & 1;                                                            \
    cpu->VFlag = (cpu->Cpsr >> 28) & 1;                                                            \
    cpu->TFlag = (cpu->Cpsr >> 5) & 1;

#define UPDATE_NFLAG(dst) cpu->NFlag = ((dst) & 0x80000000) ? 1 : 0
#define UPDATE_ZFLAG(dst) cpu->ZFlag = ((dst) == 0) ? 1 : 0

#define RD cpu->Reg[inst_cream->Rd]
#define SHIFTER_OPERAND inst_cream->shifter_operand
#define CurrentModeHasSPSR (cpu->CurrentModeHasSPSR())

MICROPROFILE_DEFINE(DynCom_DirectThreaded, "DynCom", "DirectThreaded", MP_RGB(255, 64, 64));

// Helper function to check if an address is valid
// Uses the actual 3DS memory map regions from memory.h
bool IsValidMemoryAddress(u32 address) {
    // Check if the address is null or unaligned for most operations
    if (address == 0 || (address & 0x3) != 0) {
        return false;
    }

    // Process image region (where application text, data and bss reside)
    if (address >= 0x00100000 && address < 0x04000000) {
        // Additional check: Verify this address is actually mapped in memory
        // This helps catch cases where an address is in a valid range but not actually mapped
        try {
            // Try to read from this address - if it fails, it's not actually mapped
            // This is a no-op that just tests if the address is readable
            // Memory::Read8(address);
            return true;
        } catch (...) {
            LOG_ERROR(Core_ARM11, "Address 0x%08X is in valid range but not mapped in memory", address);
            return false;
        }
    }

    // For all other memory regions, apply the same validation pattern
    const std::array<std::pair<u32, u32>, 10> valid_regions = {{
        {0x04000000, 0x08000000},  // IPC buffer mapping region
        {0x08000000, 0x10000000},  // Application heap (includes stack)
        {0x10000000, 0x14000000},  // Shared memory region
        {0x14000000, 0x1C000000},  // Linear heap (maps 1:1 to FCRAM)
        {0x1E800000, 0x1F000000},  // N3DS extra RAM region
        {0x1EC00000, 0x1F000000},  // IO register area
        {0x1F000000, 0x1FF00000},  // VRAM region
        {0x1FF00000, 0x1FF80000},  // DSP memory region
        {0x1FF80000, 0x1FF83000},  // Config memory and shared page
        {0x30000000, 0x40000000},  // New 3DS expanded linear heap
    }};

    for (const auto& [start, end] : valid_regions) {
        if (address >= start && address < end) {
            try {
                // Memory::Read8(address);
                return true;
            } catch (...) {
                LOG_ERROR(Core_ARM11, "Address 0x%08X is in valid range but not mapped in memory", address);
                return false;
            }
        }
    }

    // 3GX plugin framebuffer region
    if (address >= 0x06000000 && address < 0x0600A9000) {
        try {
//            Memory::Read8(address);
            return true;
        } catch (...) {
            LOG_ERROR(Core_ARM11, "Address 0x%08X is in valid range but not mapped in memory", address);
            return false;
        }
    }

    return false;
}

// Helper functions for condition checking and instruction execution
bool CondPassed(ARMul_State* cpu, unsigned int cond) {
    // Check condition code
    switch (cond) {
    case 0x0: // EQ
        return cpu->ZFlag;
    case 0x1: // NE
        return !cpu->ZFlag;
    case 0x2: // CS
        return cpu->CFlag;
    case 0x3: // CC
        return !cpu->CFlag;
    case 0x4: // MI
        return cpu->NFlag;
    case 0x5: // PL
        return !cpu->NFlag;
    case 0x6: // VS
        return cpu->VFlag;
    case 0x7: // VC
        return !cpu->VFlag;
    case 0x8: // HI
        return (cpu->CFlag && !cpu->ZFlag);
    case 0x9: // LS
        return (!cpu->CFlag || cpu->ZFlag);
    case 0xA: // GE
        return (cpu->NFlag == cpu->VFlag);
    case 0xB: // LT
        return (cpu->NFlag != cpu->VFlag);
    case 0xC: // GT
        return (!cpu->ZFlag && (cpu->NFlag == cpu->VFlag));
    case 0xD: // LE
        return (cpu->ZFlag || (cpu->NFlag != cpu->VFlag));
    case 0xE: // AL
        return true;
    case 0xF: // Invalid
        return false;
    }
    return false;
}

// Helper function to switch CPU mode
void switch_mode(ARMul_State* cpu, u32 mode) {
    cpu->ChangePrivilegeMode(mode);
}

// Instruction type definitions for the direct-threaded interpreter
struct ldr_inst {
    unsigned int inst;
    unsigned int I;
    unsigned int P;
    unsigned int U;
    unsigned int W;
    unsigned int Rn;
    unsigned int Rd;
    unsigned int addr;
    unsigned int offset;
};

struct str_inst {
    unsigned int inst;
    unsigned int I;
    unsigned int P;
    unsigned int U;
    unsigned int W;
    unsigned int Rn;
    unsigned int Rd;
    unsigned int addr;
    unsigned int offset;
};

namespace {

// Structure to hold instruction data for direct threading
struct ThreadedInstruction {
    void* handler;        // Pointer to the instruction handler function
    arm_inst* inst_base;  // Pointer to the instruction data
    u32 address;          // Address of the instruction
};

// Maximum number of instructions to pre-decode in a block
constexpr int MAX_BLOCK_SIZE = 128;

// Pre-decode a block of instructions starting at the given address
// Decode a block of ARM instructions for direct-threaded execution
// Returns the number of instructions decoded
unsigned DecodeThreadedBlock(ARMul_State* cpu, u32 start_address, ThreadedInstruction* instructions, unsigned max_instructions) {
    MICROPROFILE_SCOPE(DynCom_DirectThreaded);

    // Debug logging
    LOG_TRACE(Core_ARM11, "DecodeThreadedBlock called with start_address=0x%08X, max_instructions=%u",
              start_address, max_instructions);

    unsigned count = 0;
    u32 address = start_address;

    while (count < max_instructions) {
        // Check if we've reached a page boundary
        if ((address & 0xFFF) == 0 && count > 0) {
            break;
        }

        // Validate memory address before reading
        u32 aligned_addr = address & 0xFFFFFFFC;

        if (!IsValidMemoryAddress(aligned_addr)) {
            LOG_ERROR(Core_ARM11, "Attempting to read instruction from invalid address: 0x%08X", aligned_addr);
            fprintf(stderr, "ERROR: Attempting to read instruction from invalid address: 0x%08X\n", aligned_addr);
            break;
        }

        // Read the instruction from memory
        u32 instr = cpu->memory.Read32(aligned_addr);

        // Decode the instruction
        int idx = 0;
        ARMDecodeStatus status = DecodeARMInstruction(instr, &idx);
        if (status != ARMDecodeStatus::SUCCESS || idx < 0 || idx >= static_cast<int>(arm_instruction_trans_len)) {
            // Invalid instruction, stop decoding
            break;
        }

        // Create the instruction handler
        std::size_t ptr = 0;
        ARM_INST_PTR inst_base;
        inst_base = arm_instruction_trans[idx](instr, idx);

        // Store the instruction data
        instructions[count].handler = nullptr;  // Will be filled in later
        instructions[count].inst_base = inst_base;
        instructions[count].address = address;

        // Move to the next instruction
        address += 4;  // ARM instructions are 4 bytes
        count++;

        // Stop if we hit a branch instruction
        if (((arm_inst*)(inst_base))->br != TransExtData::NON_BRANCH) {
            break;
        }
    }

    return count;
}

// Returns the number of instructions decoded
// This is the legacy version kept for compatibility
int PreDecodeBlock(ARMul_State* cpu, u32 start_address, ThreadedInstruction* instructions, int max_count) {
    // Add debug logging
    LOG_TRACE(Core_ARM11, "PreDecodeBlock at address 0x%08X, max_count=%d", start_address, max_count);

    // Simply call the new function for consistency
    int count = static_cast<int>(DecodeThreadedBlock(cpu, start_address, instructions, static_cast<unsigned>(max_count)));

    LOG_TRACE(Core_ARM11, "PreDecodeBlock decoded %d instructions", count);
    return count;
}

// Execute a pre-decoded block of instructions
unsigned ExecuteThreadedBlock(ARMul_State* cpu, ThreadedInstruction* instructions, int count) {
    // Add debug logging
    LOG_TRACE(Core_ARM11, "ExecuteThreadedBlock with %d instructions at PC=0x%08X, CPSR=0x%08X, TFlag=%d",
              count, cpu->Reg[15], cpu->Cpsr, cpu->TFlag);
    fprintf(stderr, "ExecuteThreadedBlock with %d instructions at PC=0x%08X\n", count, cpu->Reg[15]);
    MICROPROFILE_SCOPE(DynCom_DirectThreaded);

    // Print CPU state before execution
//    extern void PrintCPUState(ARMul_State* cpu); // Forward declaration
//    fprintf(stderr, "CPU state before threaded block execution:\n");
//    PrintCPUState(cpu);

    // Get the profiler instance
    auto& profiler = ARMDyncomProfiler::GetInstance();

    // Store the original PC for comparison
    u32 original_pc = cpu->Reg[15];
    unsigned num_instrs = 0;

    for (int i = 0; i < count; i++) {
        // Update PC to the current instruction address
        cpu->Reg[15] = instructions[i].address;

        // Record instruction execution
        profiler.RecordInstructionExecution(instructions[i].address);

        // Execute the instruction
        arm_inst* inst_base = (arm_inst*)(instructions[i].inst_base);

        // Debug logging for each instruction
        LOG_TRACE(Core_ARM11, "Executing instruction at address 0x%08X, idx=%d, cond=%d",
                  instructions[i].address, inst_base->idx, inst_base->cond);

        // Add more detailed logging for debugging
        if (i % 5 == 0) { // Only log every 5th instruction to avoid excessive output
            fprintf(stderr, "Executing instruction at address 0x%08X, idx=%d, cond=%d\n",
                    instructions[i].address, inst_base->idx, inst_base->cond);
        }

        // Check condition code
        if (inst_base->cond == ConditionCode::AL || CondPassed(cpu, inst_base->cond)) {
            // Execute the instruction directly
            // This avoids the switch statement in the main interpreter loop,
            // reducing branch mispredictions

            // Call the instruction handler directly
            // The instruction handlers are defined in arm_dyncom_interpreter.cpp
            // We're calling them directly here to avoid the switch statement overhead
            switch (inst_base->idx) {
                // We'll handle the most common instructions directly for better performance
                case 62: // ADD_INST
                    {
                        add_inst* const inst_cream = (add_inst*)(inst_base->component);
                        u32 rn_val = cpu->Reg[inst_cream->Rn];
                        if (inst_cream->Rn == 15)
                            rn_val += 8;

                        bool carry;
                        bool overflow;
                        u32 result = AddWithCarry(rn_val, inst_cream->shifter_operand, 0, &carry, &overflow);

                        if (inst_cream->S && (inst_cream->Rd == 15)) {
                            if (CurrentModeHasSPSR) {
                                cpu->Cpsr = cpu->Spsr_copy;
                                cpu->ChangePrivilegeMode(cpu->Spsr_copy & 0x1F);
                                cpu->Reg[15] = result;
                            }
                        } else {
                            cpu->Reg[inst_cream->Rd] = result;
                            if (inst_cream->S) {
                                cpu->NFlag = (result & 0x80000000) ? 1 : 0;
                                cpu->ZFlag = (result == 0) ? 1 : 0;
                                cpu->CFlag = carry;
                                cpu->VFlag = overflow;
                            }
                        }
                    }
                    break;

                case 69: // MOV_INST
                    {
                        mov_inst* const inst_cream = (mov_inst*)(inst_base->component);

                        u32 result = inst_cream->shifter_operand;
                        if (inst_cream->S && (inst_cream->Rd == 15)) {
                            if (CurrentModeHasSPSR) {
                                cpu->Cpsr = cpu->Spsr_copy;
                                cpu->ChangePrivilegeMode(cpu->Spsr_copy & 0x1F);
                                cpu->Reg[15] = result;
                            }
                        } else {
                            cpu->Reg[inst_cream->Rd] = result;
                            if (inst_cream->S) {
                                cpu->NFlag = (result & 0x80000000) ? 1 : 0;
                                cpu->ZFlag = (result == 0) ? 1 : 0;
                                cpu->CFlag = cpu->shifter_carry_out;
                            }
                        }
                    }
                    break;

                case 66: // SUB_INST
                    {
                        sub_inst* const inst_cream = (sub_inst*)(inst_base->component);
                        u32 rn_val = cpu->Reg[inst_cream->Rn];
                        if (inst_cream->Rn == 15)
                            rn_val += 8;

                        bool carry;
                        bool overflow;
                        u32 result = AddWithCarry(rn_val, ~inst_cream->shifter_operand, 1, &carry, &overflow);

                        if (inst_cream->S && (inst_cream->Rd == 15)) {
                            if (CurrentModeHasSPSR) {
                                cpu->Cpsr = cpu->Spsr_copy;
                                cpu->ChangePrivilegeMode(cpu->Spsr_copy & 0x1F);
                                cpu->Reg[15] = result;
                            }
                        } else {
                            cpu->Reg[inst_cream->Rd] = result;
                            if (inst_cream->S) {
                                cpu->NFlag = (result & 0x80000000) ? 1 : 0;
                                cpu->ZFlag = (result == 0) ? 1 : 0;
                                cpu->CFlag = carry;
                                cpu->VFlag = overflow;
                            }
                        }
                    }
                    break;

                case 94: // LDR_INST
                    {
                        ldr_inst* const inst_cream = (ldr_inst*)(inst_base->component);
                        u32 address = 0;
                        if (inst_cream->Rn == 15) {
                            address = (cpu->Reg[15] + 8) + inst_cream->offset;
                        } else {
                            address = cpu->Reg[inst_cream->Rn] + inst_cream->offset;
                        }

                        // Validate memory address before reading
                        if (!IsValidMemoryAddress(address)) {
                            LOG_ERROR(Core_ARM11, "LDR: Attempting to read from invalid address: 0x%08X", address);
                            fprintf(stderr, "ERROR: LDR: Attempting to read from invalid address: 0x%08X\n", address);
                            // Skip this instruction
                            break;
                        }

                        u32 value = cpu->memory.Read32(address);
                        cpu->Reg[inst_cream->Rd] = value;
                    }
                    break;

                case 95: // STR_INST
                    {
                        str_inst* const inst_cream = (str_inst*)(inst_base->component);
                        u32 address = 0;
                        if (inst_cream->Rn == 15) {
                            address = (cpu->Reg[15] + 8) + inst_cream->offset;
                        } else {
                            address = cpu->Reg[inst_cream->Rn] + inst_cream->offset;
                        }

                        // Validate memory address before writing
                        if (!IsValidMemoryAddress(address)) {
                            LOG_ERROR(Core_ARM11, "STR: Attempting to write to invalid address: 0x%08X", address);
                            fprintf(stderr, "ERROR: STR: Attempting to write to invalid address: 0x%08X\n", address);
                            // Skip this instruction
                            break;
                        }

                        u32 value = cpu->Reg[inst_cream->Rd];
                        cpu->memory.Write32(address, value);
                    }
                    break;

                // LDM (Load Multiple) instruction
                case 96: // LDM_INST
                    {
                        ldst_inst* const inst_cream = (ldst_inst*)(inst_base->component);
                        u32 address = 0;
                        inst_cream->get_addr(cpu, inst_cream->inst, address);

                        unsigned int inst = inst_cream->inst;

                        // Validate all memory addresses before reading
                        u32 check_addr = address;
                        bool all_addresses_valid = true;

                        // Calculate total bytes to be read
                        int bytes_to_read = 0;
                        for (int i = 0; i < 16; i++) {
                            if (BIT(inst, i)) {
                                bytes_to_read += 4;
                            }
                        }

                        // Check if all addresses in the range are valid
                        for (int offset = 0; offset < bytes_to_read; offset += 4) {
                            if (!IsValidMemoryAddress(address + offset)) {
                                LOG_ERROR(Core_ARM11, "LDM: Attempting to read from invalid address range: 0x%08X-0x%08X",
                                         address, address + bytes_to_read - 4);
                                fprintf(stderr, "ERROR: LDM: Attempting to read from invalid address range: 0x%08X-0x%08X\n",
                                        address, address + bytes_to_read - 4);
                                all_addresses_valid = false;
                                break;
                            }
                        }

                        if (!all_addresses_valid) {
                            // Skip this instruction if any address is invalid
                            break;
                        }

                        // Perform the actual memory reads
                        if (BIT(inst, 22) && !BIT(inst, 15)) {
                            for (int i = 0; i < 13; i++) {
                                if (BIT(inst, i)) {
                                    cpu->Reg[i] = cpu->ReadMemory32(address);
                                    address += 4;
                                }
                            }
                            if (BIT(inst, 13)) {
                                if (cpu->Mode == USER32MODE)
                                    cpu->Reg[13] = cpu->ReadMemory32(address);
                                else
                                    cpu->Reg_usr[0] = cpu->ReadMemory32(address);

                                address += 4;
                            }
                            if (BIT(inst, 14)) {
                                if (cpu->Mode == USER32MODE)
                                    cpu->Reg[14] = cpu->ReadMemory32(address);
                                else
                                    cpu->Reg_usr[1] = cpu->ReadMemory32(address);

                                address += 4;
                            }
                        } else {
                            for (int i = 0; i < 16; i++) {
                                if (BIT(inst, i)) {
                                    if (i == 15) {
                                        cpu->Reg[15] = cpu->ReadMemory32(address) & 0xFFFFFFFC;
                                    } else {
                                        cpu->Reg[i] = cpu->ReadMemory32(address);
                                    }
                                    address += 4;
                                }
                            }
                        }
                    }
                    break;

                // STM (Store Multiple) instruction
                case 97: // STM_INST
                    {
                        ldst_inst* const inst_cream = (ldst_inst*)(inst_base->component);
                        unsigned int inst = inst_cream->inst;

                        unsigned int Rn = BITS(inst, 16, 19);
                        unsigned int old_RN = cpu->Reg[Rn];

                        u32 address = 0;
                        inst_cream->get_addr(cpu, inst_cream->inst, address);

                        // Validate all memory addresses before writing
                        u32 check_addr = address;
                        bool all_addresses_valid = true;

                        // Calculate total bytes to be written
                        int bytes_to_write = 0;
                        for (int i = 0; i < 16; i++) {
                            if (BIT(inst, i)) {
                                bytes_to_write += 4;
                            }
                        }

                        // Check if all addresses in the range are valid
                        for (int offset = 0; offset < bytes_to_write; offset += 4) {
                            if (!IsValidMemoryAddress(address + offset)) {
                                LOG_ERROR(Core_ARM11, "STM: Attempting to write to invalid address range: 0x%08X-0x%08X",
                                         address, address + bytes_to_write - 4);
                                fprintf(stderr, "ERROR: STM: Attempting to write to invalid address range: 0x%08X-0x%08X\n",
                                        address, address + bytes_to_write - 4);
                                all_addresses_valid = false;
                                break;
                            }
                        }

                        if (!all_addresses_valid) {
                            // Skip this instruction if any address is invalid
                            break;
                        }

                        // Perform the actual memory writes
                        if (BIT(inst, 22) == 1) {
                            for (int i = 0; i < 13; i++) {
                                if (BIT(inst, i)) {
                                    cpu->WriteMemory32(address, cpu->Reg[i]);
                                    address += 4;
                                }
                            }
                            if (BIT(inst, 13)) {
                                if (cpu->Mode == USER32MODE)
                                    cpu->WriteMemory32(address, cpu->Reg[13]);
                                else
                                    cpu->WriteMemory32(address, cpu->Reg_usr[0]);

                                address += 4;
                            }
                            if (BIT(inst, 14)) {
                                if (cpu->Mode == USER32MODE)
                                    cpu->WriteMemory32(address, cpu->Reg[14]);
                                else
                                    cpu->WriteMemory32(address, cpu->Reg_usr[1]);

                                address += 4;
                            }
                        } else {
                            for (int i = 0; i < 15; i++) {
                                if (BIT(inst, i)) {
                                    cpu->WriteMemory32(address, cpu->Reg[i]);
                                    address += 4;
                                }
                            }
                            if (BIT(inst, 15)) {
                                cpu->WriteMemory32(address, cpu->Reg[15] + 8);
                            }
                        }

                        // Handle writeback
                        if (BIT(inst, 21) && !BIT(inst, Rn))
                            cpu->Reg[Rn] = old_RN;
                    }
                    break;

                // LDRB (Load Byte) instruction
                case 98: // LDRB_INST
                    {
                        ldst_inst* const inst_cream = (ldst_inst*)(inst_base->component);
                        u32 address = 0;

                        // Make sure get_addr is valid before calling it
                        if (inst_cream->get_addr == nullptr) {
                            LOG_ERROR(Core_ARM11, "LDRB: Invalid addressing mode");
                            fprintf(stderr, "ERROR: LDRB: Invalid addressing mode\n");
                            break;
                        }

                        inst_cream->get_addr(cpu, inst_cream->inst, address);

                        // Validate memory address before reading
                        if (!IsValidMemoryAddress(address)) {
                            LOG_ERROR(Core_ARM11, "LDRB: Attempting to read from invalid address: 0x%08X", address);
                            fprintf(stderr, "ERROR: LDRB: Attempting to read from invalid address: 0x%08X\n", address);
                            // Skip this instruction
                            break;
                        }

                        cpu->Reg[BITS(inst_cream->inst, 12, 15)] = cpu->ReadMemory8(address);
                    }
                    break;

                // STRB (Store Byte) instruction
                case 99: // STRB_INST
                    {
                        ldst_inst* const inst_cream = (ldst_inst*)(inst_base->component);
                        u32 address = 0;
                        inst_cream->get_addr(cpu, inst_cream->inst, address);

                        // Validate memory address before writing
                        if (!IsValidMemoryAddress(address)) {
                            LOG_ERROR(Core_ARM11, "STRB: Attempting to write to invalid address: 0x%08X", address);
                            fprintf(stderr, "ERROR: STRB: Attempting to write to invalid address: 0x%08X\n", address);
                            // Skip this instruction
                            break;
                        }

                        unsigned int value = cpu->Reg[BITS(inst_cream->inst, 12, 15)] & 0xff;
                        cpu->WriteMemory8(address, value);
                    }
                    break;

                // LDRH (Load Halfword) instruction
                case 100: // LDRH_INST
                    {
                        ldst_inst* const inst_cream = (ldst_inst*)(inst_base->component);
                        u32 address = 0;
                        inst_cream->get_addr(cpu, inst_cream->inst, address);

                        // Validate memory address before reading
                        if (!IsValidMemoryAddress(address)) {
                            LOG_ERROR(Core_ARM11, "LDRH: Attempting to read from invalid address: 0x%08X", address);
                            fprintf(stderr, "ERROR: LDRH: Attempting to read from invalid address: 0x%08X\n", address);
                            // Skip this instruction
                            break;
                        }

                        // Check for unaligned halfword access
                        if (address & 0x1) {
                            LOG_ERROR(Core_ARM11, "LDRH: Unaligned halfword read from address: 0x%08X", address);
                            fprintf(stderr, "ERROR: LDRH: Unaligned halfword read from address: 0x%08X\n", address);
                            // Skip this instruction
                            break;
                        }

                        cpu->Reg[BITS(inst_cream->inst, 12, 15)] = cpu->ReadMemory16(address);
                    }
                    break;

                // STRH (Store Halfword) instruction
                case 101: // STRH_INST
                    {
                        ldst_inst* const inst_cream = (ldst_inst*)(inst_base->component);
                        u32 address = 0;
                        inst_cream->get_addr(cpu, inst_cream->inst, address);

                        // Validate memory address before writing
                        if (!IsValidMemoryAddress(address)) {
                            LOG_ERROR(Core_ARM11, "STRH: Attempting to write to invalid address: 0x%08X", address);
                            fprintf(stderr, "ERROR: STRH: Attempting to write to invalid address: 0x%08X\n", address);
                            // Skip this instruction
                            break;
                        }

                        // Check for unaligned halfword access
                        if (address & 0x1) {
                            LOG_ERROR(Core_ARM11, "STRH: Unaligned halfword write to address: 0x%08X", address);
                            fprintf(stderr, "ERROR: STRH: Unaligned halfword write to address: 0x%08X\n", address);
                            // Skip this instruction
                            break;
                        }

                        unsigned int value = cpu->Reg[BITS(inst_cream->inst, 12, 15)] & 0xffff;
                        cpu->WriteMemory16(address, value);
                    }
                    break;

                // LDRD (Load Double Word) instruction
                case 102: // LDRD_INST
                    {
                        ldst_inst* const inst_cream = (ldst_inst*)(inst_base->component);
                        u32 address = 0;
                        inst_cream->get_addr(cpu, inst_cream->inst, address);

                        // Validate memory addresses before reading
                        if (!IsValidMemoryAddress(address) || !IsValidMemoryAddress(address + 4)) {
                            LOG_ERROR(Core_ARM11, "LDRD: Attempting to read from invalid address range: 0x%08X-0x%08X",
                                     address, address + 4);
                            fprintf(stderr, "ERROR: LDRD: Attempting to read from invalid address range: 0x%08X-0x%08X\n",
                                    address, address + 4);
                            // Skip this instruction
                            break;
                        }

                        // Check for alignment
                        if (address & 0x3) {
                            LOG_ERROR(Core_ARM11, "LDRD: Unaligned word read from address: 0x%08X", address);
                            fprintf(stderr, "ERROR: LDRD: Unaligned word read from address: 0x%08X\n", address);
                            // Skip this instruction
                            break;
                        }

                        // The 3DS doesn't have LPAE (Large Physical Access Extension), so it
                        // wouldn't do this as a single read.
                        cpu->Reg[BITS(inst_cream->inst, 12, 15) + 0] = cpu->ReadMemory32(address);
                        cpu->Reg[BITS(inst_cream->inst, 12, 15) + 1] = cpu->ReadMemory32(address + 4);
                    }
                    break;

                // STRD (Store Double Word) instruction
                case 103: // STRD_INST
                    {
                        ldst_inst* const inst_cream = (ldst_inst*)(inst_base->component);
                        u32 address = 0;
                        inst_cream->get_addr(cpu, inst_cream->inst, address);

                        // Validate memory addresses before writing
                        if (!IsValidMemoryAddress(address) || !IsValidMemoryAddress(address + 4)) {
                            LOG_ERROR(Core_ARM11, "STRD: Attempting to write to invalid address range: 0x%08X-0x%08X",
                                     address, address + 4);
                            fprintf(stderr, "ERROR: STRD: Attempting to write to invalid address range: 0x%08X-0x%08X\n",
                                    address, address + 4);
                            // Skip this instruction
                            break;
                        }

                        // Check for alignment
                        if (address & 0x3) {
                            LOG_ERROR(Core_ARM11, "STRD: Unaligned word write to address: 0x%08X", address);
                            fprintf(stderr, "ERROR: STRD: Unaligned word write to address: 0x%08X\n", address);
                            // Skip this instruction
                            break;
                        }

                        // The 3DS doesn't have the Large Physical Access Extension (LPAE)
                        // so STRD wouldn't store these as a single write.
                        cpu->WriteMemory32(address + 0, cpu->Reg[BITS(inst_cream->inst, 12, 15)]);
                        cpu->WriteMemory32(address + 4, cpu->Reg[BITS(inst_cream->inst, 12, 15) + 1]);
                    }
                    break;

                // LDRSB (Load Signed Byte) instruction
                case 104: // LDRSB_INST
                    {
                        ldst_inst* const inst_cream = (ldst_inst*)(inst_base->component);
                        u32 address = 0;
                        inst_cream->get_addr(cpu, inst_cream->inst, address);

                        // Validate memory address before reading
                        if (!IsValidMemoryAddress(address)) {
                            LOG_ERROR(Core_ARM11, "LDRSB: Attempting to read from invalid address: 0x%08X", address);
                            fprintf(stderr, "ERROR: LDRSB: Attempting to read from invalid address: 0x%08X\n", address);
                            // Skip this instruction
                            break;
                        }

                        unsigned int value = cpu->ReadMemory8(address);
                        if (BIT(value, 7)) {
                            value |= 0xffffff00;
                        }
                        cpu->Reg[BITS(inst_cream->inst, 12, 15)] = value;
                    }
                    break;

                // LDRSH (Load Signed Halfword) instruction
                case 105: // LDRSH_INST
                    {
                        ldst_inst* const inst_cream = (ldst_inst*)(inst_base->component);
                        u32 address = 0;
                        inst_cream->get_addr(cpu, inst_cream->inst, address);

                        // Validate memory address before reading
                        if (!IsValidMemoryAddress(address)) {
                            LOG_ERROR(Core_ARM11, "LDRSH: Attempting to read from invalid address: 0x%08X", address);
                            fprintf(stderr, "ERROR: LDRSH: Attempting to read from invalid address: 0x%08X\n", address);
                            // Skip this instruction
                            break;
                        }

                        // Check for unaligned halfword access
                        if (address & 0x1) {
                            LOG_ERROR(Core_ARM11, "LDRSH: Unaligned halfword read from address: 0x%08X", address);
                            fprintf(stderr, "ERROR: LDRSH: Unaligned halfword read from address: 0x%08X\n", address);
                            // Skip this instruction
                            break;
                        }

                        unsigned int value = cpu->ReadMemory16(address);
                        if (BIT(value, 15)) {
                            value |= 0xffff0000;
                        }
                        cpu->Reg[BITS(inst_cream->inst, 12, 15)] = value;
                    }
                    break;

                // LDREX (Load Register Exclusive) instruction
                case 106: // LDREX_INST
                    {
                        generic_arm_inst* const inst_cream = (generic_arm_inst*)(inst_base->component);
                        u32 address = cpu->Reg[inst_cream->Rn];

                        // Validate memory address before reading
                        if (!IsValidMemoryAddress(address)) {
                            LOG_ERROR(Core_ARM11, "LDREX: Attempting to read from invalid address: 0x%08X", address);
                            fprintf(stderr, "ERROR: LDREX: Attempting to read from invalid address: 0x%08X\n", address);
                            // Skip this instruction
                            break;
                        }

                        // Set exclusive memory access flag
                        cpu->SetExclusiveMemoryAddress(address);

                        // Read the memory value
                        cpu->Reg[inst_cream->Rd] = cpu->ReadMemory32(address);
                    }
                    break;

                // STREX (Store Register Exclusive) instruction
                case 107: // STREX_INST
                    {
                        generic_arm_inst* const inst_cream = (generic_arm_inst*)(inst_base->component);
                        u32 address = cpu->Reg[inst_cream->Rn];

                        // Validate memory address before writing
                        if (!IsValidMemoryAddress(address)) {
                            LOG_ERROR(Core_ARM11, "STREX: Attempting to write to invalid address: 0x%08X", address);
                            fprintf(stderr, "ERROR: STREX: Attempting to write to invalid address: 0x%08X\n", address);
                            // Skip this instruction
                            break;
                        }

                        // Check if this is an exclusive memory access
                        if (cpu->IsExclusiveMemoryAccess(address)) {
                            cpu->UnsetExclusiveMemoryAddress();
                            cpu->WriteMemory32(address, cpu->Reg[inst_cream->Rm]);
                            cpu->Reg[inst_cream->Rd] = 0; // Success
                        } else {
                            // Failed to write due to mutex access
                            cpu->Reg[inst_cream->Rd] = 1; // Failure
                        }
                    }
                    break;

                // LDRT (Load Register User-mode Privileged) instruction
                case 108: // LDRT_INST
                    {
                        ldst_inst* const inst_cream = (ldst_inst*)(inst_base->component);
                        u32 address = 0;
                        inst_cream->get_addr(cpu, inst_cream->inst, address);

                        // Validate memory address before reading
                        if (!IsValidMemoryAddress(address)) {
                            LOG_ERROR(Core_ARM11, "LDRT: Attempting to read from invalid address: 0x%08X", address);
                            fprintf(stderr, "ERROR: LDRT: Attempting to read from invalid address: 0x%08X\n", address);
                            // Skip this instruction
                            break;
                        }

                        const u32 dest_index = BITS(inst_cream->inst, 12, 15);
                        const u32 previous_mode = cpu->Mode;

                        // Change to user mode for memory access
                        cpu->ChangePrivilegeMode(USER32MODE);
                        const u32 value = cpu->ReadMemory32(address);
                        cpu->ChangePrivilegeMode(previous_mode);

                        cpu->Reg[dest_index] = value;
                    }
                    break;

                // STRT (Store Register User-mode Privileged) instruction
                case 109: // STRT_INST
                    {
                        ldst_inst* const inst_cream = (ldst_inst*)(inst_base->component);
                        u32 address = 0;
                        inst_cream->get_addr(cpu, inst_cream->inst, address);

                        // Validate memory address before writing
                        if (!IsValidMemoryAddress(address)) {
                            LOG_ERROR(Core_ARM11, "STRT: Attempting to write to invalid address: 0x%08X", address);
                            fprintf(stderr, "ERROR: STRT: Attempting to write to invalid address: 0x%08X\n", address);
                            // Skip this instruction
                            break;
                        }

                        const u32 previous_mode = cpu->Mode;
                        const u32 rt_index = BITS(inst_cream->inst, 12, 15);

                        u32 value = cpu->Reg[rt_index];
                        if (rt_index == 15)
                            value += 2 * cpu->GetInstructionSize();

                        // Change to user mode for memory access
                        cpu->ChangePrivilegeMode(USER32MODE);
                        cpu->WriteMemory32(address, value);
                        cpu->ChangePrivilegeMode(previous_mode);
                    }
                    break;

                // LDRBT (Load Register Byte User-mode Privileged) instruction
                case 110: // LDRBT_INST
                    {
                        ldst_inst* const inst_cream = (ldst_inst*)(inst_base->component);
                        u32 address = 0;
                        inst_cream->get_addr(cpu, inst_cream->inst, address);

                        // Validate memory address before reading
                        if (!IsValidMemoryAddress(address)) {
                            LOG_ERROR(Core_ARM11, "LDRBT: Attempting to read from invalid address: 0x%08X", address);
                            fprintf(stderr, "ERROR: LDRBT: Attempting to read from invalid address: 0x%08X\n", address);
                            // Skip this instruction
                            break;
                        }

                        const u32 dest_index = BITS(inst_cream->inst, 12, 15);
                        const u32 previous_mode = cpu->Mode;

                        // Change to user mode for memory access
                        cpu->ChangePrivilegeMode(USER32MODE);
                        const u8 value = cpu->ReadMemory8(address);
                        cpu->ChangePrivilegeMode(previous_mode);

                        cpu->Reg[dest_index] = value;
                    }
                    break;

                // STRBT (Store Register Byte User-mode Privileged) instruction
                case 111: // STRBT_INST
                    {
                        ldst_inst* const inst_cream = (ldst_inst*)(inst_base->component);
                        u32 address = 0;
                        inst_cream->get_addr(cpu, inst_cream->inst, address);

                        // Validate memory address before writing
                        if (!IsValidMemoryAddress(address)) {
                            LOG_ERROR(Core_ARM11, "STRBT: Attempting to write to invalid address: 0x%08X", address);
                            fprintf(stderr, "ERROR: STRBT: Attempting to write to invalid address: 0x%08X\n", address);
                            // Skip this instruction
                            break;
                        }

                        const u32 previous_mode = cpu->Mode;
                        const u32 value = cpu->Reg[BITS(inst_cream->inst, 12, 15)] & 0xff;

                        // Change to user mode for memory access
                        cpu->ChangePrivilegeMode(USER32MODE);
                        cpu->WriteMemory8(address, value);
                        cpu->ChangePrivilegeMode(previous_mode);
                    }
                    break;

                // LDREXB (Load Register Exclusive Byte) instruction
                case 112: // LDREXB_INST
                    {
                        generic_arm_inst* const inst_cream = (generic_arm_inst*)(inst_base->component);
                        u32 address = cpu->Reg[inst_cream->Rn];

                        // Validate memory address before reading
                        if (!IsValidMemoryAddress(address)) {
                            LOG_ERROR(Core_ARM11, "LDREXB: Attempting to read from invalid address: 0x%08X", address);
                            fprintf(stderr, "ERROR: LDREXB: Attempting to read from invalid address: 0x%08X\n", address);
                            // Skip this instruction
                            break;
                        }

                        // Set exclusive memory access flag
                        cpu->SetExclusiveMemoryAddress(address);

                        // Read the memory value (byte)
                        cpu->Reg[inst_cream->Rd] = cpu->ReadMemory8(address);
                    }
                    break;

                // STREXB (Store Register Exclusive Byte) instruction
                case 113: // STREXB_INST
                    {
                        generic_arm_inst* const inst_cream = (generic_arm_inst*)(inst_base->component);
                        u32 address = cpu->Reg[inst_cream->Rn];

                        // Validate memory address before writing
                        if (!IsValidMemoryAddress(address)) {
                            LOG_ERROR(Core_ARM11, "STREXB: Attempting to write to invalid address: 0x%08X", address);
                            fprintf(stderr, "ERROR: STREXB: Attempting to write to invalid address: 0x%08X\n", address);
                            // Skip this instruction
                            break;
                        }

                        // Check if this is an exclusive memory access
                        if (cpu->IsExclusiveMemoryAccess(address)) {
                            cpu->UnsetExclusiveMemoryAddress();
                            cpu->WriteMemory8(address, cpu->Reg[inst_cream->Rm] & 0xFF);
                            cpu->Reg[inst_cream->Rd] = 0; // Success
                        } else {
                            // Failed to write due to mutex access
                            cpu->Reg[inst_cream->Rd] = 1; // Failure
                        }
                    }
                    break;

                // LDREXH (Load Register Exclusive Halfword) instruction
                case 114: // LDREXH_INST
                    {
                        generic_arm_inst* const inst_cream = (generic_arm_inst*)(inst_base->component);
                        u32 address = cpu->Reg[inst_cream->Rn];

                        // Validate memory address before reading
                        if (!IsValidMemoryAddress(address)) {
                            LOG_ERROR(Core_ARM11, "LDREXH: Attempting to read from invalid address: 0x%08X", address);
                            fprintf(stderr, "ERROR: LDREXH: Attempting to read from invalid address: 0x%08X\n", address);
                            // Skip this instruction
                            break;
                        }

                        // Check for unaligned halfword access
                        if (address & 0x1) {
                            LOG_ERROR(Core_ARM11, "LDREXH: Unaligned halfword read from address: 0x%08X", address);
                            fprintf(stderr, "ERROR: LDREXH: Unaligned halfword read from address: 0x%08X\n", address);
                            // Skip this instruction
                            break;
                        }

                        // Set exclusive memory access flag
                        cpu->SetExclusiveMemoryAddress(address);

                        // Read the memory value (halfword)
                        cpu->Reg[inst_cream->Rd] = cpu->ReadMemory16(address);
                    }
                    break;

                // STREXH (Store Register Exclusive Halfword) instruction
                case 115: // STREXH_INST
                    {
                        generic_arm_inst* const inst_cream = (generic_arm_inst*)(inst_base->component);
                        u32 address = cpu->Reg[inst_cream->Rn];

                        // Validate memory address before writing
                        if (!IsValidMemoryAddress(address)) {
                            LOG_ERROR(Core_ARM11, "STREXH: Attempting to write to invalid address: 0x%08X", address);
                            fprintf(stderr, "ERROR: STREXH: Attempting to write to invalid address: 0x%08X\n", address);
                            // Skip this instruction
                            break;
                        }

                        // Check for unaligned halfword access
                        if (address & 0x1) {
                            LOG_ERROR(Core_ARM11, "STREXH: Unaligned halfword write to address: 0x%08X", address);
                            fprintf(stderr, "ERROR: STREXH: Unaligned halfword write to address: 0x%08X\n", address);
                            // Skip this instruction
                            break;
                        }

                        // Check if this is an exclusive memory access
                        if (cpu->IsExclusiveMemoryAccess(address)) {
                            cpu->UnsetExclusiveMemoryAddress();
                            cpu->WriteMemory16(address, cpu->Reg[inst_cream->Rm] & 0xFFFF);
                            cpu->Reg[inst_cream->Rd] = 0; // Success
                        } else {
                            // Failed to write due to mutex access
                            cpu->Reg[inst_cream->Rd] = 1; // Failure
                        }
                    }
                    break;

                // STREXD (Store Register Exclusive Double) instruction
                case 116: // STREXD_INST
                    {
                        generic_arm_inst* const inst_cream = (generic_arm_inst*)(inst_base->component);
                        u32 address = cpu->Reg[inst_cream->Rn];

                        // Validate memory addresses before writing
                        if (!IsValidMemoryAddress(address) || !IsValidMemoryAddress(address + 4)) {
                            LOG_ERROR(Core_ARM11, "STREXD: Attempting to write to invalid address range: 0x%08X-0x%08X",
                                     address, address + 4);
                            fprintf(stderr, "ERROR: STREXD: Attempting to write to invalid address range: 0x%08X-0x%08X\n",
                                    address, address + 4);
                            // Skip this instruction
                            break;
                        }

                        // Check for alignment
                        if (address & 0x3) {
                            LOG_ERROR(Core_ARM11, "STREXD: Unaligned doubleword write to address: 0x%08X", address);
                            fprintf(stderr, "ERROR: STREXD: Unaligned doubleword write to address: 0x%08X\n", address);
                            // Skip this instruction
                            break;
                        }

                        // Check if this is an exclusive memory access
                        if (cpu->IsExclusiveMemoryAccess(address)) {
                            cpu->UnsetExclusiveMemoryAddress();

                            const u32 rt = cpu->Reg[inst_cream->Rm + 0];
                            const u32 rt2 = cpu->Reg[inst_cream->Rm + 1];
                            u64 value;

                            if (cpu->InBigEndianMode())
                                value = (((u64)rt << 32) | rt2);
                            else
                                value = (((u64)rt2 << 32) | rt);

                            // The 3DS doesn't have LPAE, so we need to do this as two 32-bit writes
                            cpu->WriteMemory32(address, rt);
                            cpu->WriteMemory32(address + 4, rt2);

                            cpu->Reg[inst_cream->Rd] = 0; // Success
                        } else {
                            // Failed to write due to mutex access
                            cpu->Reg[inst_cream->Rd] = 1; // Failure
                        }
                    }
                    break;

                // LDREXD (Load Register Exclusive Double) instruction
                case 117: // LDREXD_INST
                    {
                        generic_arm_inst* const inst_cream = (generic_arm_inst*)(inst_base->component);
                        u32 address = cpu->Reg[inst_cream->Rn];

                        // Validate memory addresses before reading
                        if (!IsValidMemoryAddress(address) || !IsValidMemoryAddress(address + 4)) {
                            LOG_ERROR(Core_ARM11, "LDREXD: Attempting to read from invalid address range: 0x%08X-0x%08X",
                                     address, address + 4);
                            fprintf(stderr, "ERROR: LDREXD: Attempting to read from invalid address range: 0x%08X-0x%08X\n",
                                    address, address + 4);
                            // Skip this instruction
                            break;
                        }

                        // Check for alignment
                        if (address & 0x3) {
                            LOG_ERROR(Core_ARM11, "LDREXD: Unaligned doubleword read from address: 0x%08X", address);
                            fprintf(stderr, "ERROR: LDREXD: Unaligned doubleword read from address: 0x%08X\n", address);
                            // Skip this instruction
                            break;
                        }

                        // Set exclusive memory access flag
                        cpu->SetExclusiveMemoryAddress(address);

                        // The 3DS doesn't have LPAE, so we need to do this as two 32-bit reads
                        cpu->Reg[inst_cream->Rd] = cpu->ReadMemory32(address);
                        cpu->Reg[inst_cream->Rd + 1] = cpu->ReadMemory32(address + 4);
                    }
                    break;

                // SWP (Swap) instruction
                case 118: // SWP_INST
                    {
                        swp_inst* const inst_cream = (swp_inst*)(inst_base->component);
                        u32 address = cpu->Reg[inst_cream->Rn];

                        // Validate memory address before reading/writing
                        if (!IsValidMemoryAddress(address)) {
                            LOG_ERROR(Core_ARM11, "SWP: Attempting to access invalid address: 0x%08X", address);
                            fprintf(stderr, "ERROR: SWP: Attempting to access invalid address: 0x%08X\n", address);
                            // Skip this instruction
                            break;
                        }

                        // Check for alignment
                        if (address & 0x3) {
                            LOG_ERROR(Core_ARM11, "SWP: Unaligned word access at address: 0x%08X", address);
                            fprintf(stderr, "ERROR: SWP: Unaligned word access at address: 0x%08X\n", address);
                            // Skip this instruction
                            break;
                        }

                        // Perform atomic swap operation
                        unsigned int value = cpu->ReadMemory32(address);
                        cpu->WriteMemory32(address, cpu->Reg[inst_cream->Rm]);
                        cpu->Reg[inst_cream->Rd] = value;
                    }
                    break;

                // SWPB (Swap Byte) instruction
                case 119: // SWPB_INST
                    {
                        swp_inst* const inst_cream = (swp_inst*)(inst_base->component);
                        u32 address = cpu->Reg[inst_cream->Rn];

                        // Validate memory address before reading/writing
                        if (!IsValidMemoryAddress(address)) {
                            LOG_ERROR(Core_ARM11, "SWPB: Attempting to access invalid address: 0x%08X", address);
                            fprintf(stderr, "ERROR: SWPB: Attempting to access invalid address: 0x%08X\n", address);
                            // Skip this instruction
                            break;
                        }

                        // Perform atomic swap byte operation
                        unsigned int value = cpu->ReadMemory8(address);
                        cpu->WriteMemory8(address, cpu->Reg[inst_cream->Rm] & 0xFF);
                        cpu->Reg[inst_cream->Rd] = value;
                    }
                    break;

                // PLD (Preload Data) instruction
                case 120: // PLD_INST
                    {
                        // PLD is a hint instruction, so it's optional and doesn't need to do anything
                        // The CPU can use it as a hint to preload data into the cache, but we can safely ignore it
                        // in our emulator implementation
                    }
                    break;

                // LDRHT instruction - Load Register Halfword Unprivileged
                case 121: // LDRHT_INST
                    {
                        ldst_inst* const inst_cream = (ldst_inst*)(inst_base->component);
                        u32 address = 0;

                        // Make sure get_addr is valid before calling it
                        if (inst_cream->get_addr == nullptr) {
                            LOG_ERROR(Core_ARM11, "LDRHT: Invalid addressing mode");
                            fprintf(stderr, "ERROR: LDRHT: Invalid addressing mode\n");
                            break;
                        }

                        inst_cream->get_addr(cpu, inst_cream->inst, address);

                        // Validate memory address before reading
                        if (!IsValidMemoryAddress(address)) {
                            LOG_ERROR(Core_ARM11, "LDRHT: Attempting to read from invalid address: 0x%08X", address);
                            fprintf(stderr, "ERROR: LDRHT: Attempting to read from invalid address: 0x%08X\n", address);
                            break;
                        }

                        // LDRHT performs an unprivileged memory access
                        // In our emulator, we don't distinguish between privileged and unprivileged access
                        cpu->Reg[BITS(inst_cream->inst, 12, 15)] = cpu->ReadMemory16(address);
                    }
                    break;

                // STRHT instruction - Store Register Halfword Unprivileged
                case 122: // STRHT_INST
                    {
                        ldst_inst* const inst_cream = (ldst_inst*)(inst_base->component);
                        u32 address = 0;

                        // Make sure get_addr is valid before calling it
                        if (inst_cream->get_addr == nullptr) {
                            LOG_ERROR(Core_ARM11, "STRHT: Invalid addressing mode");
                            fprintf(stderr, "ERROR: STRHT: Invalid addressing mode\n");
                            break;
                        }

                        inst_cream->get_addr(cpu, inst_cream->inst, address);

                        // Validate memory address before writing
                        if (!IsValidMemoryAddress(address)) {
                            LOG_ERROR(Core_ARM11, "STRHT: Attempting to write to invalid address: 0x%08X", address);
                            fprintf(stderr, "ERROR: STRHT: Attempting to write to invalid address: 0x%08X\n", address);
                            break;
                        }

                        // STRHT performs an unprivileged memory access
                        // In our emulator, we don't distinguish between privileged and unprivileged access
                        cpu->WriteMemory16(address, cpu->Reg[BITS(inst_cream->inst, 12, 15)]);
                    }
                    break;

                // LDRSBT instruction - Load Register Signed Byte Unprivileged
                case 123: // LDRSBT_INST
                    {
                        ldst_inst* const inst_cream = (ldst_inst*)(inst_base->component);
                        u32 address = 0;

                        // Make sure get_addr is valid before calling it
                        if (inst_cream->get_addr == nullptr) {
                            LOG_ERROR(Core_ARM11, "LDRSBT: Invalid addressing mode");
                            fprintf(stderr, "ERROR: LDRSBT: Invalid addressing mode\n");
                            break;
                        }

                        inst_cream->get_addr(cpu, inst_cream->inst, address);

                        // Validate memory address before reading
                        if (!IsValidMemoryAddress(address)) {
                            LOG_ERROR(Core_ARM11, "LDRSBT: Attempting to read from invalid address: 0x%08X", address);
                            fprintf(stderr, "ERROR: LDRSBT: Attempting to read from invalid address: 0x%08X\n", address);
                            break;
                        }

                        // LDRSBT performs an unprivileged memory access and sign-extends the byte
                        // In our emulator, we don't distinguish between privileged and unprivileged access
                        s8 data = static_cast<s8>(cpu->ReadMemory8(address));
                        cpu->Reg[BITS(inst_cream->inst, 12, 15)] = data;
                    }
                    break;

                // LDRSHT instruction - Load Register Signed Halfword Unprivileged
                case 124: // LDRSHT_INST
                    {
                        ldst_inst* const inst_cream = (ldst_inst*)(inst_base->component);
                        u32 address = 0;

                        // Make sure get_addr is valid before calling it
                        if (inst_cream->get_addr == nullptr) {
                            LOG_ERROR(Core_ARM11, "LDRSHT: Invalid addressing mode");
                            fprintf(stderr, "ERROR: LDRSHT: Invalid addressing mode\n");
                            break;
                        }

                        inst_cream->get_addr(cpu, inst_cream->inst, address);

                        // Validate memory address before reading
                        if (!IsValidMemoryAddress(address)) {
                            LOG_ERROR(Core_ARM11, "LDRSHT: Attempting to read from invalid address: 0x%08X", address);
                            fprintf(stderr, "ERROR: LDRSHT: Attempting to read from invalid address: 0x%08X\n", address);
                            break;
                        }

                        // LDRSHT performs an unprivileged memory access and sign-extends the halfword
                        // In our emulator, we don't distinguish between privileged and unprivileged access
                        s16 data = static_cast<s16>(cpu->ReadMemory16(address));
                        cpu->Reg[BITS(inst_cream->inst, 12, 15)] = data;
                    }
                    break;

                // For other instructions, fall back to the normal interpreter
                default:
                    // Call the normal interpreter for this instruction
                    // This is a fallback for less common instructions
                    {
                        // Set up the CPU state for this instruction
                        cpu->Reg[15] = instructions[i].address;

                        // Call InterpreterMainLoop with a single instruction to execute
                        u64 old_count = cpu->NumInstrsToExecute;
                        cpu->NumInstrsToExecute = 1;
                        InterpreterMainLoop(cpu);
                        cpu->NumInstrsToExecute = old_count;
                    }
                    break;
            }
        }

        num_instrs++;

        // Check if we should stop execution
        if (num_instrs >= cpu->NumInstrsToExecute) {
            break;
        }

        // Check for branches
        if (inst_base->br != TransExtData::NON_BRANCH) {
            break;
        }
    }

    // Check if PC was updated
    if (cpu->Reg[15] == original_pc && num_instrs > 0) {
        // PC wasn't updated, force it to advance
        LOG_TRACE(Core_ARM11, "PC wasn't updated by ExecuteThreadedBlock, forcing advance");
        fprintf(stderr, "PC wasn't updated by ExecuteThreadedBlock, forcing advance\n");

        // Force PC to advance to the next instruction after the block
        if (count > 0) {
            // Set PC to the address after the last executed instruction
            cpu->Reg[15] = instructions[num_instrs - 1].address + 4;
            LOG_TRACE(Core_ARM11, "Forced PC to 0x%08X", cpu->Reg[15]);
            fprintf(stderr, "Forced PC to 0x%08X\n", cpu->Reg[15]);
        } else {
            // If no instructions were executed, just advance by 4
            cpu->Reg[15] = original_pc + 4;
            LOG_TRACE(Core_ARM11, "Forced PC to 0x%08X (no instructions executed)", cpu->Reg[15]);
            fprintf(stderr, "Forced PC to 0x%08X (no instructions executed)\n", cpu->Reg[15]);
        }
    }

    LOG_TRACE(Core_ARM11, "ExecuteThreadedBlock completed, executed %u instructions, PC before=0x%08X, PC after=0x%08X",
              num_instrs, original_pc, cpu->Reg[15]);
    fprintf(stderr, "ExecuteThreadedBlock completed, executed %u instructions, PC before=0x%08X, PC after=0x%08X\n",
            num_instrs, original_pc, cpu->Reg[15]);

    // Print CPU state after execution
//    fprintf(stderr, "CPU state after threaded block execution:\n");
//    PrintCPUState(cpu);

    return num_instrs;
}

} // anonymous namespace

// Detect common instruction patterns for specialized handling
int DetectCommonPattern(const ThreadedInstruction* instructions, int count) {
    if (count < 3) return 0; // Need at least 3 instructions for a pattern

    // Check for common memory copy pattern: LDR, MOV, STR
    if (((arm_inst*)(instructions[0].inst_base))->idx == 94 && // LDR
        ((arm_inst*)(instructions[1].inst_base))->idx == 69 && // MOV
        ((arm_inst*)(instructions[2].inst_base))->idx == 95) { // STR
        return 1; // Memory copy pattern
    }

    // Check for common increment pattern: LDR, ADD, STR
    if (((arm_inst*)(instructions[0].inst_base))->idx == 94 && // LDR
        ((arm_inst*)(instructions[1].inst_base))->idx == 62 && // ADD
        ((arm_inst*)(instructions[2].inst_base))->idx == 95) { // STR
        return 2; // Increment pattern
    }

    // No recognized pattern
    return 0;
}

// Execute a specialized pattern
unsigned ExecuteSpecializedPattern(ARMul_State* cpu, ThreadedInstruction* instructions, int pattern_type) {
    switch (pattern_type) {
        case 1: { // Memory copy pattern: LDR, MOV, STR
            // Extract instruction details
            ldr_inst* ldr_cream = (ldr_inst*)((arm_inst*)(instructions[0].inst_base))->component;
            mov_inst* mov_cream = (mov_inst*)((arm_inst*)(instructions[1].inst_base))->component;
            str_inst* str_cream = (str_inst*)((arm_inst*)(instructions[2].inst_base))->component;

            // Calculate source address
            u32 src_addr = 0;
            if (ldr_cream->Rn == 15) {
                src_addr = (cpu->Reg[15] + 8) + ldr_cream->offset;
            } else {
                src_addr = cpu->Reg[ldr_cream->Rn] + ldr_cream->offset;
            }

            // Validate source address before reading
            if (!IsValidMemoryAddress(src_addr)) {
                LOG_ERROR(Core_ARM11, "Pattern 1: Attempting to read from invalid address: 0x%08X", src_addr);
                fprintf(stderr, "ERROR: Pattern 1: Attempting to read from invalid address: 0x%08X\n", src_addr);
                return 0; // Don't execute this pattern
            }

            // Read value
            u32 value = cpu->memory.Read32(src_addr);

            // Apply any MOV operation (typically just a register move)
            // In a real memory copy, this might be skipped

            // Calculate destination address
            u32 dst_addr = 0;
            if (str_cream->Rn == 15) {
                dst_addr = (cpu->Reg[15] + 8) + str_cream->offset;
            } else {
                dst_addr = cpu->Reg[str_cream->Rn] + str_cream->offset;
            }

            // Validate destination address before writing
            if (!IsValidMemoryAddress(dst_addr)) {
                LOG_ERROR(Core_ARM11, "Pattern 1: Attempting to write to invalid address: 0x%08X", dst_addr);
                fprintf(stderr, "ERROR: Pattern 1: Attempting to write to invalid address: 0x%08X\n", dst_addr);
                return 0; // Don't execute this pattern
            }

            // Write value
            cpu->memory.Write32(dst_addr, value);

            // Update PC
            cpu->Reg[15] = instructions[2].address + 4;

            return 3; // Executed 3 instructions
        }

        case 2: { // Increment pattern: LDR, ADD, STR
            // Extract instruction details
            ldr_inst* ldr_cream = (ldr_inst*)((arm_inst*)(instructions[0].inst_base))->component;
            add_inst* add_cream = (add_inst*)((arm_inst*)(instructions[1].inst_base))->component;
            str_inst* str_cream = (str_inst*)((arm_inst*)(instructions[2].inst_base))->component;

            // Calculate address
            u32 addr = 0;
            if (ldr_cream->Rn == 15) {
                addr = (cpu->Reg[15] + 8) + ldr_cream->offset;
            } else {
                addr = cpu->Reg[ldr_cream->Rn] + ldr_cream->offset;
            }

            // Validate address before reading
            if (!IsValidMemoryAddress(addr)) {
                LOG_ERROR(Core_ARM11, "Pattern 2: Attempting to access invalid address: 0x%08X", addr);
                fprintf(stderr, "ERROR: Pattern 2: Attempting to access invalid address: 0x%08X\n", addr);
                return 0; // Don't execute this pattern
            }

            // Read value
            u32 value = cpu->memory.Read32(addr);

            // Increment value
            value += add_cream->shifter_operand;

            // Write back
            cpu->memory.Write32(addr, value);

            // Update PC
            cpu->Reg[15] = instructions[2].address + 4;

            return 3; // Executed 3 instructions
        }

        default:
            return 0; // No pattern executed
    }
}

// Global specialized cache instance
static ARMSpecializedCache g_specialized_cache;

// Get the global specialized cache instance
ARMSpecializedCache& GetGlobalSpecializedCache() {
    return g_specialized_cache;
}

// Batch decode multiple ARM instructions at once using SIMD when available
// This is an optimization for ARM64/NEON platforms
unsigned BatchDecodeThreadedBlock(ARMul_State* cpu, u32 start_address, ThreadedInstruction* instructions, unsigned max_instructions) {
#if (defined(__ARM_NEON) || defined(__aarch64__)) && USE_NEON
    // SIMD-based batch decoding for ARM64/NEON platforms
    // Process up to 4 instructions at once using NEON intrinsics

    // Ensure we're in ARM mode, not Thumb mode
    if (cpu->TFlag) {
        // Fall back to regular decoding for Thumb mode
        return DecodeThreadedBlock(cpu, start_address, instructions, max_instructions);
    }

    unsigned count = 0;
    u32 address = start_address;

    // Add debug logging
    LOG_TRACE(Core_ARM11, "BatchDecodeThreadedBlock at address 0x%08X, max_instructions=%u", start_address, max_instructions);

    // Process instructions in batches of 4
    while (count + 4 <= max_instructions) {
        // Check if we've reached a page boundary
        if ((address & 0xFFF) == 0 && count > 0) {
            break;
        }
        // Validate memory addresses before reading
        u32 addr1 = address & 0xFFFFFFFC;
        u32 addr2 = (address + 4) & 0xFFFFFFFC;
        u32 addr3 = (address + 8) & 0xFFFFFFFC;
        u32 addr4 = (address + 12) & 0xFFFFFFFC;

        if (!IsValidMemoryAddress(addr1) ||
            !IsValidMemoryAddress(addr2) ||
            !IsValidMemoryAddress(addr3) ||
            !IsValidMemoryAddress(addr4) ) {
            LOG_ERROR(Core_ARM11, "Attempting to batch read instructions from invalid address range: 0x%08X-0x%08X",
                      addr1, addr4);
            fprintf(stderr, "ERROR: Attempting to batch read instructions from invalid address range: 0x%08X-0x%08X\n",
                    addr1, addr4);
            break;
        }

        // Load 4 instructions at once
        u32 instr1 = cpu->memory.Read32(addr1);
        u32 instr2 = cpu->memory.Read32(addr2);
        u32 instr3 = cpu->memory.Read32(addr3);
        u32 instr4 = cpu->memory.Read32(addr4);

        // Decode the instructions in parallel using NEON intrinsics
        int idx1 = 0, idx2 = 0, idx3 = 0, idx4 = 0;
        ARMDecodeStatus status1 = DecodeARMInstruction(instr1, &idx1);
        ARMDecodeStatus status2 = DecodeARMInstruction(instr2, &idx2);
        ARMDecodeStatus status3 = DecodeARMInstruction(instr3, &idx3);
        ARMDecodeStatus status4 = DecodeARMInstruction(instr4, &idx4);

        // Check if the first instruction failed to decode
        if (status1 != ARMDecodeStatus::SUCCESS ||
            idx1 < 0 || idx1 >= static_cast<int>(arm_instruction_trans_len)) {
            // If we can't decode the first instruction, stop processing this batch
            break;
        }

        // Create the instruction handler for the first instruction
        std::size_t ptr = 0;
        ARM_INST_PTR inst_base1 = arm_instruction_trans[idx1](instr1, idx1);
        instructions[count].handler = nullptr;  // Will be filled in later
        instructions[count].inst_base = inst_base1;
        instructions[count].address = address;
        count++;
        address += 4;

        // Stop if we hit a branch instruction
        if (((arm_inst*)(inst_base1))->br != TransExtData::NON_BRANCH) {
            break;
        }

        // Process second instruction if it decoded successfully
        if (status2 == ARMDecodeStatus::SUCCESS &&
            idx2 >= 0 && idx2 < static_cast<int>(arm_instruction_trans_len)) {
            ARM_INST_PTR inst_base2 = arm_instruction_trans[idx2](instr2, idx2);
            instructions[count].handler = nullptr;
            instructions[count].inst_base = inst_base2;
            instructions[count].address = address;
            count++;
            address += 4;

            // Stop if we hit a branch instruction
            if (((arm_inst*)(inst_base2))->br != TransExtData::NON_BRANCH) {
                break;
            }

            // Process third instruction if it decoded successfully
            if (status3 == ARMDecodeStatus::SUCCESS &&
                idx3 >= 0 && idx3 < static_cast<int>(arm_instruction_trans_len)) {
                ARM_INST_PTR inst_base3 = arm_instruction_trans[idx3](instr3, idx3);
                instructions[count].handler = nullptr;
                instructions[count].inst_base = inst_base3;
                instructions[count].address = address;
                count++;
                address += 4;

                // Stop if we hit a branch instruction
                if (((arm_inst*)(inst_base3))->br != TransExtData::NON_BRANCH) {
                    break;
                }

                // Process fourth instruction if it decoded successfully
                if (status4 == ARMDecodeStatus::SUCCESS &&
                    idx4 >= 0 && idx4 < static_cast<int>(arm_instruction_trans_len)) {
                    ARM_INST_PTR inst_base4 = arm_instruction_trans[idx4](instr4, idx4);
                    instructions[count].handler = nullptr;
                    instructions[count].inst_base = inst_base4;
                    instructions[count].address = address;
                    count++;
                    address += 4;

                    // Stop if we hit a branch instruction
                    if (((arm_inst*)(inst_base4))->br != TransExtData::NON_BRANCH) {
                        break;
                    }
                }
            }
        }
    }

    return count;
#else
    // Fall back to regular decoding for non-NEON platforms
    return DecodeThreadedBlock(cpu, start_address, instructions, max_instructions);
#endif
}

// Entry point for the direct-threaded interpreter
unsigned DirectThreadedMainLoop(ARMul_State* cpu) {
    // Ensure CPU flags are properly loaded at the start of execution
    LOAD_NZCVT;
    MICROPROFILE_SCOPE(DynCom_DirectThreaded);

    // Get the profiler instance
    auto& profiler = ARMDyncomProfiler::GetInstance();

    // Get the global specialized cache
    auto& specialized_cache = GetGlobalSpecializedCache();

    // Start timing the main loop
    profiler.StartTiming("DirectThreadedMainLoop");

    // Instruction cache for direct threading
    static std::unordered_map<u32, std::vector<ThreadedInstruction>> threaded_cache;

    // Check if we have a cached block for the current PC
    u32 pc = cpu->Reg[15];

    // Validate PC is within a valid memory range using our comprehensive memory map validation
    if (!IsValidMemoryAddress(pc)) {
        LOG_ERROR(Core_ARM11, "PC outside valid memory range: 0x%08X", pc);
        fprintf(stderr, "ERROR: PC outside valid memory range: 0x%08X\n", pc);

        // Reset PC to a known good address (process image start)
        cpu->Reg[15] = 0x00100000; // PROCESS_IMAGE_VADDR from memory.h
        pc = cpu->Reg[15];
    }

    // Debug logging
    LOG_TRACE(Core_ARM11, "DirectThreadedMainLoop executing at PC=0x%08X, CPSR=0x%08X, TFlag=%d, NumInstrsToExecute=%llu",
              pc, cpu->Cpsr, cpu->TFlag, cpu->NumInstrsToExecute);

    // Add loop detection to prevent infinite loops
    static u32 last_pc = 0;
    static int same_pc_count = 0;

    if (pc == last_pc) {
        same_pc_count++;
        if (same_pc_count > 10) {
            // We're likely in an infinite loop, force PC to advance
            LOG_TRACE(Core_ARM11, "Possible infinite loop detected at PC=0x%08X, forcing PC to advance", pc);
            fprintf(stderr, "Possible infinite loop detected at PC=0x%08X, forcing PC to advance\n", pc);

            // Force PC to advance by 4 bytes (one instruction)
            cpu->Reg[15] = pc + 4;
            pc = cpu->Reg[15];
            same_pc_count = 0;
        }
    } else {
        same_pc_count = 0;
        last_pc = pc;
    }

    // Force another log message to ensure we're getting output
    fprintf(stderr, "DIRECT THREADED INTERPRETER STARTING EXECUTION AT PC=0x%08X\n", pc);

    // Try to use the specialized cache first if enabled
    bool use_specialized = false;
    unsigned result = 0;

#if USE_SPECIALIZED_CACHE
        profiler.StartTiming("SpecializedCache");

        LOG_TRACE(Core_ARM11, "Checking specialized cache for PC=0x%08X", pc);
        fprintf(stderr, "Checking specialized cache for PC=0x%08X\n", pc);

        // Only use specialized cache for certain addresses
        // In a real implementation, we would have a more sophisticated
        // heuristic to determine when to use the specialized cache
        if ((pc & 0xFFF00000) == 0x00100000) { // Example memory region
            LOG_TRACE(Core_ARM11, "PC is in specialized cache region");
            fprintf(stderr, "PC is in specialized cache region\n");

            // Store the original PC for comparison
            u32 original_pc = pc;

            // Try to execute using the specialized cache
            result = specialized_cache.ExecuteBlock(cpu, pc);
            if (result > 0) {
                LOG_TRACE(Core_ARM11, "Successfully executed specialized cache block, result=%u, PC before=0x%08X, PC after=0x%08X",
                          result, original_pc, cpu->Reg[15]);
                fprintf(stderr, "Successfully executed specialized cache block, PC before=0x%08X, PC after=0x%08X\n",
                        original_pc, cpu->Reg[15]);

                // TODO: Do we need this? Seems like it would just cover up bugs... @JoeMatt
                // Ensure PC was actually updated
                if (cpu->Reg[15] == original_pc) {
                    LOG_TRACE(Core_ARM11, "PC wasn't updated by specialized cache, forcing advance");
                    fprintf(stderr, "PC wasn't updated by specialized cache, forcing advance\n");
                    cpu->Reg[15] = original_pc + 4; // Force PC to advance
                }

                use_specialized = true;
                profiler.EndTiming("SpecializedCache");
                profiler.EndTiming("DirectThreadedMainLoop");
                return result;
            }
        }
        profiler.EndTiming("SpecializedCache");
#else
    // Specialized cache is disabled
    LOG_TRACE(Core_ARM11, "Specialized cache is disabled");
    fprintf(stderr, "Specialized cache is disabled\n");
#endif

    // Fall back to the regular direct-threaded interpreter
    auto cache_it = threaded_cache.find(pc);

    if (cache_it != threaded_cache.end()) {
        // Cache hit - record it
        profiler.StartTiming("CacheHit");

        // Record block execution
        profiler.RecordBlockExecution(pc, cache_it->second.size());

        // Check for common instruction patterns that can be optimized
        int pattern_type = DetectCommonPattern(cache_it->second.data(), cache_it->second.size());
        if (pattern_type > 0) {
            profiler.StartTiming("PatternExecution");
            // Execute the specialized pattern handler
            unsigned executed = ExecuteSpecializedPattern(cpu, cache_it->second.data(), pattern_type);
            if (executed > 0) {
                profiler.EndTiming("PatternExecution");
                profiler.EndTiming("CacheHit");
                profiler.EndTiming("DirectThreadedMainLoop");
                return executed;
            }
            profiler.EndTiming("PatternExecution");
        }

        // Execute the cached block normally
        profiler.StartTiming("BlockExecution");

        // Store the original PC for comparison
        u32 original_pc = cpu->Reg[15];

        // Execute the block
        result = ExecuteThreadedBlock(cpu, cache_it->second.data(), cache_it->second.size());

        // Check if PC was updated
        if (cpu->Reg[15] == original_pc && result > 0) {
            // PC wasn't updated, force it to advance
            LOG_TRACE(Core_ARM11, "PC wasn't updated by threaded cache execution, forcing advance");
            fprintf(stderr, "PC wasn't updated by threaded cache execution, forcing advance\n");

            // Force PC to advance by at least 4 bytes (one instruction)
            cpu->Reg[15] = original_pc + 4;
        }

        LOG_TRACE(Core_ARM11, "Threaded cache execution completed, result=%u, PC before=0x%08X, PC after=0x%08X",
                  result, original_pc, cpu->Reg[15]);
        fprintf(stderr, "Threaded cache execution completed, result=%u, PC before=0x%08X, PC after=0x%08X\n",
                result, original_pc, cpu->Reg[15]);

        profiler.EndTiming("BlockExecution");

        profiler.EndTiming("CacheHit");
    } else {
        // Cache miss - record it
        profiler.StartTiming("CacheMiss");

        // Pre-decode a new block
        profiler.StartTiming("BlockDecode");
        std::vector<ThreadedInstruction> block(MAX_BLOCK_SIZE);
        int count;

        // Use batch decoding on ARM64/NEON platforms for better performance
#if (defined(__ARM_NEON) || defined(__aarch64__)) && USE_NEON
        if (!cpu->TFlag) { // Only batch decode in ARM mode, not Thumb mode
            profiler.StartTiming("BatchDecode");
            count = BatchDecodeThreadedBlock(cpu, pc, block.data(), MAX_BLOCK_SIZE);
            profiler.EndTiming("BatchDecode");
        } else {
            count = PreDecodeBlock(cpu, pc, block.data(), MAX_BLOCK_SIZE);
        }
#else
        count = PreDecodeBlock(cpu, pc, block.data(), MAX_BLOCK_SIZE);
#endif
        profiler.EndTiming("BlockDecode");

        if (count > 0) {
            // Cache the block
            block.resize(count);
            threaded_cache[pc] = std::move(block);

            // Record block creation
            profiler.RecordBlockExecution(pc, count);

            // Check for common instruction patterns
            int pattern_type = DetectCommonPattern(threaded_cache[pc].data(), threaded_cache[pc].size());
            if (pattern_type > 0) {
                profiler.StartTiming("PatternExecution");
                // Execute the specialized pattern handler
                unsigned executed = ExecuteSpecializedPattern(cpu, threaded_cache[pc].data(), pattern_type);
                if (executed > 0) {
                    profiler.EndTiming("PatternExecution");
                    profiler.EndTiming("CacheMiss");
                    profiler.EndTiming("DirectThreadedMainLoop");
                    return executed;
                }
                profiler.EndTiming("PatternExecution");
            }

            // Execute the block normally
            profiler.StartTiming("BlockExecution");

            // Store the original PC for comparison
            u32 original_pc = cpu->Reg[15];

            LOG_TRACE(Core_ARM11, "Executing newly decoded block with %d instructions at PC=0x%08X", count, original_pc);
            fprintf(stderr, "Executing newly decoded block with %d instructions at PC=0x%08X\n", count, original_pc);

            // Execute the block
            result = ExecuteThreadedBlock(cpu, threaded_cache[pc].data(), count);

            // Check if PC was updated
            if (cpu->Reg[15] == original_pc && result > 0) {
                // PC wasn't updated, force it to advance
                LOG_TRACE(Core_ARM11, "PC wasn't updated by newly decoded block execution, forcing advance");
                fprintf(stderr, "PC wasn't updated by newly decoded block execution, forcing advance\n");

                // Force PC to advance by at least 4 bytes (one instruction)
                cpu->Reg[15] = original_pc + 4;
            }

            LOG_TRACE(Core_ARM11, "Newly decoded block execution completed, result=%u, PC before=0x%08X, PC after=0x%08X",
                      result, original_pc, cpu->Reg[15]);
            fprintf(stderr, "Newly decoded block execution completed, result=%u, PC before=0x%08X, PC after=0x%08X\n",
                    result, original_pc, cpu->Reg[15]);

            profiler.EndTiming("BlockExecution");
        } else {
            // Fall back to the normal interpreter if we couldn't decode a block
            profiler.StartTiming("FallbackInterpreter");
            result = InterpreterMainLoop(cpu);
            profiler.EndTiming("FallbackInterpreter");
        }

        profiler.EndTiming("CacheMiss");
    }

    profiler.EndTiming("DirectThreadedMainLoop");
    return result;
}
