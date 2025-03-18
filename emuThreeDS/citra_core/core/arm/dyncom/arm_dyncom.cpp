// Copyright 2014 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <cstring>
#include <memory>
#include "core/arm/dyncom/arm_dyncom.h"
#include "core/arm/dyncom/arm_dyncom_direct_threaded.h"
#include "core/arm/dyncom/arm_dyncom_interpreter.h"
#include "core/arm/dyncom/arm_dyncom_profiler.h"
#include "core/arm/dyncom/arm_dyncom_trans.h"
#include "core/arm/dyncom/arm_dyncom_specialized_cache.h"
#include "core/arm/skyeye_common/armstate.h"
#include "core/core.h"
#include "core/core_timing.h"

// Configuration option to enable/disable the direct-threaded interpreter
// Enabled by default for better performance
static bool use_direct_threaded_interpreter = false;


// Flag to enable/disable specialized cache
#if USE_SPECIALIZED_CACHE
bool g_use_specialized_cache = true;

// Flag to enable/disable specialized cache for specific memory regions
// This is useful for debugging and troubleshooting
bool g_use_specialized_cache_for_app_heap = false;     // 0x08000000 - 0x10000000
bool g_use_specialized_cache_for_linear_heap = false;  // 0x14000000 - 0x1C000000
bool g_use_specialized_cache_for_vram = false;        // 0x1F000000 - 0x1FF00000
#else
bool g_use_specialized_cache = false;
bool g_use_specialized_cache_for_app_heap = false;     // 0x08000000 - 0x10000000
bool g_use_specialized_cache_for_linear_heap = false;  // 0x14000000 - 0x1C000000
bool g_use_specialized_cache_for_vram = false;
#endif

class DynComThreadContext final : public ARM_Interface::ThreadContext {
public:
    DynComThreadContext() {
        Reset();
    }
    ~DynComThreadContext() override = default;

    void Reset() override {
        cpu_registers = {};
        cpsr = 0;
        fpu_registers = {};
        fpscr = 0;
        fpexc = 0;
    }

    u32 GetCpuRegister(std::size_t index) const override {
        return cpu_registers[index];
    }
    void SetCpuRegister(std::size_t index, u32 value) override {
        cpu_registers[index] = value;
    }
    u32 GetCpsr() const override {
        return cpsr;
    }
    void SetCpsr(u32 value) override {
        cpsr = value;
    }
    u32 GetFpuRegister(std::size_t index) const override {
        return fpu_registers[index];
    }
    void SetFpuRegister(std::size_t index, u32 value) override {
        fpu_registers[index] = value;
    }
    u32 GetFpscr() const override {
        return fpscr;
    }
    void SetFpscr(u32 value) override {
        fpscr = value;
    }
    u32 GetFpexc() const override {
        return fpexc;
    }
    void SetFpexc(u32 value) override {
        fpexc = value;
    }

private:
    friend class ARM_DynCom;

    std::array<u32, 16> cpu_registers;
    u32 cpsr;
    std::array<u32, 64> fpu_registers;
    u32 fpscr;
    u32 fpexc;
};

ARM_DynCom::ARM_DynCom(Core::System* system, Memory::MemorySystem& memory,
                       PrivilegeMode initial_mode, u32 id,
                       std::shared_ptr<Core::Timing::Timer> timer)
    : ARM_Interface(id, timer), system(system) {
    state = std::make_unique<ARMul_State>(system, memory, initial_mode);
}

ARM_DynCom::~ARM_DynCom() {}

void ARM_DynCom::Run() {
    DEBUG_ASSERT(system != nullptr);
    ExecuteInstructions(std::max<s64>(timer->GetDowncount(), 0));
}

void ARM_DynCom::Step() {
    ExecuteInstructions(1);
}

void ARM_DynCom::ClearInstructionCache() {
    state->instruction_cache.clear();
    trans_cache_buf_top = 0;

#if USE_SPECIALIZED_CACHE
    if (g_use_specialized_cache) {
        // Also clear the specialized cache
        GetGlobalSpecializedCache().Clear();
    }
#endif
}

void ARM_DynCom::InvalidateCacheRange(u32, std::size_t) {
    ClearInstructionCache();
}

void ARM_DynCom::SetPageTable(const std::shared_ptr<Memory::PageTable>& page_table) {
    ClearInstructionCache();
}

std::shared_ptr<Memory::PageTable> ARM_DynCom::GetPageTable() const {
    return nullptr;
}

void ARM_DynCom::PurgeState() {}

void ARM_DynCom::SetPC(u32 pc) {
    state->Reg[15] = pc;
}

u32 ARM_DynCom::GetPC() const {
    return state->Reg[15];
}

u32 ARM_DynCom::GetReg(int index) const {
    return state->Reg[index];
}

void ARM_DynCom::SetReg(int index, u32 value) {
    state->Reg[index] = value;
}

u32 ARM_DynCom::GetVFPReg(int index) const {
    return state->ExtReg[index];
}

void ARM_DynCom::SetVFPReg(int index, u32 value) {
    state->ExtReg[index] = value;
}

u32 ARM_DynCom::GetVFPSystemReg(VFPSystemRegister reg) const {
    return state->VFP[reg];
}

void ARM_DynCom::SetVFPSystemReg(VFPSystemRegister reg, u32 value) {
    state->VFP[reg] = value;
}

u32 ARM_DynCom::GetCPSR() const {
    return state->Cpsr;
}

void ARM_DynCom::SetCPSR(u32 cpsr) {
    state->Cpsr = cpsr;
}

u32 ARM_DynCom::GetCP15Register(CP15Register reg) const {
    return state->CP15[reg];
}

void ARM_DynCom::SetCP15Register(CP15Register reg, u32 value) {
    state->CP15[reg] = value;
}

void ARM_DynCom::ExecuteInstructions(u64 num_instructions) {
    state->NumInstrsToExecute = num_instructions;

    // Add debug logging
    LOG_TRACE(Core_ARM11, "ARM_DynCom::ExecuteInstructions: num_instructions=%llu, PC=0x%08X, CPSR=0x%08X, TFlag=%d",
              num_instructions, state->Reg[15], state->Cpsr, state->TFlag);

    unsigned ticks_executed = 0;

#if USE_SPECIALIZED_CACHE
    // Try to use the specialized cache first if it's enabled
    // This works regardless of which interpreter is used
    if (g_use_specialized_cache) {
        u32 pc = state->Reg[15];

        // Check if the current PC is valid and aligned
        if (IsValidMemoryAddress(pc) && (pc & 0x3) == 0) {
            bool safe_for_specialized_cache = true;

            // Check if the PC is in a memory region where we want to use the specialized cache
            if (pc >= 0x08000000 && pc < 0x10000000) { // Application heap
                safe_for_specialized_cache = g_use_specialized_cache_for_app_heap;
                if (!safe_for_specialized_cache) {
                    LOG_TRACE(Core_ARM11, "Skipping specialized cache for app heap at PC=0x%08X", pc);
                }
            } else if (pc >= 0x14000000 && pc < 0x1C000000) { // Linear heap
                safe_for_specialized_cache = g_use_specialized_cache_for_linear_heap;
                if (!safe_for_specialized_cache) {
                    LOG_TRACE(Core_ARM11, "Skipping specialized cache for linear heap at PC=0x%08X", pc);
                }
            } else if (pc >= 0x1F000000 && pc < 0x1FF00000) { // VRAM
                safe_for_specialized_cache = g_use_specialized_cache_for_vram;
                if (!safe_for_specialized_cache) {
                    LOG_TRACE(Core_ARM11, "Skipping specialized cache for VRAM at PC=0x%08X", pc);
                }
            }

            // Always skip specialized cache for these sensitive regions
            if ((pc >= 0x1EC00000 && pc < 0x1F000000) || // IO register area
                (pc >= 0x1FF00000 && pc < 0x1FF80000) || // DSP memory
                (pc >= 0x1FF80000 && pc < 0x1FF83000)) { // Config memory
                safe_for_specialized_cache = false;
                LOG_TRACE(Core_ARM11, "Skipping specialized cache for sensitive memory region at PC=0x%08X", pc);
            }

            if (safe_for_specialized_cache) {
                // Verify we can read the instruction at this address before trying the specialized cache
                try {
                    u32 instr = state->memory.Read32(pc);

                    // Check for suspicious instruction values that might indicate invalid memory
                    if (instr == 0 || instr == 0xFFFFFFFF) {
                        LOG_WARNING(Core_ARM11, "Skipping specialized cache due to suspicious instruction 0x%08X at PC=0x%08X", instr, pc);
                    } else {
                        // Get the global specialized cache
                        auto& specialized_cache = GetGlobalSpecializedCache();

                        // Try to execute a specialized block
                        ticks_executed = specialized_cache.ExecuteBlock(state.get(), pc);

                        // If we successfully executed a specialized block, we're done
                        if (ticks_executed > 0) {
                            LOG_TRACE(Core_ARM11, "Used specialized cache for execution at PC=0x%08X", pc);
                            goto execution_complete;
                        }
                    }
                } catch (...) {
                    LOG_ERROR(Core_ARM11, "Exception reading instruction at PC=0x%08X", pc);
                    // Fall back to regular interpreter
                }
            }
        }
    }
#endif

    // If specialized cache didn't work, use the appropriate interpreter
    if (use_direct_threaded_interpreter) {
        // Use the direct-threaded interpreter for better performance
        LOG_TRACE(Core_ARM11, "Using direct-threaded interpreter");
        ticks_executed = DirectThreadedMainLoop(state.get());
    } else {
        // Fall back to the original interpreter
        LOG_TRACE(Core_ARM11, "Using original interpreter");
        ticks_executed = InterpreterMainLoop(state.get());
    }

#if USE_SPECIALIZED_CACHE
execution_complete:
#endif

    LOG_TRACE(Core_ARM11, "Executed %u ticks", ticks_executed);

    if (system != nullptr) {
        timer->AddTicks(ticks_executed);
    }
    state->ServeBreak();
}

// Function to toggle the direct-threaded interpreter
void ARM_DynCom::SetUseDirectThreadedInterpreter(bool enabled) {
    use_direct_threaded_interpreter = enabled;
}

// Reset the profiler
void ARM_DynCom::ResetProfiler() {
    ARMDyncomProfiler::GetInstance().Reset();
}

// Print profiler statistics
void ARM_DynCom::PrintProfilerStats() {
    ARMDyncomProfiler::GetInstance().PrintStats();
}

// Clear the specialized cache
void ARM_DynCom::ClearSpecializedCache() {
    GetGlobalSpecializedCache().Clear();
}

// Enable or disable the specialized cache
void ARM_DynCom::EnableSpecializedCache(bool enabled) {
    g_use_specialized_cache = enabled;
}

std::unique_ptr<ARM_Interface::ThreadContext> ARM_DynCom::NewContext() const {
    return std::make_unique<DynComThreadContext>();
}

void ARM_DynCom::SaveContext(const std::unique_ptr<ThreadContext>& arg) {
    DynComThreadContext* ctx = dynamic_cast<DynComThreadContext*>(arg.get());
    ASSERT(ctx);

    ctx->cpu_registers = state->Reg;
    ctx->cpsr = state->Cpsr;
    ctx->fpu_registers = state->ExtReg;
    ctx->fpscr = state->VFP[VFP_FPSCR];
    ctx->fpexc = state->VFP[VFP_FPEXC];
}

void ARM_DynCom::LoadContext(const std::unique_ptr<ThreadContext>& arg) {
    DynComThreadContext* ctx = dynamic_cast<DynComThreadContext*>(arg.get());
    ASSERT(ctx);

    state->Reg = ctx->cpu_registers;
    state->Cpsr = ctx->cpsr;
    state->ExtReg = ctx->fpu_registers;
    state->VFP[VFP_FPSCR] = ctx->fpscr;
    state->VFP[VFP_FPEXC] = ctx->fpexc;
}

void ARM_DynCom::PrepareReschedule() {
    state->NumInstrsToExecute = 0;
}
