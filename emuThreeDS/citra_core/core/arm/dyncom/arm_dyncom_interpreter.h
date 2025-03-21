// Copyright 2014 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "common/common_types.h"

struct ARMul_State;

unsigned InterpreterMainLoop(ARMul_State* state);
bool CondPassed(const ARMul_State* cpu, unsigned int cond);
int InterpreterTranslateSingle(ARMul_State* cpu, std::size_t& bb_start, u32 addr);
