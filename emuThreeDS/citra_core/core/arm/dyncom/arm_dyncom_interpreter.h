// Copyright 2014 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

struct ARMul_State;

// Main interpreter loop
unsigned InterpreterMainLoop(ARMul_State* state);

// Clear the translation cache
void ClearTranslationCache();
