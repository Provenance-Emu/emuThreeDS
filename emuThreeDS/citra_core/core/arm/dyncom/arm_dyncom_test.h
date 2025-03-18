#pragma once

#include "core/arm/dyncom/arm_dyncom.h"

// Test function to compare the performance of different interpreter implementations
void TestInterpreterPerformance(ARM_DynCom* arm_cpu, int num_iterations = 10, u64 num_instructions = 1000000);
