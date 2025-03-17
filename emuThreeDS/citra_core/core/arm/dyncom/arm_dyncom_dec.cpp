// Copyright 2012 Michael Kang, 2014 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "core/arm/dyncom/arm_dyncom_dec.h"
#include "core/arm/skyeye_common/armsupp.h"
#include "common/logging/log.h"

#include <vector>
#include <array>
#include <algorithm>

// Define the logging section
//ENUM_DEFINE(Log::Class, Core_ARM11);

namespace {
struct InstructionSetEncodingItem {
    const char* name;
    int attribute_value;
    int version;
    u32 content[21];
};

// ARM versions
enum {
    INVALID = 0,
    ARMALL,
    ARMV4,
    ARMV4T,
    ARMV5T,
    ARMV5TE,
    ARMV5TEJ,
    ARMV6,
    ARM1176JZF_S,
    ARMVFP2,
    ARMVFP3,
    ARMV6K,
};
} // namespace

// clang-format off
const InstructionSetEncodingItem arm_instruction[] = {
    { "vmla", 5, ARMVFP2,      { 23, 27, 0x1C, 20, 21, 0x0, 9, 11, 0x5, 6, 6, 0, 4, 4, 0 }},
    { "vmls", 5, ARMVFP2,      { 23, 27, 0x1C, 20, 21, 0x0, 9, 11, 0x5, 6, 6, 1, 4, 4, 0 }},
    { "vnmla", 5, ARMVFP2,     { 23, 27, 0x1C, 20, 21, 0x1, 9, 11, 0x5, 6, 6, 1, 4, 4, 0 }},
    { "vnmls", 5, ARMVFP2,     { 23, 27, 0x1C, 20, 21, 0x1, 9, 11, 0x5, 6, 6, 0, 4, 4, 0 }},
    { "vnmul", 5, ARMVFP2,     { 23, 27, 0x1C, 20, 21, 0x2, 9, 11, 0x5, 6, 6, 1, 4, 4, 0 }},
    { "vmul", 5, ARMVFP2,      { 23, 27, 0x1C, 20, 21, 0x2, 9, 11, 0x5, 6, 6, 0, 4, 4, 0 }},
    { "vadd", 5, ARMVFP2,      { 23, 27, 0x1C, 20, 21, 0x3, 9, 11, 0x5, 6, 6, 0, 4, 4, 0 }},
    { "vsub", 5, ARMVFP2,      { 23, 27, 0x1C, 20, 21, 0x3, 9, 11, 0x5, 6, 6, 1, 4, 4, 0 }},
    { "vdiv", 5, ARMVFP2,      { 23, 27, 0x1D, 20, 21, 0x0, 9, 11, 0x5, 6, 6, 0, 4, 4, 0 }},
    { "vmov(i)", 4, ARMVFP3,   { 23, 27, 0x1D, 20, 21, 0x3, 9, 11, 0x5, 4, 7, 0 }},
    { "vmov(r)", 5, ARMVFP3,   { 23, 27, 0x1D, 16, 21, 0x30, 9, 11, 0x5, 6, 7, 1, 4, 4, 0 }},
    { "vabs", 5, ARMVFP2,      { 23, 27, 0x1D, 16, 21, 0x30, 9, 11, 0x5, 6, 7, 3, 4, 4, 0 }},
    { "vneg", 5, ARMVFP2,      { 23, 27, 0x1D, 17, 21, 0x18, 9, 11, 0x5, 6, 7, 1, 4, 4, 0 }},
    { "vsqrt", 5, ARMVFP2,     { 23, 27, 0x1D, 16, 21, 0x31, 9, 11, 0x5, 6, 7, 3, 4, 4, 0 }},
    { "vcmp", 5, ARMVFP2,      { 23, 27, 0x1D, 16, 21, 0x34, 9, 11, 0x5, 6, 6, 1, 4, 4, 0 }},
    { "vcmp2", 5, ARMVFP2,     { 23, 27, 0x1D, 16, 21, 0x35, 9, 11, 0x5, 0, 6, 0x40 }},
    { "vcvt(bds)", 5, ARMVFP2, { 23, 27, 0x1D, 16, 21, 0x37, 9, 11, 0x5, 6, 7, 3, 4, 4, 0 }},
    { "vcvt(bff)", 6, ARMVFP3, { 23, 27, 0x1D, 19, 21, 0x7, 17, 17, 0x1, 9, 11, 5, 6, 6, 1 }},
    { "vcvt(bfi)", 5, ARMVFP2, { 23, 27, 0x1D, 19, 21, 0x7, 9, 11, 0x5, 6, 6, 1, 4, 4, 0 }},
    { "vmovbrs", 3, ARMVFP2,   { 21, 27, 0x70, 8, 11, 0xA, 0, 6, 0x10 }},
    { "vmsr", 2, ARMVFP2,      { 20, 27, 0xEE, 0, 11, 0xA10 }},
    { "vmovbrc", 4, ARMVFP2,   { 23, 27, 0x1C, 20, 20, 0x0, 8, 11, 0xB, 0, 4, 0x10 }},
    { "vmrs", 2, ARMVFP2,      { 20, 27, 0xEF, 0, 11, 0xA10 }},
    { "vmovbcr", 4, ARMVFP2,   { 24, 27, 0xE, 20, 20, 1, 8, 11, 0xB, 0, 4, 0x10 }},
    { "vmovbrrss", 3, ARMVFP2, { 21, 27, 0x62, 8, 11, 0xA, 4, 4, 1 }},
    { "vmovbrrd", 3, ARMVFP2,  { 21, 27, 0x62, 6, 11, 0x2C, 4, 4, 1 }},
    { "vstr", 3, ARMVFP2,      { 24, 27, 0xD, 20, 21, 0, 9, 11, 5 }},
    { "vpush", 3, ARMVFP2,     { 23, 27, 0x1A, 16, 21, 0x2D, 9, 11, 5 }},
    { "vstm", 3, ARMVFP2,      { 25, 27, 0x6, 20, 20, 0, 9, 11, 5 }},
    { "vpop", 3, ARMVFP2,      { 23, 27, 0x19, 16, 21, 0x3D, 9, 11, 5 }},
    { "vldr", 3, ARMVFP2,      { 24, 27, 0xD, 20, 21, 1, 9, 11, 5 }},
    { "vldm", 3, ARMVFP2,      { 25, 27, 0x6, 20, 20, 1, 9, 11, 5 }},

    { "srs", 4, 6,         { 25, 31, 0x0000007c, 22, 22, 0x00000001, 16, 20, 0x0000000d, 8, 11, 0x00000005 }},
    { "rfe", 4, 6,         { 25, 31, 0x0000007c, 22, 22, 0x00000000, 20, 20, 0x00000001, 8, 11, 0x0000000a }},
    { "bkpt", 2, 3,        { 20, 27, 0x00000012, 4, 7, 0x00000007 }},
    { "blx", 1, 3,         { 25, 31, 0x0000007d }},
    { "cps", 3, 6,         { 20, 31, 0x00000f10, 16, 16, 0x00000000, 5, 5, 0x00000000 }},
    { "pld", 4, 4,         { 26, 31, 0x0000003d, 24, 24, 0x00000001, 20, 22, 0x00000005, 12, 15, 0x0000000f }},
    { "setend", 2, 6,      { 16, 31, 0x0000f101, 4, 7, 0x00000000 }},
    { "clrex", 1, 6,       { 0, 31, 0xf57ff01f }},
    { "rev16", 2, 6,       { 16, 27, 0x000006bf, 4, 11, 0x000000fb }},
    { "usad8", 3, 6,       { 20, 27, 0x00000078, 12, 15, 0x0000000f, 4, 7, 0x00000001 }},
    { "sxtb", 2, 6,        { 16, 27, 0x000006af, 4, 7, 0x00000007 }},
    { "uxtb", 2, 6,        { 16, 27, 0x000006ef, 4, 7, 0x00000007 }},
    { "sxth", 2, 6,        { 16, 27, 0x000006bf, 4, 7, 0x00000007 }},
    { "sxtb16", 2, 6,      { 16, 27, 0x0000068f, 4, 7, 0x00000007 }},
    { "uxth", 2, 6,        { 16, 27, 0x000006ff, 4, 7, 0x00000007 }},
    { "uxtb16", 2, 6,      { 16, 27, 0x000006cf, 4, 7, 0x00000007 }},
    { "cpy", 2, 6,         { 20, 27, 0x0000001a, 4, 11, 0x00000000 }},
    { "uxtab", 2, 6,       { 20, 27, 0x0000006e, 4, 9, 0x00000007 }},
    { "ssub8", 2, 6,       { 20, 27, 0x00000061, 4, 7, 0x0000000f }},
    { "shsub8", 2, 6,      { 20, 27, 0x00000063, 4, 7, 0x0000000f }},
    { "ssubaddx", 2, 6,    { 20, 27, 0x00000061, 4, 7, 0x00000005 }},
    { "strex", 2, 6,       { 20, 27, 0x00000018, 4, 7, 0x00000009 }},
    { "strexb", 2, 7,      { 20, 27, 0x0000001c, 4, 7, 0x00000009 }},
    { "swp", 2, 0,         { 20, 27, 0x00000010, 4, 7, 0x00000009 }},
    { "swpb", 2, 0,        { 20, 27, 0x00000014, 4, 7, 0x00000009 }},
    { "ssub16", 2, 6,      { 20, 27, 0x00000061, 4, 7, 0x00000007 }},
    { "ssat16", 2, 6,      { 20, 27, 0x0000006a, 4, 7, 0x00000003 }},
    { "shsubaddx", 2, 6,   { 20, 27, 0x00000063, 4, 7, 0x00000005 }},
    { "qsubaddx", 2, 6,    { 20, 27, 0x00000062, 4, 7, 0x00000005 }},
    { "shaddsubx", 2, 6,   { 20, 27, 0x00000063, 4, 7, 0x00000003 }},
    { "shadd8", 2, 6,      { 20, 27, 0x00000063, 4, 7, 0x00000009 }},
    { "shadd16", 2, 6,     { 20, 27, 0x00000063, 4, 7, 0x00000001 }},
    { "sel", 2, 6,         { 20, 27, 0x00000068, 4, 7, 0x0000000b }},
    { "saddsubx", 2, 6,    { 20, 27, 0x00000061, 4, 7, 0x00000003 }},
    { "sadd8", 2, 6,       { 20, 27, 0x00000061, 4, 7, 0x00000009 }},
    { "sadd16", 2, 6,      { 20, 27, 0x00000061, 4, 7, 0x00000001 }},
    { "shsub16", 2, 6,     { 20, 27, 0x00000063, 4, 7, 0x00000007 }},
    { "umaal", 2, 6,       { 20, 27, 0x00000004, 4, 7, 0x00000009 }},
    { "uxtab16", 2, 6,     { 20, 27, 0x0000006c, 4, 7, 0x00000007 }},
    { "usubaddx", 2, 6,    { 20, 27, 0x00000065, 4, 7, 0x00000005 }},
    { "usub8", 2, 6,       { 20, 27, 0x00000065, 4, 7, 0x0000000f }},
    { "usub16", 2, 6,      { 20, 27, 0x00000065, 4, 7, 0x00000007 }},
    { "usat16", 2, 6,      { 20, 27, 0x0000006e, 4, 7, 0x00000003 }},
    { "usada8", 2, 6,      { 20, 27, 0x00000078, 4, 7, 0x00000001 }},
    { "uqsubaddx", 2, 6,   { 20, 27, 0x00000066, 4, 7, 0x00000005 }},
    { "uqsub8", 2, 6,      { 20, 27, 0x00000066, 4, 7, 0x0000000f }},
    { "uqsub16", 2, 6,     { 20, 27, 0x00000066, 4, 7, 0x00000007 }},
    { "uqaddsubx", 2, 6,   { 20, 27, 0x00000066, 4, 7, 0x00000003 }},
    { "uqadd8", 2, 6,      { 20, 27, 0x00000066, 4, 7, 0x00000009 }},
    { "uqadd16", 2, 6,     { 20, 27, 0x00000066, 4, 7, 0x00000001 }},
    { "sxtab", 2, 6,       { 20, 27, 0x0000006a, 4, 7, 0x00000007 }},
    { "uhsubaddx", 2, 6,   { 20, 27, 0x00000067, 4, 7, 0x00000005 }},
    { "uhsub8", 2, 6,      { 20, 27, 0x00000067, 4, 7, 0x0000000f }},
    { "uhsub16", 2, 6,     { 20, 27, 0x00000067, 4, 7, 0x00000007 }},
    { "uhaddsubx", 2, 6,   { 20, 27, 0x00000067, 4, 7, 0x00000003 }},
    { "uhadd8", 2, 6,      { 20, 27, 0x00000067, 4, 7, 0x00000009 }},
    { "uhadd16", 2, 6,     { 20, 27, 0x00000067, 4, 7, 0x00000001 }},
    { "uaddsubx", 2, 6,    { 20, 27, 0x00000065, 4, 7, 0x00000003 }},
    { "uadd8", 2, 6,       { 20, 27, 0x00000065, 4, 7, 0x00000009 }},
    { "uadd16", 2, 6,      { 20, 27, 0x00000065, 4, 7, 0x00000001 }},
    { "sxtah", 2, 6,       { 20, 27, 0x0000006b, 4, 7, 0x00000007 }},
    { "sxtab16", 2, 6,     { 20, 27, 0x00000068, 4, 7, 0x00000007 }},
    { "qadd8", 2, 6,       { 20, 27, 0x00000062, 4, 7, 0x00000009 }},
    { "bxj", 2, 5,         { 20, 27, 0x00000012, 4, 7, 0x00000002 }},
    { "clz", 2, 3,         { 20, 27, 0x00000016, 4, 7, 0x00000001 }},
    { "uxtah", 2, 6,       { 20, 27, 0x0000006f, 4, 7, 0x00000007 }},
    { "bx", 2, 2,          { 20, 27, 0x00000012, 4, 7, 0x00000001 }},
    { "rev", 2, 6,         { 20, 27, 0x0000006b, 4, 7, 0x00000003 }},
    { "blx", 2, 3,         { 20, 27, 0x00000012, 4, 7, 0x00000003 }},
    { "revsh", 2, 6,       { 20, 27, 0x0000006f, 4, 7, 0x0000000b }},
    { "qadd", 2, 4,        { 20, 27, 0x00000010, 4, 7, 0x00000005 }},
    { "qadd16", 2, 6,      { 20, 27, 0x00000062, 4, 7, 0x00000001 }},
    { "qaddsubx", 2, 6,    { 20, 27, 0x00000062, 4, 7, 0x00000003 }},
    { "ldrex", 2, 0,       { 20, 27, 0x00000019, 4, 7, 0x00000009 }},
    { "qdadd", 2, 4,       { 20, 27, 0x00000014, 4, 7, 0x00000005 }},
    { "qdsub", 2, 4,       { 20, 27, 0x00000016, 4, 7, 0x00000005 }},
    { "qsub", 2, 4,        { 20, 27, 0x00000012, 4, 7, 0x00000005 }},
    { "ldrexb", 2, 7,      { 20, 27, 0x0000001d, 4, 7, 0x00000009 }},
    { "qsub8", 2, 6,       { 20, 27, 0x00000062, 4, 7, 0x0000000f }},
    { "qsub16", 2, 6,      { 20, 27, 0x00000062, 4, 7, 0x00000007 }},
    { "smuad", 4, 6,       { 20, 27, 0x00000070, 12, 15, 0x0000000f, 6, 7, 0x00000000, 4, 4, 0x00000001 }},
    { "smmul", 4, 6,       { 20, 27, 0x00000075, 12, 15, 0x0000000f, 6, 7, 0x00000000, 4, 4, 0x00000001 }},
    { "smusd", 4, 6,       { 20, 27, 0x00000070, 12, 15, 0x0000000f, 6, 7, 0x00000001, 4, 4, 0x00000001 }},
    { "smlsd", 3, 6,       { 20, 27, 0x00000070, 6, 7, 0x00000001, 4, 4, 0x00000001 }},
    { "smlsld", 3, 6,      { 20, 27, 0x00000074, 6, 7, 0x00000001, 4, 4, 0x00000001 }},
    { "smmla", 3, 6,       { 20, 27, 0x00000075, 6, 7, 0x00000000, 4, 4, 0x00000001 }},
    { "smmls", 3, 6,       { 20, 27, 0x00000075, 6, 7, 0x00000003, 4, 4, 0x00000001 }},
    { "smlald", 3, 6,      { 20, 27, 0x00000074, 6, 7, 0x00000000, 4, 4, 0x00000001 }},
    { "smlad", 3, 6,       { 20, 27, 0x00000070, 6, 7, 0x00000000, 4, 4, 0x00000001 }},
    { "smlaw", 3, 4,       { 20, 27, 0x00000012, 7, 7, 0x00000001, 4, 5, 0x00000000 }},
    { "smulw", 3, 4,       { 20, 27, 0x00000012, 7, 7, 0x00000001, 4, 5, 0x00000002 }},
    { "pkhtb", 2, 6,       { 20, 27, 0x00000068, 4, 6, 0x00000005 }},
    { "pkhbt", 2, 6,       { 20, 27, 0x00000068, 4, 6, 0x00000001 }},
    { "smul", 3, 4,        { 20, 27, 0x00000016, 7, 7, 0x00000001, 4, 4, 0x00000000 }},
    { "smlalxy", 3, 4,     { 20, 27, 0x00000014, 7, 7, 0x00000001, 4, 4, 0x00000000 }},
    { "smla", 3, 4,        { 20, 27, 0x00000010, 7, 7, 0x00000001, 4, 4, 0x00000000 }},
    { "mcrr", 1, 6,        { 20, 27, 0x000000c4 }},
    { "mrrc", 1, 6,        { 20, 27, 0x000000c5 }},
    { "cmp", 2, 0,         { 26, 27, 0x00000000, 20, 24, 0x00000015 }},
    { "tst", 2, 0,         { 26, 27, 0x00000000, 20, 24, 0x00000011 }},
    { "teq", 2, 0,         { 26, 27, 0x00000000, 20, 24, 0x00000013 }},
    { "cmn", 2, 0,         { 26, 27, 0x00000000, 20, 24, 0x00000017 }},
    { "smull", 2, 0,       { 21, 27, 0x00000006, 4, 7, 0x00000009 }},
    { "umull", 2, 0,       { 21, 27, 0x00000004, 4, 7, 0x00000009 }},
    { "umlal", 2, 0,       { 21, 27, 0x00000005, 4, 7, 0x00000009 }},
    { "smlal", 2, 0,       { 21, 27, 0x00000007, 4, 7, 0x00000009 }},
    { "mul", 2, 0,         { 21, 27, 0x00000000, 4, 7, 0x00000009 }},
    { "mla", 2, 0,         { 21, 27, 0x00000001, 4, 7, 0x00000009 }},
    { "ssat", 2, 6,        { 21, 27, 0x00000035, 4, 5, 0x00000001 }},
    { "usat", 2, 6,        { 21, 27, 0x00000037, 4, 5, 0x00000001 }},
    { "mrs", 4, 0,         { 23, 27, 0x00000002, 20, 21, 0x00000000, 16, 19, 0x0000000f, 0, 11, 0x00000000 }},
    { "msr", 3, 0,         { 23, 27, 0x00000002, 20, 21, 0x00000002, 4, 7, 0x00000000 }},
    { "and", 2, 0,         { 26, 27, 0x00000000, 21, 24, 0x00000000 }},
    { "bic", 2, 0,         { 26, 27, 0x00000000, 21, 24, 0x0000000e }},
    { "ldm", 3, 0,         { 25, 27, 0x00000004, 20, 22, 0x00000005, 15, 15, 0x00000000 }},
    { "eor", 2, 0,         { 26, 27, 0x00000000, 21, 24, 0x00000001 }},
    { "add", 2, 0,         { 26, 27, 0x00000000, 21, 24, 0x00000004 }},
    { "rsb", 2, 0,         { 26, 27, 0x00000000, 21, 24, 0x00000003 }},
    { "rsc", 2, 0,         { 26, 27, 0x00000000, 21, 24, 0x00000007 }},
    { "sbc", 2, 0,         { 26, 27, 0x00000000, 21, 24, 0x00000006 }},
    { "adc", 2, 0,         { 26, 27, 0x00000000, 21, 24, 0x00000005 }},
    { "sub", 2, 0,         { 26, 27, 0x00000000, 21, 24, 0x00000002 }},
    { "orr", 2, 0,         { 26, 27, 0x00000000, 21, 24, 0x0000000c }},
    { "mvn", 2, 0,         { 26, 27, 0x00000000, 21, 24, 0x0000000f }},
    { "mov", 2, 0,         { 26, 27, 0x00000000, 21, 24, 0x0000000d }},
    { "stm", 2, 0,         { 25, 27, 0x00000004, 20, 22, 0x00000004 }},
    { "ldm", 4, 0,         { 25, 27, 0x00000004, 22, 22, 0x00000001, 20, 20, 0x00000001, 15, 15, 0x00000001 }},
    { "ldrsh", 3, 2,       { 25, 27, 0x00000000, 20, 20, 0x00000001, 4, 7, 0x0000000f }},
    { "stm", 3, 0,         { 25, 27, 0x00000004, 22, 22, 0x00000000, 20, 20, 0x00000000 }},
    { "ldm", 3, 0,         { 25, 27, 0x00000004, 22, 22, 0x00000000, 20, 20, 0x00000001 }},
    { "ldrsb", 3, 2,       { 25, 27, 0x00000000, 20, 20, 0x00000001, 4, 7, 0x0000000d }},
    { "strd", 3, 4,        { 25, 27, 0x00000000, 20, 20, 0x00000000, 4, 7, 0x0000000f }},
    { "ldrh", 3, 0,        { 25, 27, 0x00000000, 20, 20, 0x00000001, 4, 7, 0x0000000b }},
    { "strh", 3, 0,        { 25, 27, 0x00000000, 20, 20, 0x00000000, 4, 7, 0x0000000b }},
    { "ldrd", 3, 4,        { 25, 27, 0x00000000, 20, 20, 0x00000000, 4, 7, 0x0000000d }},
    { "strt", 3, 0,        { 26, 27, 0x00000001, 24, 24, 0x00000000, 20, 22, 0x00000002 }},
    { "strbt", 3, 0,       { 26, 27, 0x00000001, 24, 24, 0x00000000, 20, 22, 0x00000006 }},
    { "ldrbt", 3, 0,       { 26, 27, 0x00000001, 24, 24, 0x00000000, 20, 22, 0x00000007 }},
    { "ldrt", 3, 0,        { 26, 27, 0x00000001, 24, 24, 0x00000000, 20, 22, 0x00000003 }},
    { "mrc", 3, 6,         { 24, 27, 0x0000000e, 20, 20, 0x00000001, 4, 4, 0x00000001 }},
    { "mcr", 3, 0,         { 24, 27, 0x0000000e, 20, 20, 0x00000000, 4, 4, 0x00000001 }},
    { "msr", 3, 0,         { 23, 27, 0x00000006, 20, 21, 0x00000002, 22, 22, 0x00000001 }},
    { "msr", 4, 0,         { 23, 27, 0x00000006, 20, 21, 0x00000002, 22, 22, 0x00000000, 16, 19, 0x00000004 }},
    { "msr", 5, 0,         { 23, 27, 0x00000006, 20, 21, 0x00000002, 22, 22, 0x00000000, 19, 19, 0x00000001, 16, 17, 0x00000000 }},
    { "msr", 4, 0,         { 23, 27, 0x00000006, 20, 21, 0x00000002, 22, 22, 0x00000000, 16, 17, 0x00000001 }},
    { "msr", 4, 0,         { 23, 27, 0x00000006, 20, 21, 0x00000002, 22, 22, 0x00000000, 17, 17, 0x00000001 }},
    { "ldrb", 3, 0,        { 26, 27, 0x00000001, 22, 22, 0x00000001, 20, 20, 0x00000001 }},
    { "strb", 3, 0,        { 26, 27, 0x00000001, 22, 22, 0x00000001, 20, 20, 0x00000000 }},
    { "ldr", 4, 0,         { 28, 31, 0x0000000e, 26, 27, 0x00000001, 22, 22, 0x00000000, 20, 20, 0x00000001 }},
    { "ldrcond", 3, 0,     { 26, 27, 0x00000001, 22, 22, 0x00000000, 20, 20, 0x00000001 }},
    { "str", 3, 0,         { 26, 27, 0x00000001, 22, 22, 0x00000000, 20, 20, 0x00000000 }},
    { "cdp", 2, 0,         { 24, 27, 0x0000000e, 4, 4, 0x00000000 }},
    { "stc", 2, 0,         { 25, 27, 0x00000006, 20, 20, 0x00000000 }},
    { "ldc", 2, 0,         { 25, 27, 0x00000006, 20, 20, 0x00000001 }},
    { "ldrexd", 2, ARMV6K, { 20, 27, 0x0000001B, 4, 7, 0x00000009 }},
    { "strexd", 2, ARMV6K, { 20, 27, 0x0000001A, 4, 7, 0x00000009 }},
    { "ldrexh", 2, ARMV6K, { 20, 27, 0x0000001F, 4, 7, 0x00000009 }},
    { "strexh", 2, ARMV6K, { 20, 27, 0x0000001E, 4, 7, 0x00000009 }},
    { "nop", 5, ARMV6K,    { 23, 27, 0x00000006, 22, 22, 0x00000000, 20, 21, 0x00000002, 16, 19, 0x00000000, 0, 7, 0x00000000 }},
    { "yield", 5, ARMV6K,  { 23, 27, 0x00000006, 22, 22, 0x00000000, 20, 21, 0x00000002, 16, 19, 0x00000000, 0, 7, 0x00000001 }},
    { "wfe", 5, ARMV6K,    { 23, 27, 0x00000006, 22, 22, 0x00000000, 20, 21, 0x00000002, 16, 19, 0x00000000, 0, 7, 0x00000002 }},
    { "wfi", 5, ARMV6K,    { 23, 27, 0x00000006, 22, 22, 0x00000000, 20, 21, 0x00000002, 16, 19, 0x00000000, 0, 7, 0x00000003 }},
    { "sev", 5, ARMV6K,    { 23, 27, 0x00000006, 22, 22, 0x00000000, 20, 21, 0x00000002, 16, 19, 0x00000000, 0, 7, 0x00000004 }},
    { "swi", 1, 0,         { 24, 27, 0x0000000f }},
    { "bbl", 1, 0,         { 25, 27, 0x00000005 }},
};


const InstructionSetEncodingItem arm_exclusion_code[] = {
    { "vmla", 0, ARMVFP2,      { 0 }},
    { "vmls", 0, ARMVFP2,      { 0 }},
    { "vnmla", 0, ARMVFP2,     { 0 }},
    { "vnmls", 0, ARMVFP2,     { 0 }},
    { "vnmul", 0, ARMVFP2,     { 0 }},
    { "vmul", 0, ARMVFP2,      { 0 }},
    { "vadd", 0, ARMVFP2,      { 0 }},
    { "vsub", 0, ARMVFP2,      { 0 }},
    { "vdiv", 0, ARMVFP2,      { 0 }},
    { "vmov(i)", 0, ARMVFP3,   { 0 }},
    { "vmov(r)", 0, ARMVFP3,   { 0 }},
    { "vabs", 0, ARMVFP2,      { 0 }},
    { "vneg", 0, ARMVFP2,      { 0 }},
    { "vsqrt", 0, ARMVFP2,     { 0 }},
    { "vcmp", 0, ARMVFP2,      { 0 }},
    { "vcmp2", 0, ARMVFP2,     { 0 }},
    { "vcvt(bff)", 0, ARMVFP3, { 4, 4, 1 }},
    { "vcvt(bds)", 0, ARMVFP2, { 0 }},
    { "vcvt(bfi)", 0, ARMVFP2, { 0 }},
    { "vmovbrs", 0, ARMVFP2,   { 0 }},
    { "vmsr", 0, ARMVFP2,      { 0 }},
    { "vmovbrc", 0, ARMVFP2,   { 0 }},
    { "vmrs", 0, ARMVFP2,      { 0 }},
    { "vmovbcr", 0, ARMVFP2,   { 0 }},
    { "vmovbrrss", 0, ARMVFP2, { 0 }},
    { "vmovbrrd", 0, ARMVFP2,  { 0 }},
    { "vstr", 0, ARMVFP2,      { 0 }},
    { "vpush", 0, ARMVFP2,     { 0 }},
    { "vstm", 0, ARMVFP2,      { 0 }},
    { "vpop", 0, ARMVFP2,      { 0 }},
    { "vldr", 0, ARMVFP2,      { 0 }},
    { "vldm", 0, ARMVFP2,      { 0 }},

    { "srs", 0, 6,         { 0 }},
    { "rfe", 0, 6,         { 0 }},
    { "bkpt", 0, 3,        { 0 }},
    { "blx", 0, 3,         { 0 }},
    { "cps", 0, 6,         { 0 }},
    { "pld", 0, 4,         { 0 }},
    { "setend", 0, 6,      { 0 }},
    { "clrex", 0, 6,       { 0 }},
    { "rev16", 0, 6,       { 0 }},
    { "usad8", 0, 6,       { 0 }},
    { "sxtb", 0, 6,        { 0 }},
    { "uxtb", 0, 6,        { 0 }},
    { "sxth", 0, 6,        { 0 }},
    { "sxtb16", 0, 6,      { 0 }},
    { "uxth", 0, 6,        { 0 }},
    { "uxtb16", 0, 6,      { 0 }},
    { "cpy", 0, 6,         { 0 }},
    { "uxtab", 0, 6,       { 0 }},
    { "ssub8", 0, 6,       { 0 }},
    { "shsub8", 0, 6,      { 0 }},
    { "ssubaddx", 0, 6,    { 0 }},
    { "strex", 0, 6,       { 0 }},
    { "strexb", 0, 7,      { 0 }},
    { "swp", 0, 0,         { 0 }},
    { "swpb", 0, 0,        { 0 }},
    { "ssub16", 0, 6,      { 0 }},
    { "ssat16", 0, 6,      { 0 }},
    { "shsubaddx", 0, 6,   { 0 }},
    { "qsubaddx", 0, 6,    { 0 }},
    { "shaddsubx", 0, 6,   { 0 }},
    { "shadd8", 0, 6,      { 0 }},
    { "shadd16", 0, 6,     { 0 }},
    { "sel", 0, 6,         { 0 }},
    { "saddsubx", 0, 6,    { 0 }},
    { "sadd8", 0, 6,       { 0 }},
    { "sadd16", 0, 6,      { 0 }},
    { "shsub16", 0, 6,     { 0 }},
    { "umaal", 0, 6,       { 0 }},
    { "uxtab16", 0, 6,     { 0 }},
    { "usubaddx", 0, 6,    { 0 }},
    { "usub8", 0, 6,       { 0 }},
    { "usub16", 0, 6,      { 0 }},
    { "usat16", 0, 6,      { 0 }},
    { "usada8", 0, 6,      { 0 }},
    { "uqsubaddx", 0, 6,   { 0 }},
    { "uqsub8", 0, 6,      { 0 }},
    { "uqsub16", 0, 6,     { 0 }},
    { "uqaddsubx", 0, 6,   { 0 }},
    { "uqadd8", 0, 6,      { 0 }},
    { "uqadd16", 0, 6,     { 0 }},
    { "sxtab", 0, 6,       { 0 }},
    { "uhsubaddx", 0, 6,   { 0 }},
    { "uhsub8", 0, 6,      { 0 }},
    { "uhsub16", 0, 6,     { 0 }},
    { "uhaddsubx", 0, 6,   { 0 }},
    { "uhadd8", 0, 6,      { 0 }},
    { "uhadd16", 0, 6,     { 0 }},
    { "uaddsubx", 0, 6,    { 0 }},
    { "uadd8", 0, 6,       { 0 }},
    { "uadd16", 0, 6,      { 0 }},
    { "sxtah", 0, 6,       { 0 }},
    { "sxtab16", 0, 6,     { 0 }},
    { "qadd8", 0, 6,       { 0 }},
    { "bxj", 0, 5,         { 0 }},
    { "clz", 0, 3,         { 0 }},
    { "uxtah", 0, 6,       { 0 }},
    { "bx", 0, 2,          { 0 }},
    { "rev", 0, 6,         { 0 }},
    { "blx", 0, 3,         { 0 }},
    { "revsh", 0, 6,       { 0 }},
    { "qadd", 0, 4,        { 0 }},
    { "qadd16", 0, 6,      { 0 }},
    { "qaddsubx", 0, 6,    { 0 }},
    { "ldrex", 0, 0,       { 0 }},
    { "qdadd", 0, 4,       { 0 }},
    { "qdsub", 0, 4,       { 0 }},
    { "qsub", 0, 4,        { 0 }},
    { "ldrexb", 0, 7,      { 0 }},
    { "qsub8", 0, 6,       { 0 }},
    { "qsub16", 0, 6,      { 0 }},
    { "smuad", 0, 6,       { 0 }},
    { "smmul", 0, 6,       { 0 }},
    { "smusd", 0, 6,       { 0 }},
    { "smlsd", 0, 6,       { 0 }},
    { "smlsld", 0, 6,      { 0 }},
    { "smmla", 0, 6,       { 0 }},
    { "smmls", 0, 6,       { 0 }},
    { "smlald", 0, 6,      { 0 }},
    { "smlad", 0, 6,       { 0 }},
    { "smlaw", 0, 4,       { 0 }},
    { "smulw", 0, 4,       { 0 }},
    { "pkhtb", 0, 6,       { 0 }},
    { "pkhbt", 0, 6,       { 0 }},
    { "smul", 0, 4,        { 0 }},
    { "smlal", 0, 4,       { 0 }},
    { "smla", 0, 4,        { 0 }},
    { "mcrr", 0, 6,        { 0 }},
    { "mrrc", 0, 6,        { 0 }},
    { "cmp", 3, 0,         { 4, 4, 0x00000001, 7, 7, 0x00000001, 25, 25, 0x00000000 }},
    { "tst", 3, 0,         { 4, 4, 0x00000001, 7, 7, 0x00000001, 25, 25, 0x00000000 }},
    { "teq", 3, 0,         { 4, 4, 0x00000001, 7, 7, 0x00000001, 25, 25, 0x00000000 }},
    { "cmn", 3, 0,         { 4, 4, 0x00000001, 7, 7, 0x00000001, 25, 25, 0x00000000 }},
    { "smull", 0, 0,       { 0 }},
    { "umull", 0, 0,       { 0 }},
    { "umlal", 0, 0,       { 0 }},
    { "smlal", 0, 0,       { 0 }},
    { "mul", 0, 0,         { 0 }},
    { "mla", 0, 0,         { 0 }},
    { "ssat", 0, 6,        { 0 }},
    { "usat", 0, 6,        { 0 }},
    { "mrs", 0, 0,         { 0 }},
    { "msr", 0, 0,         { 0 }},
    { "and", 3, 0,         { 4, 4, 0x00000001, 7, 7, 0x00000001, 25, 25, 0x00000000 }},
    { "bic", 3, 0,         { 4, 4, 0x00000001, 7, 7, 0x00000001, 25, 25, 0x00000000 }},
    { "ldm", 0, 0,         { 0 }},
    { "eor", 3, 0,         { 4, 4, 0x00000001, 7, 7, 0x00000001, 25, 25, 0x00000000 }},
    { "add", 3, 0,         { 4, 4, 0x00000001, 7, 7, 0x00000001, 25, 25, 0x00000000 }},
    { "rsb", 3, 0,         { 4, 4, 0x00000001, 7, 7, 0x00000001, 25, 25, 0x00000000 }},
    { "rsc", 3, 0,         { 4, 4, 0x00000001, 7, 7, 0x00000001, 25, 25, 0x00000000 }},
    { "sbc", 3, 0,         { 4, 4, 0x00000001, 7, 7, 0x00000001, 25, 25, 0x00000000 }},
    { "adc", 3, 0,         { 4, 4, 0x00000001, 7, 7, 0x00000001, 25, 25, 0x00000000 }},
    { "sub", 3, 0,         { 4, 4, 0x00000001, 7, 7, 0x00000001, 25, 25, 0x00000000 }},
    { "orr", 3, 0,         { 4, 4, 0x00000001, 7, 7, 0x00000001, 25, 25, 0x00000000 }},
    { "mvn", 3, 0,         { 4, 4, 0x00000001, 7, 7, 0x00000001, 25, 25, 0x00000000 }},
    { "mov", 3, 0,         { 4, 4, 0x00000001, 7, 7, 0x00000001, 25, 25, 0x00000000 }},
    { "stm", 0, 0,         { 0 }},
    { "ldm", 0, 0,         { 0 }},
    { "ldrsh", 0, 2,       { 0 }},
    { "stm", 0, 0,         { 0 }},
    { "ldm", 0, 0,         { 0 }},
    { "ldrsb", 0, 2,       { 0 }},
    { "strd", 0, 4,        { 0 }},
    { "ldrh", 0, 0,        { 0 }},
    { "strh", 0, 0,        { 0 }},
    { "ldrd", 0, 4,        { 0 }},
    { "strt", 0, 0,        { 0 }},
    { "strbt", 0, 0,       { 0 }},
    { "ldrbt", 0, 0,       { 0 }},
    { "ldrt", 0, 0,        { 0 }},
    { "mrc", 0, 6,         { 0 }},
    { "mcr", 0, 0,         { 0 }},
    { "msr", 0, 0,         { 0 }},
    { "msr", 0, 0,         { 0 }},
    { "msr", 0, 0,         { 0 }},
    { "msr", 0, 0,         { 0 }},
    { "msr", 0, 0,         { 0 }},
    { "ldrb", 0, 0,        { 0 }},
    { "strb", 0, 0,        { 0 }},
    { "ldr", 0, 0,         { 0 }},
    { "ldrcond", 1, 0,     { 28, 31, 0x0000000e }},
    { "str", 0, 0,         { 0 }},
    { "cdp", 0, 0,         { 0 }},
    { "stc", 0, 0,         { 0 }},
    { "ldc", 0, 0,         { 0 }},
    { "ldrexd", 0, ARMV6K, { 0 }},
    { "strexd", 0, ARMV6K, { 0 }},
    { "ldrexh", 0, ARMV6K, { 0 }},
    { "strexh", 0, ARMV6K, { 0 }},
    { "nop", 0, ARMV6K,    { 0 }},
    { "yield", 0, ARMV6K,  { 0 }},
    { "wfe", 0, ARMV6K,    { 0 }},
    { "wfi", 0, ARMV6K,    { 0 }},
    { "sev", 0, ARMV6K,    { 0 }},
    { "swi", 0, 0,         { 0 }},
    { "bbl", 0, 0,         { 0 }},

    { "bl_1_thumb", 0, INVALID,  { 0 }}, // Should be table[-4]
    { "bl_2_thumb", 0, INVALID,  { 0 }}, // Should be located at the end of the table[-3]
    { "blx_1_thumb", 0, INVALID, { 0 }}, // Should be located at table[-2]
    { "invalid", 0, INVALID,     { 0 }}
};
// clang-format on

// Optimized instruction decoder using a binary search tree for faster matching
namespace {
// Optimized bit extraction function to replace the BITS macro
template<typename T>
inline T ExtractBits(T value, u32 start, u32 end) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    // ARM-optimized implementation
    const u32 num_bits = end - start + 1;
    const T mask = (static_cast<T>(1) << num_bits) - 1;
    return (value >> start) & mask;
#else
    // Standard implementation that exactly matches the BITS macro
    // #define BITS(s, a, b) ((s << ((sizeof(s) * 8 - 1) - b)) >> (sizeof(s) * 8 - b + a - 1))
    const u32 type_size = sizeof(T) * 8;
    return ((value << ((type_size - 1) - end)) >> (type_size - end + start - 1));
#endif
}
    // Pre-computed hash table mapping instruction keys to potential instruction indices
    struct InstrLookupEntry {
        u8 num_candidates = 0; // Number of potential matches
        u8 indices[8] = {};    // Indices into arm_instruction array (most instructions with same key < 8)
    };

    // Lookup table - initialized at first use
    InstrLookupEntry instr_lookup_table[256] = {};
    bool lookup_table_initialized = false;
    
    // Binary search tree node for instruction lookups
    struct InstrBSTNode {
        u32 key_mask = 0;       // Mask of bits that are relevant for this node
        u32 key_value = 0;      // Expected value of those bits
        int instruction_idx = -1; // Index into arm_instruction array if this is a leaf node
        InstrBSTNode* left = nullptr;  // Child node if bits don't match
        InstrBSTNode* right = nullptr; // Child node if bits match
    };

    // Root of the binary search tree - initialized at first use
    InstrBSTNode* bst_root = nullptr;
    bool bst_initialized = false;

    // Pool of BST nodes to avoid dynamic allocation during initialization
    static constexpr size_t MAX_BST_NODES = 1024; // Should be enough for all instructions
    InstrBSTNode bst_node_pool[MAX_BST_NODES];
    size_t next_free_node = 0;

    // Get a new BST node from the pool
    InstrBSTNode* AllocateBSTNode() {
        if (next_free_node >= MAX_BST_NODES) {
            LOG_ERROR(Core_ARM11, "BST node pool exhausted!");
            return nullptr;
        }
        return &bst_node_pool[next_free_node++];
    }

    // No longer need a hash key function since we're using a binary search tree
    // We'll keep this function for backward compatibility with the instruction cache
    inline u8 ExtractInstrKey(u32 instr) {
        // Use bits 20-27 as they're the most discriminative for ARM instructions
        // This is a simple shift and mask operation that's fast on all platforms
        return static_cast<u8>((instr >> 20) & 0xFF);
    }

    // Helper function to find the most discriminative bit for a set of instructions
    // Creates a weight-balanced tree for optimal search performance
    std::pair<u32, u32> FindBestDiscriminativeBit(const std::vector<int>& instruction_indices) {
        // If there's only one instruction or empty list, use a safe default
        if (instruction_indices.size() <= 1) {
            return {1U, 0}; // Use bit 0 with value 0
        }
        
        // For weight balancing, we'll analyze both bit values (0 and 1) for each bit position
        std::array<int, 32> bit_ones = {}; // Count of instructions with bit=1
        std::array<int, 32> bit_zeros = {}; // Count of instructions with bit=0
        std::array<int, 32> bit_coverage = {}; // Count of instructions where this bit is specified
        
        // For each instruction, analyze its bit patterns
        for (size_t i = 0; i < instruction_indices.size(); i++) {
            int idx = instruction_indices[i];
            
            // Safety check for valid index
            if (idx < 0 || idx >= static_cast<int>(sizeof(arm_instruction) / sizeof(arm_instruction[0]))) {
                LOG_ERROR(Core_ARM11, "Invalid instruction index %d in FindBestDiscriminativeBit", idx);
                continue; // Skip invalid indices
            }
            
            const auto& instr = arm_instruction[idx];
            
            // Skip invalid attribute values
            if (instr.attribute_value <= 0 || instr.attribute_value > 100) { // Arbitrary upper limit for safety
                LOG_WARNING(Core_ARM11, "Suspicious attribute value %d for instruction %d", instr.attribute_value, idx);
                continue;
            }
            
            // Track which bits we've seen for this instruction
            std::array<bool, 32> seen_bits = {};
            
            // Look at each pattern in the instruction
            for (int j = 0; j < instr.attribute_value; j++) {
                // Safety check for content array access
                if (j*3 + 2 >= static_cast<int>(sizeof(instr.content) / sizeof(instr.content[0]))) {
                    LOG_WARNING(Core_ARM11, "Content array index out of bounds for instruction %d", idx);
                    break;
                }
                
                int bit_start = instr.content[j*3];
                int bit_end = instr.content[j*3 + 1];
                u32 pattern_value = instr.content[j*3 + 2];
                
                // Validate bit range
                if (bit_start < 0 || bit_end >= 32 || bit_start > bit_end) {
                    LOG_WARNING(Core_ARM11, "Invalid bit range [%d,%d] for instruction %d", bit_start, bit_end, idx);
                    continue; // Skip invalid bit ranges
                }
                
                // For each bit in the pattern
                for (int bit = bit_start; bit <= bit_end && bit < 32; bit++) {
                    // Skip if we've already processed this bit for this instruction
                    if (seen_bits[bit]) continue;
                    seen_bits[bit] = true;
                    
                    // Calculate if this bit is set in the pattern
                    int bit_pos = bit - bit_start;
                    if (bit_pos < 0 || bit_pos >= 32) continue; // Safety check
                    
                    // Count this bit as covered
                    bit_coverage[bit]++;
                    
                    // Count if the bit is 0 or 1
                    if ((pattern_value & (1U << bit_pos)) != 0) {
                        bit_ones[bit]++;
                    } else {
                        bit_zeros[bit]++;
                    }
                }
            }
        }
        
        // Find the bit position that creates the most weight-balanced split
        int best_bit = -1;
        double best_balance = -1.0;
        bool use_one_branch = false;
        size_t total = instruction_indices.size();
        
        // Prevent division by zero
        if (total == 0) {
            LOG_ERROR(Core_ARM11, "Empty instruction set in FindBestDiscriminativeBit");
            return {1U, 0}; // Use bit 0 with value 0
        }
        
        for (int bit = 0; bit < 32; bit++) {
            // Skip bits that don't have good coverage
            if (bit_coverage[bit] < total * 0.7) continue;
            
            // Calculate balance factors for both 0 and 1 branches
            double balance_zero = static_cast<double>(bit_zeros[bit]) / total;
            double balance_one = static_cast<double>(bit_ones[bit]) / total;
            
            // Normalize to 0-0.5 range (closer to 0.5 is better balanced)
            if (balance_zero > 0.5) balance_zero = 1.0 - balance_zero;
            if (balance_one > 0.5) balance_one = 1.0 - balance_one;
            
            // Use the better of the two balances
            double balance = std::max(balance_zero, balance_one);
            bool use_one = (balance_one >= balance_zero);
            
            // We want the most balanced split (closest to 0.5)
            if (balance > best_balance) {
                best_balance = balance;
                best_bit = bit;
                use_one_branch = use_one;
            }
        }
        
        // If we couldn't find a good bit, try again with lower coverage requirement
        if (best_bit == -1) {
            for (int bit = 0; bit < 32; bit++) {
                // Skip bits with no coverage
                if (bit_coverage[bit] == 0) continue;
                
                // Calculate balance factors with safety check for division by zero
                double balance_zero = 0.0;
                double balance_one = 0.0;
                
                if (bit_coverage[bit] > 0) {
                    balance_zero = static_cast<double>(bit_zeros[bit]) / bit_coverage[bit];
                    balance_one = static_cast<double>(bit_ones[bit]) / bit_coverage[bit];
                }
                
                // Normalize to 0-0.5 range
                if (balance_zero > 0.5) balance_zero = 1.0 - balance_zero;
                if (balance_one > 0.5) balance_one = 1.0 - balance_one;
                
                double balance = std::max(balance_zero, balance_one);
                bool use_one = (balance_one >= balance_zero);
                
                if (balance > best_balance) {
                    best_balance = balance;
                    best_bit = bit;
                    use_one_branch = use_one;
                }
            }
        }
        
        // If we still couldn't find a good bit, use bit 0 as default
        if (best_bit == -1) {
            LOG_WARNING(Core_ARM11, "Could not find a good discriminative bit, using default");
            best_bit = 0;
            use_one_branch = false;
        }
        
        // Safety check for bit_pos
        if (best_bit < 0 || best_bit >= 32) {
            LOG_ERROR(Core_ARM11, "Invalid best_bit %d, using default", best_bit);
            best_bit = 0;
            use_one_branch = false;
        }
        
        return {1U << best_bit, use_one_branch ? (1U << best_bit) : 0};
    }
    
    // Non-recursive helper to create a leaf node
    InstrBSTNode* CreateLeafNode(int instruction_idx) {
        InstrBSTNode* node = AllocateBSTNode();
        if (!node) return nullptr;
        
        node->instruction_idx = instruction_idx;
        node->left = nullptr;
        node->right = nullptr;
        node->key_mask = 0;
        node->key_value = 0;
        
        return node;
    }
    
    // Build a weight-balanced binary search tree using an iterative approach to prevent stack overflow
    InstrBSTNode* BuildBSTNode(const std::vector<int>& instruction_indices) {
        // Handle base cases
        if (instruction_indices.empty()) {
            return nullptr;
        }
        
        if (instruction_indices.size() == 1) {
            return CreateLeafNode(instruction_indices[0]);
        }
        
        // Safety check - if we're about to exceed our node pool, return a leaf node with the first instruction
        if (next_free_node >= MAX_BST_NODES - 10) { // Leave room for a few more nodes
            LOG_WARNING(Core_ARM11, "BST node pool nearly exhausted, creating leaf node instead");
            return CreateLeafNode(instruction_indices[0]);
        }
        
        // Use a queue-based approach for breadth-first tree construction
        // Each entry contains: (node pointer, vector of instruction indices to process)
        struct BuildTask {
            InstrBSTNode* node;
            std::vector<int> indices;
            bool is_left_child;
            InstrBSTNode* parent;
        };
        
        std::vector<BuildTask> tasks;
        
        // Create the root node first
        InstrBSTNode* root = AllocateBSTNode();
        if (!root) return nullptr;
        
        // Initialize root as internal node
        root->instruction_idx = -1;
        root->left = nullptr;
        root->right = nullptr;
        
        // Add the initial task for the root
        tasks.push_back({root, instruction_indices, false, nullptr});
        
        // Process tasks in a breadth-first manner
        size_t current_task = 0;
        
        // Limit the number of iterations to prevent infinite loops
        const size_t max_iterations = MAX_BST_NODES * 2;
        size_t iteration_count = 0;
        
        while (current_task < tasks.size() && iteration_count < max_iterations) {
            iteration_count++;
            
            // Get the current task
            BuildTask& task = tasks[current_task++];
            InstrBSTNode* node = task.node;
            const std::vector<int>& indices = task.indices;
            
            // If we only have one instruction, make this a leaf node
            if (indices.size() == 1) {
                node->instruction_idx = indices[0];
                continue; // Done with this node
            }
            
            // Find the best bit to split on
            auto [mask, value] = FindBestDiscriminativeBit(indices);
            
            // Safety check for mask
            if (mask == 0) {
                LOG_WARNING(Core_ARM11, "Invalid mask 0 in BuildBSTNode, creating leaf node");
                node->instruction_idx = indices[0];
                continue; // Done with this node
            }
            
            // Set node properties
            node->key_mask = mask;
            node->key_value = value;
            
            // Find which bit position we're testing
            int bit_pos = __builtin_ctz(mask); // Get position of least significant bit
            bool expected_value = (value != 0);
            
            // Split instructions based on this bit
            std::vector<int> left_indices;
            std::vector<int> right_indices;
            
            // Process each instruction
            for (size_t i = 0; i < indices.size(); i++) {
                int idx = indices[i];
                if (idx < 0 || idx >= static_cast<int>(sizeof(arm_instruction) / sizeof(arm_instruction[0]))) {
                    LOG_ERROR(Core_ARM11, "Invalid instruction index %d in BuildBSTNode", idx);
                    continue; // Skip invalid indices
                }
                
                const auto& instr = arm_instruction[idx];
                bool matches = false;
                
                // Check if this instruction matches the bit pattern
                for (int j = 0; j < instr.attribute_value && !matches; j++) {
                    int bit_start = instr.content[j*3];
                    int bit_end = instr.content[j*3 + 1];
                    
                    // Validate bit range
                    if (bit_start < 0 || bit_end >= 32 || bit_start > bit_end) {
                        continue; // Skip invalid bit ranges
                    }
                    
                    u32 pattern_value = instr.content[j*3 + 2];
                    
                    // Check if this pattern covers our bit of interest
                    if (bit_start <= bit_pos && bit_end >= bit_pos) {
                        // Extract the bit from the pattern
                        int shift = bit_pos - bit_start;
                        if (shift >= 0 && shift < 32) {
                            bool bit_value = ((pattern_value >> shift) & 1) != 0;
                            matches = (bit_value == expected_value);
                            break;
                        }
                    }
                }
                
                // Add to appropriate child list
                if (matches) {
                    right_indices.push_back(idx);
                } else {
                    left_indices.push_back(idx);
                }
            }
            
            // Handle degenerate cases
            if (left_indices.empty() && right_indices.size() == 1) {
                // All instructions went to the right branch and there's only one
                node->instruction_idx = right_indices[0];
                continue; // Done with this node
            } else if (right_indices.empty() && left_indices.size() == 1) {
                // All instructions went to the left branch and there's only one
                node->instruction_idx = left_indices[0];
                continue; // Done with this node
            }
            
            // Check if we're about to exceed node pool capacity
            if (next_free_node >= MAX_BST_NODES - 2) {
                LOG_WARNING(Core_ARM11, "BST node pool exhausted during iterative build");
                node->instruction_idx = indices[0]; // Convert to leaf node
                continue; // Done with this node
            }
            
            // Create child nodes if needed
            if (!left_indices.empty()) {
                InstrBSTNode* left_node = AllocateBSTNode();
                if (left_node) {
                    left_node->instruction_idx = -1; // Mark as internal node initially
                    left_node->left = nullptr;
                    left_node->right = nullptr;
                    node->left = left_node;
                    
                    // Add left child task
                    tasks.push_back({left_node, std::move(left_indices), true, node});
                } else {
                    LOG_ERROR(Core_ARM11, "Failed to allocate left child node");
                }
            }
            
            if (!right_indices.empty()) {
                InstrBSTNode* right_node = AllocateBSTNode();
                if (right_node) {
                    right_node->instruction_idx = -1; // Mark as internal node initially
                    right_node->left = nullptr;
                    right_node->right = nullptr;
                    node->right = right_node;
                    
                    // Add right child task
                    tasks.push_back({right_node, std::move(right_indices), false, node});
                } else {
                    LOG_ERROR(Core_ARM11, "Failed to allocate right child node");
                }
            }
        }
        
        // Check if we hit the iteration limit
        if (iteration_count >= max_iterations) {
            LOG_ERROR(Core_ARM11, "Exceeded maximum iterations in iterative BST build");
        }
        
        return root;
    }
    
    // Initialize the weight-balanced binary search tree at runtime
    void InitBST() {
        if (bst_initialized) {
            LOG_INFO(Core_ARM11, "BST already initialized, skipping");
            return;
        }

        // Reset the node pool
        next_free_node = 0;
        
        // Initialize all nodes to a safe state
        for (size_t i = 0; i < MAX_BST_NODES; i++) {
            bst_node_pool[i].instruction_idx = -1;
            bst_node_pool[i].key_mask = 0;
            bst_node_pool[i].key_value = 0;
            bst_node_pool[i].left = nullptr;
            bst_node_pool[i].right = nullptr;
        }
        
        // Collect all valid instruction indices
        std::vector<int> all_indices;
        int instr_slots = sizeof(arm_instruction) / sizeof(InstructionSetEncodingItem);
        
        LOG_INFO(Core_ARM11, "Processing %d instruction slots for BST initialization", instr_slots);
        
        for (int i = 0; i < instr_slots; i++) {
            // Skip VFP3 instructions as 3DS doesn't support them
            if (arm_instruction[i].version == ARMVFP3)
                continue;

            // Only process instructions with at least one pattern
            if (arm_instruction[i].attribute_value <= 0 || 
                arm_instruction[i].attribute_value > 100) { // Arbitrary upper limit for safety
                LOG_WARNING(Core_ARM11, "Skipping instruction %d with suspicious attribute value %d", 
                           i, arm_instruction[i].attribute_value);
                continue;
            }
                
            all_indices.push_back(i);
        }
        
        LOG_INFO(Core_ARM11, "Building weight-balanced BST with %zu valid instructions", all_indices.size());
        
        // Safety check - make sure we have instructions to process
        if (all_indices.empty()) {
            LOG_ERROR(Core_ARM11, "No valid instructions found for BST initialization");
            bst_initialized = true; // Mark as initialized to prevent repeated attempts
            return;
        }
        
        try {
            // Build the weight-balanced binary search tree
            bst_root = BuildBSTNode(all_indices);
            
            // Verify the root node was created
            if (!bst_root) {
                LOG_ERROR(Core_ARM11, "Failed to create BST root node");
                bst_initialized = true; // Mark as initialized to prevent repeated attempts
                return;
            }
            
            bst_initialized = true;
            LOG_INFO(Core_ARM11, "ARM instruction weight-balanced BST initialized with %zu nodes", next_free_node);
        } catch (const std::exception& e) {
            LOG_ERROR(Core_ARM11, "Exception during BST initialization: %s", e.what());
            bst_initialized = true; // Mark as initialized to prevent repeated attempts
        } catch (...) {
            LOG_ERROR(Core_ARM11, "Unknown exception during BST initialization");
            bst_initialized = true; // Mark as initialized to prevent repeated attempts
        }
    }
    
    // For backward compatibility, keep the old lookup table initialization
    void InitLookupTable() {
        // We'll initialize both the lookup table and the BST
        if (lookup_table_initialized)
            return;

        // Initialize the BST first
        InitBST();
        
        // Populate the table for backward compatibility
        int instr_slots = sizeof(arm_instruction) / sizeof(InstructionSetEncodingItem);
        for (int i = 0; i < instr_slots; i++) {
            // Skip VFP3 instructions as 3DS doesn't support them
            if (arm_instruction[i].version == ARMVFP3)
                continue;

            // Only process instructions with at least one pattern
            if (arm_instruction[i].attribute_value == 0)
                continue;

            // Calculate the key from the instruction pattern
            u8 key = 0;
            bool key_found = false;

            // Extract key bits from the pattern
            for (int j = 0; j < arm_instruction[i].attribute_value && !key_found; j++) {
                int bit_start = arm_instruction[i].content[j*3];
                int bit_end = arm_instruction[i].content[j*3 + 1];
                u32 pattern_value = arm_instruction[i].content[j*3 + 2];

                // Check if this pattern covers our key range (bits 20-27)
                if (bit_start <= 27 && bit_end >= 20) {
                    // Calculate how many bits from our key range this pattern covers
                    int start_bit = std::max(20, bit_start);
                    int end_bit = std::min(27, bit_end);
                    int num_bits = end_bit - start_bit + 1;

                    // Extract the relevant bits from the pattern value
                    int shift = start_bit - bit_start;
                    u32 mask = ((1U << num_bits) - 1) << shift;
                    u32 extracted = (pattern_value & mask) >> shift;

                    // Place these bits in the correct position in our key
                    key |= (extracted << (start_bit - 20));

                    // Mark that we've found at least some key bits
                    key_found = true;
                }
            }

            // If we couldn't extract a key from the patterns, use a default hash
            if (!key_found) {
                // Use a simple hash based on the instruction name
                const char* name = arm_instruction[i].name;
                key = 0;
                while (*name) {
                    key = (key * 31 + *name) & 0xFF;
                    name++;
                }
            }

            // Add to lookup table if there's room
            if (instr_lookup_table[key].num_candidates < 8) {
                instr_lookup_table[key].indices[instr_lookup_table[key].num_candidates++] = i;
            }
        }

        lookup_table_initialized = true;
    }

    // Fast check for instruction match
    inline bool CheckInstructionMatch(u32 instr, const InstructionSetEncodingItem& pattern) {
        int n = pattern.attribute_value;
        if (n == 0) return true;  // Empty pattern always matches

        int base = 0;

        while (n) {
            if (pattern.content[base + 1] == 31 && pattern.content[base] == 0) {
                // Special case for clrex and other full-word matches
                if (instr != pattern.content[base + 2]) {
                    return false;
                }
            } else {
                // Normal case - check if bits match expected pattern
                u32 start_bit = pattern.content[base];
                u32 end_bit = pattern.content[base + 1];
                u32 expected_value = pattern.content[base + 2];

                // Extract bits directly for better performance
                u32 num_bits = end_bit - start_bit + 1;
                u32 mask = (1U << num_bits) - 1;
                u32 extracted_bits = (instr >> start_bit) & mask;

                if (extracted_bits != expected_value) {
                    return false;
                }
            }
            base += 3;
            n--;
        }

        return true;
    }

    // Fast check for exclusion match
    inline bool CheckExclusionMatch(u32 instr, const InstructionSetEncodingItem& pattern) {
        int n = pattern.attribute_value;
        if (n == 0) return false;

        int base = 0;
        while (n) {
            u32 start_bit = pattern.content[base];
            u32 end_bit = pattern.content[base + 1];
            u32 expected_value = pattern.content[base + 2];

            // Extract bits directly for better performance
            u32 num_bits = end_bit - start_bit + 1;
            u32 mask = (1U << num_bits) - 1;
            u32 extracted_bits = (instr >> start_bit) & mask;

            if (extracted_bits != expected_value) {
                return false;
            }
            base += 3;
            n--;
        }

        return true;
    }
    
    // Search the weight-balanced binary search tree for a matching instruction
    int SearchBST(u32 instr) {
        if (!bst_initialized) {
            InitBST();
        }
        
        // Start at the root node
        InstrBSTNode* node = bst_root;
        if (!node) {
            LOG_ERROR(Core_ARM11, "BST root is null");
            return -1;
        }
        
        // Prevent infinite loops by limiting traversal depth
        int max_depth = 100; // This should be more than enough for any reasonable tree
        int current_depth = 0;
        
        // Traverse the tree until we reach a leaf node
        while (node && node->instruction_idx == -1 && current_depth < max_depth) {
            // Check if the instruction matches the current node's pattern
            // For weight-balanced trees, we use a precise bit comparison
            u32 mask = node->key_mask;
            u32 value = node->key_value;
            
            // Safety check for invalid mask
            if (mask == 0) {
                LOG_ERROR(Core_ARM11, "Invalid mask 0 in SearchBST at depth %d", current_depth);
                break;
            }
            
            bool matches = ((instr & mask) == value);
            
            // Go to the appropriate child node
            InstrBSTNode* next_node = matches ? node->right : node->left;
            
            // Check for null child pointer
            if (!next_node) {
                // This is unexpected in our tree structure, but handle it gracefully
                LOG_WARNING(Core_ARM11, "Unexpected null child in BST at depth %d", current_depth);
                break;
            }
            
            node = next_node;
            current_depth++;
        }
        
        // Check if we hit the depth limit
        if (current_depth >= max_depth) {
            LOG_ERROR(Core_ARM11, "BST traversal exceeded max depth, possible infinite loop");
            return -1;
        }
        
        // If we found a leaf node, return its instruction index
        if (node && node->instruction_idx != -1) {
            int idx = node->instruction_idx;
            
            // Safety check for valid index
            if (idx < 0 || idx >= static_cast<int>(sizeof(arm_instruction) / sizeof(arm_instruction[0]))) {
                LOG_ERROR(Core_ARM11, "Invalid instruction index %d in SearchBST", idx);
                return -1;
            }
            
            // Verify that the instruction actually matches the pattern
            if (CheckInstructionMatch(instr, arm_instruction[idx])) {
                // Check exclusions
                if (!CheckExclusionMatch(instr, arm_exclusion_code[idx])) {
                    return idx;
                }
            }
        }
        
        // No match found
        return -1;
    }
// Global instruction decode cache
static std::unordered_map<u32, ARMInstructionInfo> instruction_cache;
} // namespace

// Clear the instruction decode cache
void ClearARMInstructionCache() {
    instruction_cache.clear();
}

ARMDecodeStatus DecodeARMInstruction(u32 instr, int* idx) {
    // Check if we have this instruction in our cache
    auto cache_it = instruction_cache.find(instr);
    if (cache_it != instruction_cache.end()) {
        // Cache hit! Use the cached result
        if (cache_it->second.status == ARMDecodeStatus::SUCCESS) {
            *idx = cache_it->second.instruction_index;
        }
        return cache_it->second.status;
    }

    // Cache miss - need to decode the instruction
    ARMInstructionInfo result;
    
    // First, try using the weight-balanced binary search tree for optimal lookup performance
    int bst_result = SearchBST(instr);
    if (bst_result >= 0) {
        // Found a match in the weight-balanced BST!
        *idx = bst_result;
        
        // Cache the successful result
        result.instruction_index = bst_result;
        result.status = ARMDecodeStatus::SUCCESS;
        instruction_cache[instr] = result;
        
        return ARMDecodeStatus::SUCCESS;
    }
    
    // Weight-balanced BST lookup failed, fall back to the hash table for backward compatibility
    // This should rarely happen with our optimized tree structure
    // Initialize lookup table if needed
    if (!lookup_table_initialized) {
        InitLookupTable();
    }
    
    // Get the key bits from the instruction
    u8 key = ExtractInstrKey(instr);

    // Look up potential matches
    const InstrLookupEntry& entry = instr_lookup_table[key];

    // Check each candidate
    for (int i = 0; i < entry.num_candidates; i++) {
        int candidate_idx = entry.indices[i];

        // Skip VFP3 instructions (shouldn't be in table, but double-check)
        if (arm_instruction[candidate_idx].version == ARMVFP3)
            continue;

        // Check if instruction matches the pattern
        if (CheckInstructionMatch(instr, arm_instruction[candidate_idx])) {
            // Check exclusions
            if (CheckExclusionMatch(instr, arm_exclusion_code[candidate_idx])) {
                continue; // Exclusion matched, so this isn't the right instruction
            }

            // Found a match!
            *idx = candidate_idx;
            
            // Cache the successful result
            result.instruction_index = candidate_idx;
            result.status = ARMDecodeStatus::SUCCESS;
            instruction_cache[instr] = result;
            
            // This is a miss in our weight-balanced BST, which shouldn't happen often
            // Log it for debugging purposes
            LOG_DEBUG(Core_ARM11, "Weight-balanced BST miss for instruction 0x%08x, found via hash table", instr);
            
            return ARMDecodeStatus::SUCCESS;
        }
    }

    // If we got here, we need to do a full search as fallback
    // This handles cases where both the BST and hash table might have missed something
    int instr_slots = sizeof(arm_instruction) / sizeof(InstructionSetEncodingItem);

    for (int i = 0; i < instr_slots; i++) {
        // Skip VFP3 instructions
        if (arm_instruction[i].version == ARMVFP3)
            continue;

        // Skip instructions we already checked via the lookup table
        bool already_checked = false;
        for (int j = 0; j < entry.num_candidates; j++) {
            if (entry.indices[j] == i) {
                already_checked = true;
                break;
            }
        }
        if (already_checked)
            continue;

        // Check if instruction matches the pattern
        if (CheckInstructionMatch(instr, arm_instruction[i])) {
            // Check exclusions
            if (CheckExclusionMatch(instr, arm_exclusion_code[i])) {
                continue; // Exclusion matched, so this isn't the right instruction
            }

            // Found a match!
            *idx = i;
            
            // Cache the successful result
            result.instruction_index = i;
            result.status = ARMDecodeStatus::SUCCESS;
            instruction_cache[instr] = result;
            
            // This is a complete miss in all our optimized structures
            // Log it for debugging purposes
            LOG_DEBUG(Core_ARM11, "Complete lookup miss for instruction 0x%08x, found via full scan", instr);
            
            return ARMDecodeStatus::SUCCESS;
        }
    }

    // Cache the failure result
    result.instruction_index = 0;
    result.status = ARMDecodeStatus::FAILURE;
    instruction_cache[instr] = result;
    
    return ARMDecodeStatus::FAILURE;
}
