// Copyright 2012 Michael Kang, 2014 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "core/arm/dyncom/arm_dyncom_dec.h"
#include "core/arm/skyeye_common/armsupp.h"

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

// Optimized instruction decoder using a hash-based approach for faster matching
namespace {
#if defined(__ARM_NEON) || defined(__aarch64__)
    // ARM-optimized bit extraction function to replace the BITS macro
    template<typename T>
    inline T ExtractBits(T value, u32 start, u32 end) {
        // This implementation is optimized for ARM platforms
        // It extracts bits from start to end (inclusive) from the value
        const u32 type_size = sizeof(T) * 8;
        const u32 num_bits = end - start + 1;
        const T mask = (static_cast<T>(1) << num_bits) - 1;
        return (value >> start) & mask;
    }
#else
    // Standard bit extraction function to match the BITS macro exactly
    template<typename T>
    inline T ExtractBits(T value, u32 start, u32 end) {
        // This implementation exactly matches the BITS macro:
        // #define BITS(s, a, b) ((s << ((sizeof(s) * 8 - 1) - b)) >> (sizeof(s) * 8 - b + a - 1))
        const u32 type_size = sizeof(T) * 8;
        return ((value << ((type_size - 1) - end)) >> (type_size - end + start - 1));
    }
#endif
    // Pre-computed hash table mapping instruction keys to potential instruction indices
    struct InstrLookupEntry {
        u8 num_candidates = 0; // Number of potential matches
        u8 indices[8] = {};    // Indices into arm_instruction array (most instructions with same key < 8)
    };

    // Lookup table - initialized at first use
    InstrLookupEntry instr_lookup_table[256] = {};
    bool lookup_table_initialized = false;

    // Extract key bits from instruction to use as a hash key
#if defined(__ARM_NEON) || defined(__aarch64__)
    inline u8 ExtractInstrKey(u32 instr) {
        // Use bits 20-27 as they're the most discriminative for ARM instructions
        // Optimized for ARM64 - use our optimized ExtractBits function
        return static_cast<u8>(ExtractBits<u32>(instr, 20, 27));
    }
#else
    inline u8 ExtractInstrKey(u32 instr) {
        // Use bits 20-27 as they're the most discriminative for ARM instructions
        // Use our standard ExtractBits function that matches the BITS macro
        return static_cast<u8>(ExtractBits<u32>(instr, 20, 27));
    }
#endif

    // Initialize the lookup table at runtime
    void InitLookupTable() {
        if (lookup_table_initialized)
            return;

        // Populate the table
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

                // Extract bits using our optimized function with explicit template parameter
                u32 extracted_bits = ExtractBits<u32>(instr, start_bit, end_bit);

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

            // Extract bits using our optimized function with explicit template parameter
            u32 extracted_bits = ExtractBits<u32>(instr, start_bit, end_bit);

            if (extracted_bits != expected_value) {
                return false;
            }
            base += 3;
            n--;
        }

        return true;
    }
#if defined(__ARM_NEON) || defined(__aarch64__)
    // Global instruction decode cache for ARM64/NEON platforms
    static std::unordered_map<u32, ARMInstructionInfo> instruction_cache;
#else
    // Global instruction decode cache for other platforms
    static std::unordered_map<u32, ARMInstructionInfo> instruction_cache;
#endif
} // namespace

// Clear the instruction decode cache
void ClearARMInstructionCache() {
#if defined(__ARM_NEON) || defined(__aarch64__)
    // Clear cache for ARM64/NEON platforms
    instruction_cache.clear();
#else
    // Clear cache for other platforms
    instruction_cache.clear();
#endif
}

ARMDecodeStatus DecodeARMInstruction(u32 instr, int* idx) {
    // Initialize lookup table if needed
    if (!lookup_table_initialized) {
        InitLookupTable();
    }

#if defined(__ARM_NEON) || defined(__aarch64__)
    // ARM64/NEON optimized path
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
#else
    // Standard path for other platforms
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
#endif
    
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
#if defined(__ARM_NEON) || defined(__aarch64__)
            result.instruction_index = candidate_idx;
            result.status = ARMDecodeStatus::SUCCESS;
            instruction_cache[instr] = result;
#else
            result.instruction_index = candidate_idx;
            result.status = ARMDecodeStatus::SUCCESS;
            instruction_cache[instr] = result;
#endif
            
            return ARMDecodeStatus::SUCCESS;
        }
    }

    // If we got here, we need to do a full search as fallback
    // This handles cases where our hash table might have missed something
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
#if defined(__ARM_NEON) || defined(__aarch64__)
            result.instruction_index = i;
            result.status = ARMDecodeStatus::SUCCESS;
            instruction_cache[instr] = result;
#else
            result.instruction_index = i;
            result.status = ARMDecodeStatus::SUCCESS;
            instruction_cache[instr] = result;
#endif
            
            return ARMDecodeStatus::SUCCESS;
        }
    }

    // Cache the failure result
#if defined(__ARM_NEON) || defined(__aarch64__)
    result.instruction_index = 0;
    result.status = ARMDecodeStatus::FAILURE;
    instruction_cache[instr] = result;
#else
    result.instruction_index = 0;
    result.status = ARMDecodeStatus::FAILURE;
    instruction_cache[instr] = result;
#endif
    
    return ARMDecodeStatus::FAILURE;
}
