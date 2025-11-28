/*******************************************************************************
 * Copyright (c) 2007 MITSUNARI Shigeo
 * Copyright (C), 2023-2025, KNS Group LLC (YADRO)
 *
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * Neither the name of the copyright owner nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 *******************************************************************************/
 
#pragma once
namespace Xbyak_riscv {

// Control and Status Register
enum class CSR : uint32_t {
    // FP CSRs
    fflags = 0x001, // Floating-Point Accrued Exceptions
    frm    = 0x002, // Floating-Point Dynamic Rounding Mode
    fcsr   = 0x003, // Floating-Point Control and Status register
    // vector CSRs
    vstart = 0x008, // Vector start position
    vxsat  = 0x009, // Fixed-Point Saturate Flag
    vxrm   = 0x00A, // Fixed-Point Rounding Mode
    vcsr   = 0x00F, // Vector control and status register
    vl     = 0xC20, // Vector length
    vtype  = 0xC21, // Vector data type register
    vlenb  = 0xC22, // VLEN/8 (vector register length in bytes)
};


// Selected Element Width
enum class SEW : uint32_t {
    e8  = 0x0,
    e16 = 0x1,
    e32 = 0x2,
    e64 = 0x3
};

// Vector Length Multiplier
enum class LMUL : uint32_t {
    mf8 = 0x5,
    mf4 = 0x6,
    mf2 = 0x7,
    m1  = 0x0,
    m2  = 0x1,
    m4  = 0x2,
    m8  = 0x3
};

// Vector Mask Agnostic
enum class VMA : uint32_t {
    mu = 0, // undisturbed
    ma = 1, // agnostic
};

// Vector Tail Agnostic
enum class VTA : uint32_t {
    tu = 0, // undisturbed
    ta = 1, // agnostic
};

enum class VectorAddressingMode : uint32_t {
    unitStride       = 0x0,
    indexedUnordered = 0x1,
    strided          = 0x2,
    indexedOrdered   = 0x3
    // other encodings are reserved
};

enum class UnitStrideVectorAddressingModeLoad : uint32_t {
    load              = 0x0, // unit-stride load
    wholeRegisterLoad = 0x8, // unit-stride, whole register load
    maskLoad          = 0xb, // unit-stride, mask load, EEW=8
    faultOnlyFirst    = 0x10  // unit-stride fault-only-first
    // other encodings are reserved
};

enum class UnitStrideVectorAddressingModeStore : uint32_t {
    store              = 0x0, // unit-stride store
    wholeRegisterStore = 0x8, // unit-stride, whole register store
    maskStore          = 0xb  // unit-stride, mask store, EEW=8
    // other encodings are reserved
};

enum class WidthEncoding : uint32_t {
    e8  = 0x0, // Vector 8-bit  element
    e16 = 0x5, // Vector 16-bit element
    e32 = 0x6, // Vector 32-bit element
    e64 = 0x7, // Vector 64-bit element
};

enum class VM : uint32_t {
    unmasked = 1,
    masked = 0
};

enum class RM : uint32_t {
    rne = 0x0, // Round to Nearest, ties to Even
    rtz = 0x1, // Round towards Zero
    rdn = 0x2, // Round Down (towards -infinity)
    rup = 0x3, // Round Up (towards + infinity)
    rmm = 0x4, // Round to Nearest, ties to Max Magnitude
    dyn = 0x7  // In instruction’s rm field, selects dynamic rounding mode;
               // In Rounding Mode register, reserved.
};

enum class FFlags : uint32_t {
    NV = 0x01, // Invalid Operation
    DZ = 0x02, // Divide by Zero
    OF = 0x04, // Overflow
    UF = 0x08, // Underflow
    NX = 0x10  // Inexact
};

} // Xbyak_riscv
