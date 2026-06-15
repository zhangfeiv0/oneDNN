/*******************************************************************************
* Copyright 2026 Institute of Software, Chinese Academy of Sciences
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/
#ifndef CPU_RV64_INJECTORS_INJECTOR_UTILS_HPP
#define CPU_RV64_INJECTORS_INJECTOR_UTILS_HPP

#include <cstddef>
#include <set>
#include <vector>
#include <initializer_list>

#include "common/utils.hpp"
#include "cpu/rv64/cpu_isa_traits.hpp"
#include "cpu/rv64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

// JIT register-type traits, the codegen companion to cpu_isa_traits.hpp (which
// stays free of the xbyak_riscv code-generator include for the intrinsics
// build). RVV has a single architectural vector-register type; the active
// element count is governed at run time by vsetvli rather than by the register
// type, so unlike x64/aarch64 there is no per-isa vector-width type. The traits
// are still keyed on isa for structural parity with x64/aarch64 and to leave
// room for future isa-specific register budgets.
template <cpu_isa_t isa>
struct jit_isa_traits_t {
    using Vmm = Xbyak_riscv::VReg;
    static constexpr int n_vregs = 32;
};

namespace injector_utils {

using vmm_index_set_t = typename std::set<size_t>;
using vmm_index_set_iterator_t = typename std::set<size_t>::iterator;

// Scope guard that preserves scalar (GPR) and floating-point (FReg) registers
// across a region by spilling them to the stack on construction and restoring
// them on destruction.
//
// NOTE: currently unused — every rv64 injector/kernel runs with only
// caller-saved registers and no preamble, so nothing constructs this yet. It is
// kept as scaffolding for x64/aarch64 parity and for future injectors that may
// need to spill; remove it if that need never materializes.
//
// Note: vector-register preservation is intentionally not provided. RVV vector
// length is a run-time quantity (VLEN), so a vector register cannot be spilled
// to a compile-time-sized stack slot without first reading vlenb and growing
// the frame dynamically. Injectors therefore take caller-provided *free* vector
// scratch through their static_params_t instead of preserving vectors here,
// which also avoids the v0 mask-register conflict on kernels whose accumulators
// occupy v0 (e.g. the brgemm kernel).
class register_preserve_guard_t {
public:
    register_preserve_guard_t(jit_generator_t *host,
            std::initializer_list<Xbyak_riscv::Reg> gpr_to_preserve,
            std::initializer_list<Xbyak_riscv::FReg> freg_to_preserve = {});
    register_preserve_guard_t(register_preserve_guard_t &&) = default;
    register_preserve_guard_t &operator=(register_preserve_guard_t &&)
            = default;
    DNNL_DISALLOW_COPY_AND_ASSIGN(register_preserve_guard_t);
    ~register_preserve_guard_t();

    // Number of stack bytes the guard currently occupies (16B-aligned).
    size_t stack_space_occupied() const { return stack_bytes_; }

private:
    jit_generator_t *host_;
    std::vector<Xbyak_riscv::Reg> gpr_regs_;
    std::vector<Xbyak_riscv::FReg> freg_regs_;
    size_t stack_bytes_;
};

} // namespace injector_utils
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_INJECTORS_INJECTOR_UTILS_HPP
