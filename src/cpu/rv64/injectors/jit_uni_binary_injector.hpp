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
#ifndef CPU_RV64_INJECTORS_JIT_UNI_BINARY_INJECTOR_HPP
#define CPU_RV64_INJECTORS_JIT_UNI_BINARY_INJECTOR_HPP

#include "common/c_types_map.hpp"
#include "common/primitive_attr.hpp"

#include "cpu/rv64/injectors/injector_utils.hpp"
#include "cpu/rv64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

namespace binary_injector {

// How the right-hand-side (src1) operand is laid out relative to the
// accumulator lanes the host is currently holding.
//   scalar      - one value broadcast to every lane (per_tensor src1)
//   per_element - a contiguous run of values matching the active lanes; this
//                 covers a plain dst-shaped src1 and the common channels-last
//                 per-channel (per_oc) case where channel is the vector dim.
enum class broadcast_t { scalar, per_element };

// Caller-provided scratch. Two addressing modes:
//   direct   - the host owns rhs_addr and keeps it positioned at the element
//              matching the first active lane (the injector does not advance
//              it). Used by single-binary consumers (e.g. pooling) and supports
//              an optional vlse byte stride.
//   indirect - rhs_addr points to an array of per-binary base pointers
//              (post_ops_binary_rhs_arg_vec). Each binary injector loads its own
//              base from rhs_addr[arg_idx] and addresses lane data at base + off,
//              where off is the byte offset of the first active lane within the
//              unit (shared across the chain, advanced by the host). This is the
//              x64/aarch64-style scheme that lets one chain carry any number of
//              binaries. `gpr` is a scratch the injector clobbers per binary.
struct static_params_t {
    // Direct, contiguous (unit-stride) per-element rhs: loaded with vle.
    static_params_t(const Xbyak_riscv::VReg &v_rhs,
            const Xbyak_riscv::FReg &f_rhs, const Xbyak_riscv::Reg &rhs_addr)
        : v_rhs(v_rhs)
        , f_rhs(f_rhs)
        , rhs_addr(rhs_addr)
        , rhs_stride(rhs_addr) // unused when !strided
        , strided(false)
        , off(rhs_addr) // unused when !indirect
        , gpr(rhs_addr) // unused when !indirect
        , indirect(false) {}
    // Direct, strided per-element rhs: rhs_stride holds the byte stride between
    // consecutive lanes, so the injector loads with vlse. Used when the rhs
    // lanes are not contiguous in memory (e.g. channels-strided ncsp dst).
    static_params_t(const Xbyak_riscv::VReg &v_rhs,
            const Xbyak_riscv::FReg &f_rhs, const Xbyak_riscv::Reg &rhs_addr,
            const Xbyak_riscv::Reg &rhs_stride)
        : v_rhs(v_rhs)
        , f_rhs(f_rhs)
        , rhs_addr(rhs_addr)
        , rhs_stride(rhs_stride)
        , strided(true)
        , off(rhs_addr)
        , gpr(rhs_addr)
        , indirect(false) {}
    // Indirect: rhs_ptrs is the base of the per-binary pointer array; off is the
    // shared byte offset of the first active lane; gpr is a scratch register.
    // Per-element loads are contiguous vle from base + off.
    static_params_t(const Xbyak_riscv::VReg &v_rhs,
            const Xbyak_riscv::FReg &f_rhs, const Xbyak_riscv::Reg &rhs_ptrs,
            const Xbyak_riscv::Reg &off, const Xbyak_riscv::Reg &gpr)
        : v_rhs(v_rhs)
        , f_rhs(f_rhs)
        , rhs_addr(rhs_ptrs)
        , rhs_stride(rhs_ptrs) // unused when indirect (contiguous vle)
        , strided(false)
        , off(off)
        , gpr(gpr)
        , indirect(true) {}

    Xbyak_riscv::VReg v_rhs; // scratch to load a per-element rhs vector
    Xbyak_riscv::FReg f_rhs; // scratch to load a scalar rhs value
    Xbyak_riscv::Reg rhs_addr; // direct: rhs ptr; indirect: ptr-array base
    Xbyak_riscv::Reg rhs_stride; // byte stride for the strided (vlse) load
    bool strided; // false: contiguous vle; true: vlse with rhs_stride
    Xbyak_riscv::Reg off; // indirect: byte offset of the first active lane
    Xbyak_riscv::Reg gpr; // indirect: scratch clobbered per binary
    bool indirect; // false: rhs_addr is the data ptr; true: ptr-array base
};

// Which binary algorithms the in-kernel injector can emit. Arithmetic ops only;
// comparison ops (ge/gt/le/lt/eq/ne) produce a 0/1 result that needs a mask
// register and are left to a reference impl (the consumer pd rejects them).
bool is_alg_supported(alg_kind_t alg);

} // namespace binary_injector

// In-kernel binary post-op injector for RVV: applies `dst = dst OP rhs` to an
// accumulator register group in place, reading the rhs from a host-positioned
// address with the configured broadcast strategy.
template <cpu_isa_t isa>
struct jit_uni_binary_injector_t {
    using Vmm = typename jit_isa_traits_t<isa>::Vmm;

    // arg_idx selects this binary's base from the rhs pointer array in indirect
    // mode (its position among the chain's binary entries); ignored in direct
    // mode.
    jit_uni_binary_injector_t(jit_generator_t *host, alg_kind_t alg,
            binary_injector::broadcast_t bcast,
            const binary_injector::static_params_t &sp, int arg_idx = 0)
        : alg_(alg)
        , bcast_(bcast)
        , h_(host)
        , v_rhs_(sp.v_rhs)
        , f_rhs_(sp.f_rhs)
        , rhs_addr_(sp.rhs_addr)
        , rhs_stride_(sp.rhs_stride)
        , strided_(sp.strided)
        , off_(sp.off)
        , gpr_(sp.gpr)
        , indirect_(sp.indirect)
        , arg_idx_(arg_idx) {
        assert(binary_injector::is_alg_supported(alg_));
    }

    // Index-based interface (x64/aarch64 parity): applies dst = dst OP rhs to
    // the register group(s) by index, reading rhs from the host-positioned
    // address.
    void compute_vector(size_t idx) { compute_vector_range(idx, idx + 1); }
    void compute_vector_range(size_t start_idx, size_t end_idx);

private:
    void compute_body(const Vmm &dst);
    const alg_kind_t alg_;
    const binary_injector::broadcast_t bcast_;
    jit_generator_t *const h_;
    const Xbyak_riscv::VReg v_rhs_;
    const Xbyak_riscv::FReg f_rhs_;
    const Xbyak_riscv::Reg rhs_addr_;
    const Xbyak_riscv::Reg rhs_stride_;
    const bool strided_;
    const Xbyak_riscv::Reg off_;
    const Xbyak_riscv::Reg gpr_;
    const bool indirect_;
    const int arg_idx_;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_INJECTORS_JIT_UNI_BINARY_INJECTOR_HPP
