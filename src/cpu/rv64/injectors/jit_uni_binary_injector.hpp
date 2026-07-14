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

#include <map>
#include <memory>

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
//   per_element - a contiguous run of values matching the active lanes (a plain
//                 dst-shaped src1).
//   per_oc      - one value per output channel [1, C, 1, ...]. The dynamic path
//                 maps the output element offset to a channel index using C and
//                 oc_stride: channels-last/blocked (oc_stride==1) loads a channel
//                 vector; channels-first (oc_stride>1) broadcasts one channel
//                 value across the run.
enum class broadcast_t { scalar, per_element, per_oc };

// Caller-provided scratch + config. The rhs is addressed through an indirect
// per-binary pointer array (post_ops_binary_rhs_arg_vec): each binary injector
// loads its own base from rhs_ptrs[arg_idx], then the dynamic-params path
// computes the lane address from the per-register output offset (the caller
// hands it in at call time). `off` and `gpr` are address scratch the injector
// clobbers per binary; `strided`/`rhs_stride` request a vlse load for
// channels-strided (ncsp) rhs.
struct static_params_t {
    // Contiguous rhs: per-element vle / scalar flw.
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
    // Indirect + strided: like the indirect ctor but per-element lanes are
    // rhs_stride bytes apart (vlse from base + off). Used when the rhs lanes are
    // not contiguous in memory (e.g. a full-dst binary over a channels-strided
    // ncsp destination) while still carrying any number of binaries via the
    // per-binary pointer array.
    static_params_t(const Xbyak_riscv::VReg &v_rhs,
            const Xbyak_riscv::FReg &f_rhs, const Xbyak_riscv::Reg &rhs_ptrs,
            const Xbyak_riscv::Reg &off, const Xbyak_riscv::Reg &gpr,
            const Xbyak_riscv::Reg &rhs_stride)
        : v_rhs(v_rhs)
        , f_rhs(f_rhs)
        , rhs_addr(rhs_ptrs)
        , rhs_stride(rhs_stride)
        , strided(true)
        , off(off)
        , gpr(gpr)
        , indirect(true) {}

    Xbyak_riscv::VReg v_rhs; // scratch to load a per-element rhs vector
    // Second vector scratch for the per_oc gather (per-lane channel index).
    // Only read on the dynamic per_oc path; consumers that enable per_oc must
    // set it to a group distinct from v_rhs and the live accumulator/aux.
    // v0 is the reserved mask register and serves as an unset sentinel here.
    Xbyak_riscv::VReg v_idx = Xbyak_riscv::VReg(0);
    // Staging group for a narrow rhs (the narrow load lands here and is
    // widened into v_rhs; a widening op cannot overlap its source's low part, so
    // this must be distinct from v_rhs, v_idx and the live accumulator/aux).
    // Only read for f16/s8/u8; f32/s32 consumers may leave it default.
    // v0 is the reserved mask register and serves as an unset sentinel here.
    Xbyak_riscv::VReg v_tmp = Xbyak_riscv::VReg(0);
    Xbyak_riscv::FReg f_rhs; // scratch to load a scalar rhs value
    Xbyak_riscv::Reg rhs_addr; // direct: rhs ptr; indirect: ptr-array base
    Xbyak_riscv::Reg rhs_stride; // byte stride for the strided (vlse) load
    bool strided; // false: contiguous vle; true: vlse with rhs_stride
    // Address scratch for the dynamic output offset. off_is_bytes selects
    // whether that offset is already in bytes or must be scaled by rhs_dt.
    Xbyak_riscv::Reg off;
    Xbyak_riscv::Reg gpr; // indirect: scratch clobbered per binary
    bool indirect; // false: rhs_addr is the data ptr; true: ptr-array base
    // rhs data type; the rhs is loaded and converted from this dtype to f32.
    data_type_t rhs_dt = data_type::f32;
    // If true, the dynamic out-offset is already a byte offset (added straight to
    // the base) rather than an element offset the injector scales by the dtype
    // size. Indirect consumers that maintain a byte offset (pooling) set this to
    // avoid needing a second address-scratch register. Incompatible with per_oc.
    bool off_is_bytes = false;
    // broadcast strategy for the rhs (mirrors the `broadcast_t bcast` ctor arg).
    // per_oc covers the per-lane gather strategies (per_oc / per_oc_spatial /
    // per_w): the gathered index is ((out_off / oc_stride) % C) * blk +
    // (out_off % blk).
    //   per_oc/per_w plain: blk=1, C=dim size, oc_stride=that dim's stride.
    //   per_oc blocked (nChw8c): blk=block, C=C/block, oc_stride=outer stride.
    broadcast_t bcast = broadcast_t::per_element;
    dim_t C = 0; // outer count of the broadcast dim (per_oc: C/blk; per_w: W)
    dim_t oc_stride = 0; // outer stride of the broadcast dim
    dim_t blk = 1; // inner block of the broadcast dim (1 = non-blocked / per_w)
};

// Call-time parameters, mirroring x64/aarch64 rhs_arg_dynamic_params_t: each
// accumulator vmm is mapped to a register holding that register's output
// ELEMENT offset (index of its first active lane into the dst logical tensor).
// The injector maps that offset to the rhs slice address per broadcast strategy
// (no_broadcast/per_element, per_oc, scalar), so the address is computed inside
// the injector at call time rather than pre-positioned by the caller.
struct rhs_arg_dynamic_params_t {
    std::map<int, Xbyak_riscv::Reg> vmm_idx_to_out_off;
};

// Which binary algorithms the in-kernel injector can emit. Comparisons produce
// the oneDNN-required f32 0/1 result and use v0 as their temporary mask.
bool is_alg_supported(alg_kind_t alg);

} // namespace binary_injector

// In-kernel binary post-op injector for RVV: applies `dst = dst OP rhs` to an
// accumulator register group in place, loading the rhs base from the pointer
// array and deriving its address from the configured broadcast strategy.
template <cpu_isa_t isa>
struct jit_uni_binary_injector_t {
    using Vmm = typename jit_isa_traits_t<isa>::Vmm;

private:
    struct operand_t {
        operand_t(binary_injector::broadcast_t bcast,
                const binary_injector::static_params_t &sp, int arg_idx)
            : bcast(bcast)
            , v_rhs(sp.v_rhs)
            , v_idx(sp.v_idx)
            , v_tmp(sp.v_tmp)
            , f_rhs(sp.f_rhs)
            , rhs_addr(sp.rhs_addr)
            , rhs_stride(sp.rhs_stride)
            , strided(sp.strided)
            , off(sp.off)
            , gpr(sp.gpr)
            , indirect(sp.indirect)
            , arg_idx(arg_idx)
            , rhs_dt(sp.rhs_dt)
            , C(sp.C)
            , oc_stride(sp.oc_stride)
            , blk(sp.blk)
            , off_is_bytes(sp.off_is_bytes) {}

        binary_injector::broadcast_t bcast;
        Xbyak_riscv::VReg v_rhs, v_idx, v_tmp;
        Xbyak_riscv::FReg f_rhs;
        Xbyak_riscv::Reg rhs_addr, rhs_stride;
        bool strided;
        Xbyak_riscv::Reg off, gpr;
        bool indirect;
        int arg_idx;
        data_type_t rhs_dt;
        dim_t C, oc_stride, blk;
        bool off_is_bytes;
    };

public:
    // arg_idx selects this operand's base from the rhs pointer array. Ordinary
    // binary entries consume one slot; select consumes adjacent src1/src2 slots.
    jit_uni_binary_injector_t(jit_generator_t *host, alg_kind_t alg,
            binary_injector::broadcast_t bcast,
            const binary_injector::static_params_t &sp, int arg_idx = 0,
            binary_injector::broadcast_t select_bcast
            = binary_injector::broadcast_t::scalar,
            const binary_injector::static_params_t *select_sp = nullptr)
        : alg_(alg), h_(host), rhs_(bcast, sp, arg_idx) {
        if (select_sp)
            select_.reset(new operand_t(select_bcast, *select_sp, arg_idx + 1));
        assert(alg_ == alg_kind::binary_select
                        ? select_ != nullptr
                        : binary_injector::is_alg_supported(alg_));
    }

    // The caller passes the output element (or byte) offset per register; the
    // injector computes the rhs address from it and the broadcast strategy /
    // rhs dtype held in static_params (x64/aarch64 dynamic-params parity).
    void compute_vector(
            size_t idx, const binary_injector::rhs_arg_dynamic_params_t &dyn);
    void compute_vector_range(size_t start_idx, size_t end_idx,
            const binary_injector::rhs_arg_dynamic_params_t &dyn);

private:
    void apply_op(const Vmm &dst, bool scalar); // the op switch
    void apply_select(const Vmm &dst, const Xbyak_riscv::Reg &out_off);
    void materialize_cmp(const Vmm &dst); // v0 mask -> f32 0/1 in dst
    // Dynamic operand load into v_rhs/f_rhs (f32) from base + strategy(out_off);
    // returns whether the rhs is a broadcast scalar (f_rhs_) vs a vector (v_rhs_).
    bool load_operand(const Xbyak_riscv::Reg &out_off, operand_t &op);
    const alg_kind_t alg_;
    jit_generator_t *const h_;
    operand_t rhs_;
    std::unique_ptr<operand_t> select_;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_INJECTORS_JIT_UNI_BINARY_INJECTOR_HPP
