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
#ifndef CPU_RV64_INJECTORS_JIT_UNI_ELTWISE_INJECTOR_HPP
#define CPU_RV64_INJECTORS_JIT_UNI_ELTWISE_INJECTOR_HPP

#include <vector>

#include "common/c_types_map.hpp"
#include "common/primitive_attr.hpp"

#include "cpu/rv64/injectors/injector_utils.hpp"
#include "cpu/rv64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

namespace eltwise_injector {

// Caller-provided scratch the eltwise injector may freely clobber inside
// compute_vector(). The host kernel guarantees these registers are dead during
// post-op application. RVV vector length is a run-time quantity, so the host
// (which owns the vsetvli state and the accumulator allocation) supplies free
// vector scratch rather than the injector spilling. The injector may use v0 as
// a mask register, so hosts must keep accumulator groups away from v0.
struct static_params_t {
    // Post-op / 3-aux form (all forward post-op consumers use this). v_aux3 and
    // v_aux4 default to v_aux2: only algorithms gated to the standalone eltwise
    // primitive (which supplies the wider form below) read them.
    static_params_t(const Xbyak_riscv::VReg &v_aux0,
            const Xbyak_riscv::VReg &v_aux1, const Xbyak_riscv::VReg &v_aux2,
            const Xbyak_riscv::FReg &f_aux0, const Xbyak_riscv::FReg &f_aux1,
            const Xbyak_riscv::Reg &gpr_aux0, bool is_fwd = true)
        : v_aux0(v_aux0)
        , v_aux1(v_aux1)
        , v_aux2(v_aux2)
        , v_aux3(v_aux2)
        , v_aux4(v_aux2)
        , f_aux0(f_aux0)
        , f_aux1(f_aux1)
        , gpr_aux0(gpr_aux0)
        , is_fwd(is_fwd) {}

    // Wide 5-aux form for the standalone eltwise primitive. The fourth group
    // lets gelu_erf forward keep its input across erf; backward algorithms such
    // as gelu_tanh use both the fourth and fifth groups. The backward injector is
    // standalone-only, so widening it does not affect any post-op consumer.
    static_params_t(const Xbyak_riscv::VReg &v_aux0,
            const Xbyak_riscv::VReg &v_aux1, const Xbyak_riscv::VReg &v_aux2,
            const Xbyak_riscv::VReg &v_aux3, const Xbyak_riscv::VReg &v_aux4,
            const Xbyak_riscv::FReg &f_aux0, const Xbyak_riscv::FReg &f_aux1,
            const Xbyak_riscv::Reg &gpr_aux0, bool is_fwd)
        : v_aux0(v_aux0)
        , v_aux1(v_aux1)
        , v_aux2(v_aux2)
        , v_aux3(v_aux3)
        , v_aux4(v_aux4)
        , f_aux0(f_aux0)
        , f_aux1(f_aux1)
        , gpr_aux0(gpr_aux0)
        , is_fwd(is_fwd) {}

    // Up to five vector scratch groups (same LMUL as the host accumulator).
    // Forward arithmetic algorithms use at most v_aux0; exp/logistic use
    // v_aux0 and v_aux2; tanh/elu/swish/gelu_tanh/log/soft_relu/mish use all
    // three; gelu_erf (fwd) additionally uses v_aux3, while some backward
    // algorithms use both v_aux3 and v_aux4.
    // Forward and backward may use v0 as a mask.
    Xbyak_riscv::VReg v_aux0, v_aux1, v_aux2, v_aux3, v_aux4;
    Xbyak_riscv::FReg f_aux0, f_aux1; // two FP scratch regs for constants
    Xbyak_riscv::Reg gpr_aux0; // one GPR scratch for constant materialization
    // Forward (d = alg(s)) or backward (ds = alg'(s)).
    bool is_fwd;
};

// Whether the JIT eltwise injector can emit this forward algorithm within the
// 3-aux post-op budget. Covers the arithmetic algorithms, the exp()-based
// transcendentals (exp/logistic/tanh/elu/swish/gelu_tanh), and mish/round.
// log/soft_relu/gelu_erf need a 4th aux (see needs_extra_aux). pow is not
// supported (like aarch64) and falls back to a reference impl.
bool is_alg_supported(alg_kind_t alg);

// Forward algorithms the standalone eltwise primitive accepts: everything in
// is_alg_supported() plus the ones that need a 4th aux (see needs_extra_aux),
// which the primitive supplies via the 5-aux static_params.
bool is_fwd_alg_supported(alg_kind_t alg);

// Forward algorithms outside the 3-aux budget: log/soft_relu/gelu_erf need a
// 4th vector aux (v_aux3). A post-op consumer may enable them only when it
// supplies that extra scratch (see the post_ops_ok n_vaux argument).
bool needs_extra_aux(alg_kind_t alg);

// Backward algorithms with an implemented derivative (src-based and
// use-dst-based). The standalone eltwise backward primitive supplies the 5-aux
// static_params these need.
bool is_bwd_alg_supported(alg_kind_t alg);

} // namespace eltwise_injector

// In-kernel forward eltwise post-op injector for RVV.
//
// compute_vector(v) applies `scale * alg(alpha, beta, v)` element-wise to the
// active lanes of vector register group `v`, in place. The host sets the vtype
// (SEW=e32 / data LMUL) and the active vl via vsetvli before calling; the
// injector emits only data-path instructions and never changes vl/vtype.
template <cpu_isa_t isa>
struct jit_uni_eltwise_injector_t {
    using Vmm = typename jit_isa_traits_t<isa>::Vmm;

    jit_uni_eltwise_injector_t(jit_generator_t *host, alg_kind_t alg,
            float alpha, float beta, float scale,
            const eltwise_injector::static_params_t &sp)
        : alg_(alg)
        , alpha_(alpha)
        , beta_(beta)
        , scale_(scale)
        , h_(host)
        , is_fwd_(sp.is_fwd)
        , v_aux0_(sp.v_aux0)
        , v_aux1_(sp.v_aux1)
        , v_aux2_(sp.v_aux2)
        , v_aux3_(sp.v_aux3)
        , v_aux4_(sp.v_aux4)
        , f_aux0_(sp.f_aux0)
        , f_aux1_(sp.f_aux1)
        , gpr_aux0_(sp.gpr_aux0) {
        assert(is_fwd_ ? eltwise_injector::is_fwd_alg_supported(alg_)
                       : eltwise_injector::is_bwd_alg_supported(alg_));
    }

    jit_uni_eltwise_injector_t(jit_generator_t *host,
            const post_ops_t::entry_t::eltwise_t &e,
            const eltwise_injector::static_params_t &sp)
        : jit_uni_eltwise_injector_t(
                  host, e.alg, e.alpha, e.beta, e.scale, sp) {}

    // ARM/x64-style index-based interface: apply the eltwise op to the vector
    // register group(s) identified by index. Forward: d = scale * alg(s).
    // Backward (is_fwd=false): transforms `s` into the derivative alg'(s); the
    // caller multiplies by diff_dst to get diff_src.
    void compute_vector(size_t idx) { compute_vector_range(idx, idx + 1); }
    void compute_vector_range(size_t start_idx, size_t end_idx);
    void compute_vector_range(const injector_utils::vmm_index_set_t &vmm_idxs);

private:
    // Worker: applies the op to one register group, dispatching on alg/is_fwd.
    void compute_body(const Vmm &v);
    void compute_vector_bwd(const Vmm &v);
    void load_f32_const(const Xbyak_riscv::FReg &f, float val);
    // NaN-preserving clamp(v, lo, hi); reuses f_aux0_/f_aux1_ and v0.
    void clamp(const Vmm &v, float lo, float hi);
    // exp(v) in place: range-reduce + Horner poly + 2^n scale. Uses v_aux0
    // and v_aux2; leaves v_aux1 free for callers. Building block for the
    // other transcendentals.
    void exp_compute_vector(const Vmm &v);
    // sigmoid(v) = 1 / (1 + exp(-v)), in place. Uses the exp building block.
    void logistic_compute_vector(const Vmm &v);
    // tanh(v) in place via 2*sigmoid(2v)-1 with a small-|v| linear blend. Uses
    // v_aux0/v_aux1/v_aux2 (the full 3-aux budget). Building block for
    // gelu_tanh and the tanh/gelu_tanh backward derivatives.
    void tanh_compute_vector(const Vmm &v);
    // log(v) in place: Cephes logf (frexp + degree-8 mantissa poly). Uses
    // v_aux0 and v_aux2 plus v0 as a mask; leaves v_aux1 free.
    void log_compute_vector(const Vmm &v);
    // erf(v) in place: Abramowitz-Stegun 7.1.26 (t-poly * exp(-v^2)). Uses
    // v_aux0/v_aux1/v_aux2 and v0; it leaves v_aux3/v_aux4 untouched for any
    // values the caller must preserve.
    void erf_compute_vector(const Vmm &v);

    const alg_kind_t alg_;
    const float alpha_;
    const float beta_;
    const float scale_;

    jit_generator_t *const h_;
    const bool is_fwd_;
    const Xbyak_riscv::VReg v_aux0_;
    const Xbyak_riscv::VReg v_aux1_;
    const Xbyak_riscv::VReg v_aux2_;
    const Xbyak_riscv::VReg v_aux3_;
    const Xbyak_riscv::VReg v_aux4_;
    const Xbyak_riscv::FReg f_aux0_;
    const Xbyak_riscv::FReg f_aux1_;
    const Xbyak_riscv::Reg gpr_aux0_;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_INJECTORS_JIT_UNI_ELTWISE_INJECTOR_HPP
