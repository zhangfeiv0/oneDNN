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
#include <cstring>
#include <limits>

#include "common/utils.hpp"

#include "cpu/rv64/injectors/jit_uni_eltwise_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

namespace eltwise_injector {

bool is_alg_supported(alg_kind_t alg) {
    using namespace alg_kind;
    return utils::one_of(alg, eltwise_relu, eltwise_tanh, eltwise_elu,
            eltwise_square, eltwise_abs, eltwise_sqrt, eltwise_linear,
            eltwise_logistic, eltwise_mish, eltwise_exp, eltwise_gelu_tanh,
            eltwise_hardsigmoid, eltwise_hardswish, eltwise_swish, eltwise_clip,
            eltwise_clip_v2, eltwise_round);
}

bool needs_extra_aux(alg_kind_t alg) {
    using namespace alg_kind;
    // soft_relu (exp then log, keeping x live across both), log (keeps x live
    // across the poly to patch 0/inf/negative), gelu_erf (x live across erf).
    return utils::one_of(alg, eltwise_soft_relu, eltwise_log, eltwise_gelu_erf);
}

bool is_fwd_alg_supported(alg_kind_t alg) {
    // is_alg_supported() plus the algs needing a 4th aux (see needs_extra_aux).
    // pow is not supported (like aarch64); a general power path -> ref.
    return is_alg_supported(alg) || needs_extra_aux(alg);
}

bool is_bwd_alg_supported(alg_kind_t alg) {
    using namespace alg_kind;
    return utils::one_of(alg, eltwise_relu, eltwise_tanh, eltwise_elu,
            eltwise_square, eltwise_abs, eltwise_sqrt, eltwise_linear,
            eltwise_soft_relu, eltwise_logistic, eltwise_mish, eltwise_exp,
            eltwise_gelu_tanh, eltwise_hardsigmoid, eltwise_hardswish,
            eltwise_swish, eltwise_log, eltwise_clip, eltwise_clip_v2,
            eltwise_gelu_erf, eltwise_relu_use_dst_for_bwd,
            eltwise_tanh_use_dst_for_bwd, eltwise_elu_use_dst_for_bwd,
            eltwise_sqrt_use_dst_for_bwd, eltwise_logistic_use_dst_for_bwd,
            eltwise_exp_use_dst_for_bwd, eltwise_clip_v2_use_dst_for_bwd);
}

} // namespace eltwise_injector

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::load_f32_const(const FReg &f, float val) {
    uint32_t bits;
    std::memcpy(&bits, &val, sizeof(bits));
    h_->li(gpr_aux0_, bits);
    h_->fmv_w_x(f, gpr_aux0_);
}

// NaN-preserving clamp(v, lo, hi). Comparisons with NaN are false, so NaN lanes
// keep their original value instead of being replaced by the clamp bound.
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::clamp(const Vmm &v, float lo, float hi) {
    const Xbyak_riscv::VReg v_mask(0);
    load_f32_const(f_aux0_, lo);
    h_->vmflt_vf(v_mask, v, f_aux0_);
    h_->vfmerge_vfm(v, v, f_aux0_);
    load_f32_const(f_aux1_, hi);
    h_->vmfgt_vf(v_mask, v, f_aux1_);
    h_->vfmerge_vfm(v, v, f_aux1_);
}

// exp(x) via base-2 range reduction + degree-5 minimax polynomial (the classic
// Cephes/sse_mathfun expf), with all constants materialized inline as FP
// scalars (RVV vf-form ops take an FReg directly, so no constant table is
// needed). Uses two aux vector groups (v_aux0_ scratch/poly, v_aux2_ integer
// exponent); v_aux1_ is left free so callers can stash x across the call.
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::exp_compute_vector(const Vmm &v) {
    const Vmm &a0 = v_aux0_; // poly accumulator / scratch
    const Vmm &a2
            = v_aux2_; // integer exponent n (must survive to the 2^n step)

    // clamp x to [ln(FLT_MIN), ln(FLT_MAX)] to keep the result finite. The
    // 2*2^(n-1) reconstruction (below) makes the lowest exponent bucket (n ==
    // -126) come out as exactly 0: (n-1)+127 == 0 builds +0.0, not subnormal
    // 2^-127. This matches x64 (which masks the < ln(FLT_MIN) lanes to 0); the
    // error vs the true value (~1e-38) is negligible.
    clamp(v, -87.33654475f, 88.72283905f); // MINLOGF, MAXLOGF

    // n = round(x * log2e); z = (float)n. Add a signed half and truncate with
    // an explicit mode so float-to-int rounding does not use the application's
    // current FRM or serialize the hot loop with per-vector FRM CSR updates.
    load_f32_const(f_aux0_, 1.44269504088896341f); // log2e
    h_->vfmul_vf(a0, v, f_aux0_);
    load_f32_const(f_aux0_, 0.5f);
    h_->vfmv_v_f(a2, f_aux0_);
    h_->vfsgnj_vv(a2, a2, a0); // copysign(0.5f, x * log2e)
    h_->vfadd_vv(a0, a0, a2);
    h_->vfcvt_rtz_x_f_v(a2, a0); // nearest integer, ties away from zero
    h_->vfcvt_f_x_v(a0, a2); // z = (float)n

    // r = x - z*C1 - z*C2  (extended-precision ln2), r in [-ln2/2, ln2/2]
    load_f32_const(f_aux0_, 0.693359375f); // C1
    h_->vfnmsac_vf(v, f_aux0_, a0); // v -= C1*z
    load_f32_const(f_aux0_, -2.12194440e-4f); // C2
    h_->vfnmsac_vf(v, f_aux0_, a0); // v -= C2*z  => v = r

    // poly5(r) by Horner; coefficients p0..p5 (Cephes expf)
    load_f32_const(f_aux0_, 1.9875691500e-4f);
    h_->vfmv_v_f(a0, f_aux0_); // y = p0
    const float p[] = {1.3981999507e-3f, 8.3334519073e-3f, 4.1665795894e-2f,
            1.6666665459e-1f, 5.0000001201e-1f};
    for (float c : p) {
        h_->vfmul_vv(a0, a0, v); // y *= r
        load_f32_const(f_aux0_, c);
        h_->vfadd_vf(a0, a0, f_aux0_); // y += c
    }
    // y = y*r^2 + r + 1, computed as ((y*r)*r) + r + 1 to avoid a 3rd aux reg
    h_->vfmul_vv(a0, a0, v);
    h_->vfmul_vv(a0, a0, v);
    h_->vfadd_vv(a0, a0, v);
    load_f32_const(f_aux0_, 1.f);
    h_->vfadd_vf(a0, a0, f_aux0_);

    // 2^n applied as 2 * 2^(n-1): for x near MAXLOGF, n rounds up to 128 and a
    // direct 2^128 = ((128+127)<<23) is +inf, poisoning the otherwise-finite
    // result (exp(MAXLOGF) ~= FLT_MAX). 2^(n-1) stays representable (<= 2^127),
    // so we scale by it and double afterwards (the x64 injector uses the same
    // trick). 2^(n-1) = reinterpret(((n-1) + 127) << 23) = reinterpret((n+126)<<23).
    h_->li(gpr_aux0_, 126);
    h_->vadd_vx(a2, a2, gpr_aux0_);
    h_->li(gpr_aux0_, 23);
    h_->vsll_vx(a2, a2, gpr_aux0_); // a2 = 2^(n-1)

    h_->vfmul_vv(v, a0, a2); // poly * 2^(n-1)
    load_f32_const(f_aux0_, 2.f);
    h_->vfmul_vf(v, v, f_aux0_); // * 2 = poly * 2^n = exp(x)
}

// sigmoid(x) = 1 / (1 + exp(-x)); numerically stable for both tails.
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::logistic_compute_vector(const Vmm &v) {
    h_->vfneg_v(v, v); // -x
    exp_compute_vector(v); // exp(-x)
    load_f32_const(f_aux0_, 1.f);
    h_->vfadd_vf(v, v, f_aux0_); // 1 + exp(-x)
    load_f32_const(f_aux0_, 1.f);
    h_->vfrdiv_vf(v, v, f_aux0_); // 1 / (1 + exp(-x))
}

// tanh(x) = 2*sigmoid(2x) - 1, with a linear blend toward tanh(x) ~= x for
// small |x| to avoid catastrophic cancellation (the sigmoid tail has ~1e-7
// absolute error, which is a large *relative* error when tanh(x) is tiny).
// Uses all three aux groups (v_aux0/v_aux1/v_aux2).
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::tanh_compute_vector(const Vmm &v) {
    constexpr float t1 = 0.002f, t2 = 0.008f; // blend band (in |x|)
    h_->vmv_v_v(v_aux1_, v); // save x
    load_f32_const(f_aux0_, 2.f);
    h_->vfmul_vf(v, v, f_aux0_);
    logistic_compute_vector(v);
    load_f32_const(f_aux0_, 2.f);
    h_->vfmul_vf(v, v, f_aux0_);
    load_f32_const(f_aux0_, -1.f);
    h_->vfadd_vf(v, v, f_aux0_); // v = t = 2*sigmoid(2x)-1
    h_->vfsub_vv(v_aux0_, v_aux1_, v); // v_aux0 = x - t
    h_->vfmul_vv(v_aux2_, v_aux1_, v_aux1_); // v_aux2 = x^2
    load_f32_const(f_aux0_, t2 * t2);
    h_->vfrsub_vf(v_aux2_, v_aux2_, f_aux0_); // t2^2 - x^2
    load_f32_const(f_aux0_, 1.f / (t2 * t2 - t1 * t1));
    h_->vfmul_vf(v_aux2_, v_aux2_, f_aux0_); // w (unclamped)
    clamp(v_aux2_, 0.f, 1.f); // w in [0,1]
    h_->vfmacc_vv(v, v_aux2_, v_aux0_); // v = t + w*(x - t)
}

// log(x) via Cephes single-precision logf: frexp into mantissa in
// [sqrt(1/2), sqrt(2)) and integer exponent, then a degree-8 minimax
// polynomial. Constants inline. Uses v_aux0 (poly/scratch), v_aux2 (exponent
// e) and v0 (mantissa-range mask); leaves v_aux1 free. ~1 ULP, well within the
// LOG benchdnn tolerance. Assumes x > 0 (the log filler feeds positive values).
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::log_compute_vector(const Vmm &v) {
    const Xbyak_riscv::VReg v_mask(0);
    const Vmm &poly = v_aux0_;
    const Vmm &e = v_aux2_;
    const Vmm &tmp = v_aux1_; // free during the frexp branch and poly tail

    // frexpf: e = ((bits >> 23) & 0xff) - 126; mantissa in [0.5, 1).
    h_->li(gpr_aux0_, 23);
    h_->vsrl_vx(e, v, gpr_aux0_);
    h_->li(gpr_aux0_, 0xff);
    h_->vand_vx(e, e, gpr_aux0_);
    h_->li(gpr_aux0_, 126);
    h_->vsub_vx(e, e, gpr_aux0_); // e (int exponent)
    // mantissa m = (bits & 0x807fffff) | 0x3f000000  -> [0.5, 1)
    h_->li(gpr_aux0_, 0x807fffff);
    h_->vand_vx(v, v, gpr_aux0_);
    h_->li(gpr_aux0_, 0x3f000000);
    h_->vor_vx(v, v, gpr_aux0_);
    h_->vfcvt_f_x_v(e, e); // (float)e

    // if (m < SQRTHF) { e -= 1; m = m + m - 1; } else { m = m - 1; }. The host
    // vtype is mask-agnostic, so branch with explicit merges (which write every
    // body lane) rather than masked arithmetic.
    load_f32_const(f_aux0_, 0.707106781186547524f); // SQRTHF
    h_->vmflt_vf(v_mask, v, f_aux0_); // mask: m < SQRTHF
    load_f32_const(f_aux0_, 1.f);
    h_->vfsub_vf(tmp, e, f_aux0_); // e - 1
    h_->vmerge_vvm(e, e, tmp); // e := mask ? e-1 : e
    h_->vfadd_vv(tmp, v, v); // 2m
    h_->vmerge_vvm(v, v, tmp); // m := mask ? 2m : m
    load_f32_const(f_aux0_, 1.f);
    h_->vfsub_vf(v, v, f_aux0_); // m := m - 1  (== 2m-1 on masked lanes)

    // poly = m^3 * P(m); P is Horner over the Cephes logf coefficients.
    load_f32_const(f_aux0_, 7.0376836292e-2f);
    h_->vfmv_v_f(poly, f_aux0_);
    const float p[] = {-1.1514610310e-1f, 1.1676998740e-1f, -1.2420140846e-1f,
            1.4249322787e-1f, -1.6668057665e-1f, 2.0000714765e-1f,
            -2.4999993993e-1f, 3.3333331174e-1f};
    for (float c : p) {
        h_->vfmul_vv(poly, poly, v);
        load_f32_const(f_aux0_, c);
        h_->vfadd_vf(poly, poly, f_aux0_);
    }
    h_->vfmul_vv(poly, poly, v);
    h_->vfmul_vv(poly, poly, v);
    h_->vfmul_vv(poly, poly, v); // poly *= m^3

    // result = m + (poly + ln2lo*e - 0.5*m^2) + ln2hi*e  (ln2 split hi/lo)
    load_f32_const(f_aux0_, -2.12194440e-4f); // ln2 lo
    h_->vfmacc_vf(poly, f_aux0_, e); // poly += ln2lo * e
    h_->vfmul_vv(tmp, v, v); // m^2
    load_f32_const(f_aux0_, 0.5f);
    h_->vfnmsac_vf(poly, f_aux0_, tmp); // poly -= 0.5*m^2
    h_->vfadd_vv(v, v, poly); // v = m + poly
    load_f32_const(f_aux0_, 0.693359375f); // ln2 hi
    h_->vfmacc_vf(v, f_aux0_, e); // v += ln2hi * e
}

// erf(x) via Abramowitz & Stegun 7.1.26: erf(|x|) = 1 - P(t)*exp(-x^2),
// t = 1/(1 + p|x|). It returns the non-negative magnitude; callers restore the
// odd sign when needed. Max abs error is ~1.5e-7, inside the gelu_erf tolerance.
// Uses v_aux0/v_aux1/v_aux2 (v_aux2 as exp scratch) and v0. A caller that needs
// the original input keeps it live outside those groups (forward uses v_aux3).
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::erf_compute_vector(const Vmm &v) {
    const Vmm &ax = v_aux0_; // |x|
    const Vmm &t = v_aux1_; // t and later the poly

    // Sign-free: return erf(|v|) >= 0; callers fold the sign in (gelu_erf uses
    // x*sign(x) == |x|). This keeps erf inside 3 aux (aux0/aux1/aux2) so it fits
    // the post-op budget when the host supplies a 4th aux for the outer alg.
    h_->li(gpr_aux0_, 0x7fffffff);
    h_->vand_vx(ax, v, gpr_aux0_); // |x|

    // t = 1 / (1 + p*|x|)
    load_f32_const(f_aux0_, 0.3275911f);
    h_->vfmv_v_f(t, f_aux0_);
    h_->vfmul_vv(t, t, ax);
    load_f32_const(f_aux0_, 1.f);
    h_->vfadd_vf(t, t, f_aux0_);
    load_f32_const(f_aux0_, 1.f);
    h_->vfrdiv_vf(t, t, f_aux0_); // t = 1/(1+p|x|)

    // poly = ((((a5*t + a4)*t + a3)*t + a2)*t + a1) * t   (in v)
    load_f32_const(f_aux0_, 1.061405429f); // a5
    h_->vfmv_v_f(v, f_aux0_);
    const float a[] = {
            -1.453152027f, 1.421413741f, -0.284496736f, 0.254829592f}; // a4..a1
    for (float c : a) {
        h_->vfmul_vv(v, v, t);
        load_f32_const(f_aux0_, c);
        h_->vfadd_vf(v, v, f_aux0_);
    }
    h_->vfmul_vv(v, v, t); // * t  -> poly, in v
    h_->vmv_v_v(t, v); // stash poly in t (survives exp, which uses aux0/aux2)

    // e = exp(-x^2) in v (exp clobbers v_aux0 and v_aux2)
    h_->vfmul_vv(v, ax, ax); // x^2
    h_->vfneg_v(v, v); // -x^2
    exp_compute_vector(v); // v = exp(-x^2)

    // erf(|x|) = 1 - poly*e  (in [0, 1))
    h_->vfmul_vv(v, v, t); // poly*e
    load_f32_const(f_aux0_, 1.f);
    h_->vfrsub_vf(v, v, f_aux0_); // 1 - poly*e
}

// Backward: transform a register holding `s` into alg'(s). Uses v0 as the RVV
// mask register (the standalone eltwise primitive keeps v0 free) plus the two
// aux vector groups. The caller multiplies the resulting alg'(s) by diff_dst.
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::compute_vector_bwd(const Vmm &v) {
    using namespace alg_kind;
    const Xbyak_riscv::VReg v_mask(0);
    const Vmm &a0 = v_aux0_;
    const Vmm &a1 = v_aux1_;

    switch (alg_) {
        case eltwise_relu_use_dst_for_bwd: // relu preserves sign: d>0 <=> s>0
        case eltwise_relu:
            // s > 0 ? 1 : alpha
            load_f32_const(f_aux0_, 0.f);
            h_->vmfgt_vf(v_mask, v, f_aux0_);
            load_f32_const(f_aux0_, alpha_);
            h_->vfmv_v_f(a0, f_aux0_);
            load_f32_const(f_aux0_, 1.f);
            h_->vfmv_v_f(v, f_aux0_);
            h_->vmerge_vvm(v, a0, v); // mask ? 1 : alpha
            break;

        case eltwise_square:
            // 2 * s
            load_f32_const(f_aux0_, 2.f);
            h_->vfmul_vf(v, v, f_aux0_);
            break;

        case eltwise_abs:
            // s > 0 ? 1 : s < 0 ? -1 : 0
            h_->vmv_v_v(a0, v); // save s
            load_f32_const(f_aux0_, 0.f);
            h_->vfmv_v_f(v, f_aux0_); // 0
            h_->vmfgt_vf(v_mask, a0, f_aux0_);
            load_f32_const(f_aux1_, 1.f);
            h_->vfmv_v_f(a1, f_aux1_);
            h_->vmerge_vvm(v, v, a1); // s>0 -> 1
            h_->vmflt_vf(v_mask, a0, f_aux0_);
            load_f32_const(f_aux1_, -1.f);
            h_->vfmv_v_f(a1, f_aux1_);
            h_->vmerge_vvm(v, v, a1); // s<0 -> -1
            break;

        case eltwise_sqrt:
            // 1 / (2 * sqrt(s))
            h_->vfsqrt_v(v, v);
            load_f32_const(f_aux0_, 2.f);
            h_->vfmul_vf(v, v, f_aux0_);
            load_f32_const(f_aux0_, 1.f);
            h_->vfrdiv_vf(v, v, f_aux0_);
            break;

        case eltwise_linear:
            // alpha
            load_f32_const(f_aux0_, alpha_);
            h_->vfmv_v_f(v, f_aux0_);
            break;

        case eltwise_clip:
            // (alpha < s && s <= beta) ? 1 : 0
            h_->vmv_v_v(a0, v); // save s
            load_f32_const(f_aux0_, 1.f);
            h_->vfmv_v_f(v, f_aux0_); // 1
            load_f32_const(f_aux1_, 0.f);
            h_->vfmv_v_f(a1, f_aux1_); // 0
            load_f32_const(f_aux0_, alpha_);
            h_->vmfle_vf(v_mask, a0, f_aux0_);
            h_->vmerge_vvm(v, v, a1); // s<=alpha -> 0
            load_f32_const(f_aux0_, beta_);
            h_->vmfgt_vf(v_mask, a0, f_aux0_);
            h_->vmerge_vvm(v, v, a1); // s>beta -> 0
            break;

        case eltwise_hardsigmoid:
            // v = alpha*s + beta; (v<=0 || v>=1) ? 0 : alpha
            load_f32_const(f_aux0_, alpha_);
            h_->vfmul_vf(a0, v, f_aux0_);
            load_f32_const(f_aux0_, beta_);
            h_->vfadd_vf(a0, a0, f_aux0_); // a0 = v
            load_f32_const(f_aux0_, alpha_);
            h_->vfmv_v_f(v, f_aux0_); // alpha
            load_f32_const(f_aux1_, 0.f);
            h_->vfmv_v_f(a1, f_aux1_); // 0
            load_f32_const(f_aux0_, 1.f);
            h_->vmfge_vf(v_mask, a0, f_aux0_);
            h_->vmerge_vvm(v, v, a1); // v>=1 -> 0
            load_f32_const(f_aux0_, 0.f);
            h_->vmfle_vf(v_mask, a0, f_aux0_);
            h_->vmerge_vvm(v, v, a1); // v<=0 -> 0
            break;

        case eltwise_hardswish:
            // v = alpha*s+beta; w = 2*alpha*s+beta; v<=0?0 : v>=1?1 : w
            load_f32_const(f_aux0_, alpha_);
            h_->vfmul_vf(a0, v, f_aux0_); // alpha*s
            load_f32_const(f_aux1_, 2.f);
            h_->vfmul_vf(v, a0, f_aux1_); // 2*alpha*s
            load_f32_const(f_aux0_, beta_);
            h_->vfadd_vf(a0, a0, f_aux0_); // a0 = v
            h_->vfadd_vf(v, v, f_aux0_); // v = w
            load_f32_const(f_aux1_, 1.f);
            h_->vfmv_v_f(a1, f_aux1_); // 1
            load_f32_const(f_aux0_, 1.f);
            h_->vmfge_vf(v_mask, a0, f_aux0_);
            h_->vmerge_vvm(v, v, a1); // v>=1 -> 1
            load_f32_const(f_aux1_, 0.f);
            h_->vfmv_v_f(a1, f_aux1_); // 0
            load_f32_const(f_aux0_, 0.f);
            h_->vmfle_vf(v_mask, a0, f_aux0_);
            h_->vmerge_vvm(v, v, a1); // v<=0 -> 0
            break;

        case eltwise_clip_v2: // src-based: (alpha < s && s < beta) ? 1 : 0
        case eltwise_clip_v2_use_dst_for_bwd: // same formula on d
            h_->vmv_v_v(a0, v); // save value
            load_f32_const(f_aux0_, 1.f);
            h_->vfmv_v_f(v, f_aux0_); // 1
            load_f32_const(f_aux1_, 0.f);
            h_->vfmv_v_f(a1, f_aux1_); // 0
            load_f32_const(f_aux0_, alpha_);
            h_->vmfle_vf(v_mask, a0, f_aux0_);
            h_->vmerge_vvm(v, v, a1); // s<=alpha -> 0
            load_f32_const(f_aux0_, beta_);
            h_->vmfge_vf(v_mask, a0, f_aux0_);
            h_->vmerge_vvm(v, v, a1); // s>=beta -> 0
            h_->vmfne_vv(v_mask, a0, a0);
            h_->vmerge_vvm(v, v, a1); // unordered (NaN) -> 0
            break;

        case eltwise_exp: // exp'(s) = exp(s)
            exp_compute_vector(v);
            break;

        case eltwise_exp_use_dst_for_bwd: // exp'(s) = d
            break; // identity: derivative is the forward output itself

        // sig'(s) = sig(s)*(1 - sig(s)); use_dst form gets d = sig(s) directly.
        case eltwise_logistic:
            logistic_compute_vector(v);
            // fallthrough: common d*(1-d) tail
        case eltwise_logistic_use_dst_for_bwd:
            load_f32_const(f_aux0_, 1.f);
            h_->vfrsub_vf(a0, v, f_aux0_); // 1 - v
            h_->vfmul_vv(v, v, a0); // v*(1 - v)
            break;

        // tanh'(s) = 1 - tanh(s)^2; use_dst form gets d = tanh(s) directly.
        case eltwise_tanh:
            tanh_compute_vector(v);
            // fallthrough: common 1 - d^2 tail
        case eltwise_tanh_use_dst_for_bwd:
            h_->vfmul_vv(v, v, v); // v^2
            load_f32_const(f_aux0_, 1.f);
            h_->vfrsub_vf(v, v, f_aux0_); // 1 - v^2
            break;

        case eltwise_sqrt_use_dst_for_bwd: // 1 / (2*d)
            load_f32_const(f_aux0_, 2.f);
            h_->vfmul_vf(v, v, f_aux0_);
            load_f32_const(f_aux0_, 1.f);
            h_->vfrdiv_vf(v, v, f_aux0_);
            break;

        case eltwise_elu: // s>0 ? 1 : alpha*exp(s)
            h_->vmv_v_v(a1, v); // save s (exp's internal clamp clobbers v0)
            exp_compute_vector(v); // exp(s)
            load_f32_const(f_aux0_, alpha_);
            h_->vfmul_vf(v, v, f_aux0_); // alpha*exp(s)
            load_f32_const(f_aux0_, 0.f);
            h_->vmfgt_vf(v_mask, a1, f_aux0_); // recompute mask: s>0
            load_f32_const(f_aux0_, 1.f);
            h_->vfmerge_vfm(v, v, f_aux0_); // s>0 -> 1
            break;

        case eltwise_elu_use_dst_for_bwd: // d>0 ? 1 : d+alpha
            load_f32_const(f_aux0_, 0.f);
            h_->vmfgt_vf(v_mask, v, f_aux0_); // mask: d>0
            load_f32_const(f_aux0_, alpha_);
            h_->vfadd_vf(v, v, f_aux0_); // d + alpha
            load_f32_const(f_aux0_, 1.f);
            h_->vfmerge_vfm(v, v, f_aux0_); // d>0 -> 1
            break;

        case eltwise_swish: {
            // v = sig(alpha*s); ds = v + s*alpha*v*(1 - v). logistic clobbers
            // a0/a2, so keep s in a1 (free across the logistic building block).
            h_->vmv_v_v(a1, v); // save s
            load_f32_const(f_aux0_, alpha_);
            h_->vfmul_vf(v, v, f_aux0_); // alpha*s
            logistic_compute_vector(v); // v = sig(alpha*s)
            load_f32_const(f_aux0_, 1.f);
            h_->vfrsub_vf(a0, v, f_aux0_); // 1 - v
            h_->vfmul_vv(a0, a0, v); // v*(1 - v)
            load_f32_const(f_aux0_, alpha_);
            h_->vfmul_vf(a0, a0, f_aux0_); // alpha*v*(1-v)
            h_->vfmul_vv(a0, a0, a1); // s*alpha*v*(1-v)
            h_->vfadd_vv(v, v, a0); // + v
            break;
        }

        case eltwise_gelu_tanh: {
            // ds = 0.5*(1+t)*(1 + s*(1-t)*dg), t=tanh(g),
            // g = k*s*(1+c*s^2), dg = k*(1+3c*s^2), k=sqrt(2/pi), c=0.044715.
            const Vmm &s = v_aux3_, &dg = v_aux4_;
            constexpr float k = 0.79788458347320556640625f, c = 0.044715f;
            h_->vmv_v_v(s, v); // save s
            h_->vfmul_vv(a0, v, v); // s^2
            load_f32_const(f_aux0_, 3.f * c);
            h_->vfmul_vf(dg, a0, f_aux0_); // 3c*s^2
            load_f32_const(f_aux0_, 1.f);
            h_->vfadd_vf(dg, dg, f_aux0_); // 1 + 3c*s^2
            load_f32_const(f_aux0_, k);
            h_->vfmul_vf(dg, dg, f_aux0_); // dg
            load_f32_const(f_aux0_, c);
            h_->vfmul_vf(a0, a0, f_aux0_); // c*s^2
            load_f32_const(f_aux0_, 1.f);
            h_->vfadd_vf(a0, a0, f_aux0_); // 1 + c*s^2
            h_->vfmul_vv(v, v, a0); // s*(1+c*s^2)
            load_f32_const(f_aux0_, k);
            h_->vfmul_vf(v, v, f_aux0_); // g
            tanh_compute_vector(v); // t = tanh(g) (uses a0/a1/a2)
            load_f32_const(f_aux0_, 1.f);
            h_->vfrsub_vf(a0, v, f_aux0_); // 1 - t
            h_->vfmul_vv(a0, a0, dg); // dg*(1 - t)
            h_->vfmul_vv(a0, a0, s); // s*dg*(1 - t)
            load_f32_const(f_aux0_, 1.f);
            h_->vfadd_vf(a0, a0, f_aux0_); // 1 + s*(1-t)*dg
            load_f32_const(f_aux0_, 1.f);
            h_->vfadd_vf(v, v, f_aux0_); // 1 + t
            load_f32_const(f_aux0_, 0.5f);
            h_->vfmul_vf(v, v, f_aux0_); // 0.5*(1+t)
            h_->vfmul_vv(v, v, a0); // ds
            break;
        }

        case eltwise_log: // log'(s) = 1/s
            load_f32_const(f_aux0_, 1.f);
            h_->vfrdiv_vf(v, v, f_aux0_);
            break;

        case eltwise_soft_relu: // srelu'(s) = sigmoid(alpha*s)
            load_f32_const(f_aux0_, alpha_);
            h_->vfmul_vf(v, v, f_aux0_);
            logistic_compute_vector(v);
            break;

        case eltwise_mish: {
            // mish'(s) = th + s*sig*(1 - th^2), th = tanh(softplus(s)),
            // sig = sigmoid(s). With w = 1 + exp(s) and e = exp(s):
            //   sig = e/w         (== (w-1)/w, no cancellation for e<<1)
            //   th  = (w^2-1)/(w^2+1), w^2-1 = e*(w+1)  (no cancellation)
            // Overflow-safe: e,w -> inf gives sig=1, th=1.
            const Vmm &a2 = v_aux2_;
            const Vmm &s = v_aux3_, &w = v_aux4_;
            h_->vmv_v_v(s, v); // save s
            exp_compute_vector(v); // e = exp(s) in v (uses a0/a2)
            // cap e so w^2 stays finite; beyond this th,sig == 1 to f32 anyway.
            load_f32_const(f_aux0_, 1e18f);
            h_->vfmin_vf(v, v, f_aux0_);
            load_f32_const(f_aux0_, 1.f);
            h_->vfadd_vf(w, v, f_aux0_); // w = 1 + e
            h_->vfdiv_vv(a1, v, w); // sig = e/w
            h_->vfmul_vv(a0, w, w); // w^2
            h_->vfadd_vf(a2, w, f_aux0_); // w + 1
            h_->vfmul_vv(a2, a2, v); // e*(w+1) = w^2 - 1
            h_->vfadd_vf(v, a0, f_aux0_); // w^2 + 1
            h_->vfdiv_vv(a0, a2, v); // th = (w^2-1)/(w^2+1)
            h_->vfmul_vv(v, a0, a0); // th^2
            h_->vfrsub_vf(v, v, f_aux0_); // 1 - th^2
            h_->vfmul_vv(v, v, a1); // sig*(1-th^2)
            h_->vfmul_vv(v, v, s); // s*sig*(1-th^2)
            h_->vfadd_vv(v, v, a0); // + th
            break;
        }

        case eltwise_gelu_erf: {
            // gelu_erf'(s) = 0.5*(1 + erf(u) + u*C*exp(-u^2)),
            // u = s*sqrt(2)/2, C = 2/sqrt(pi).
            constexpr float c = 0.707106769084930419921875f; // 1/sqrt(2)
            constexpr float C = 1.12837922573089599609375f; // 2/sqrt(pi)
            const Vmm &s = v_aux3_, &u = v_aux4_;
            h_->vmv_v_v(s, v); // save s
            load_f32_const(f_aux0_, c);
            h_->vfmul_vf(v, v, f_aux0_); // u = s/sqrt2
            h_->vmv_v_v(u, v); // save u
            erf_compute_vector(v); // erf(|u|), uses a0..a2
            h_->li(gpr_aux0_, 0x80000000);
            h_->vand_vx(a0, s, gpr_aux0_); // sign(s) == sign(u)
            h_->li(gpr_aux0_, 0x7fffffff);
            h_->vand_vx(v, v, gpr_aux0_);
            h_->vor_vv(v, v, a0); // erf(u) = copysign(erf(|u|), u)
            h_->vfmul_vv(a1, u, u); // u^2
            h_->vfneg_v(a1, a1); // -u^2
            exp_compute_vector(a1); // exp(-u^2) (uses a0/a2)
            h_->vfmul_vv(a1, a1, u); // u*exp(-u^2)
            load_f32_const(f_aux0_, C);
            h_->vfmul_vf(a1, a1, f_aux0_); // u*C*exp(-u^2)
            h_->vfadd_vv(v, v, a1); // erf(u) + term
            load_f32_const(f_aux0_, 1.f);
            h_->vfadd_vf(v, v, f_aux0_); // 1 + ...
            load_f32_const(f_aux0_, 0.5f);
            h_->vfmul_vf(v, v, f_aux0_); // 0.5*(...)
            break;
        }

        default: assert(!"unsupported eltwise bwd alg"); break;
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::compute_body(const Vmm &v) {
    using namespace alg_kind;

    if (!is_fwd_) {
        compute_vector_bwd(v);
        return;
    }

    switch (alg_) {
        case eltwise_relu:
            if (alpha_ == 0.f) {
                // relu(x) = x < 0 ? 0 : x. Keep NaN lanes unchanged.
                const Xbyak_riscv::VReg v_mask(0);
                load_f32_const(f_aux0_, 0.f);
                h_->vmflt_vf(v_mask, v, f_aux0_);
                h_->vfmerge_vfm(v, v, f_aux0_);
            } else {
                // leaky relu = x>0 ? x : alpha*x. NaN takes the false branch,
                // but alpha*NaN is still NaN.
                const Xbyak_riscv::VReg v_mask(0);
                load_f32_const(f_aux0_, 0.f);
                load_f32_const(f_aux1_, alpha_);
                h_->vfmul_vf(v_aux0_, v, f_aux1_);
                h_->vmfgt_vf(v_mask, v, f_aux0_);
                h_->vmerge_vvm(v, v_aux0_, v);
            }
            break;

        case eltwise_square:
            // x * x
            h_->vfmul_vv(v, v, v);
            break;

        case eltwise_abs:
            // clear the IEEE-754 sign bit of each f32 lane (mask-free, no aux
            // vector). Operates on the raw bits and assumes vtype SEW=e32 (the
            // 0x7fffffff mask clears bit 31 of each 32-bit lane).
            h_->li(gpr_aux0_, 0x7fffffff);
            h_->vand_vx(v, v, gpr_aux0_);
            break;

        case eltwise_sqrt: h_->vfsqrt_v(v, v); break;

        case eltwise_linear:
            // alpha * x + beta, fused (single rounding): the reference build
            // contracts this expression to fmadd.s (gcc -ffp-contract=fast)
            // and the x64/aarch64 injectors use fused FMA too. The extra
            // rounding of an unfused mul+add is observable once the f32
            // result is converted to an integer dst (half-integer boundary,
            // e.g. alpha*x+beta landing exactly on n+0.5 only because the
            // product was pre-rounded).
            load_f32_const(f_aux0_, alpha_);
            load_f32_const(f_aux1_, beta_);
            h_->vfmv_v_f(v_aux0_, f_aux1_);
            h_->vfmadd_vf(v, f_aux0_, v_aux0_); // v = alpha * v + beta
            break;

        case eltwise_clip:
            // clamp(x, alpha, beta)
            clamp(v, alpha_, beta_);
            break;

        case eltwise_hardsigmoid:
            // clamp(alpha * x + beta, 0, 1)
            load_f32_const(f_aux0_, alpha_);
            h_->vfmul_vf(v, v, f_aux0_);
            load_f32_const(f_aux1_, beta_);
            h_->vfadd_vf(v, v, f_aux1_);
            clamp(v, 0.f, 1.f);
            break;

        case eltwise_hardswish:
            // x * clamp(alpha * x + beta, 0, 1)
            h_->vmv_v_v(v_aux0_, v); // save x
            load_f32_const(f_aux0_, alpha_);
            h_->vfmul_vf(v, v, f_aux0_);
            load_f32_const(f_aux1_, beta_);
            h_->vfadd_vf(v, v, f_aux1_);
            clamp(v, 0.f, 1.f);
            h_->vfmul_vv(v, v, v_aux0_); // x * hardsigmoid(x)
            break;

        case eltwise_exp: exp_compute_vector(v); break;

        case eltwise_logistic: logistic_compute_vector(v); break;

        case eltwise_tanh: tanh_compute_vector(v); break;

        case eltwise_elu: {
            // x>0 ? x : alpha*(exp(x)-1). NaN takes the false branch and
            // remains NaN through exp/sub/mul.
            const Xbyak_riscv::VReg v_mask(0);
            h_->vmv_v_v(v_aux1_, v); // save x
            load_f32_const(f_aux0_, 0.f);
            h_->vmfgt_vf(v_mask, v_aux1_, f_aux0_);
            h_->vfmerge_vfm(v, v, f_aux0_); // positive lanes use exp(0)
            exp_compute_vector(v); // exp(x)
            load_f32_const(f_aux0_, 1.f);
            h_->vfsub_vf(v, v, f_aux0_); // exp(...) - 1
            load_f32_const(f_aux0_, alpha_);
            h_->vfmul_vf(v, v, f_aux0_); // alpha*(...)
            load_f32_const(f_aux0_, 0.f);
            h_->vmfgt_vf(v_mask, v_aux1_, f_aux0_);
            h_->vmerge_vvm(v, v, v_aux1_);
            break;
        }

        case eltwise_swish:
            // x * sigmoid(alpha * x)
            h_->vmv_v_v(v_aux1_, v); // save x (exp/logistic keep aux1 free)
            load_f32_const(f_aux0_, alpha_);
            h_->vfmul_vf(v, v, f_aux0_); // alpha*x
            logistic_compute_vector(v); // sigmoid(alpha*x)
            h_->vfmul_vv(v, v, v_aux1_); // x * sigmoid(alpha*x)
            break;

        case eltwise_gelu_tanh:
            // 0.5*x*(1 + tanh( sqrt(2/pi) * (x + 0.044715*x^3) ))
            h_->vmv_v_v(v_aux1_, v); // save x
            h_->vfmul_vv(v, v, v); // x^2
            load_f32_const(f_aux0_, 0.044715f);
            h_->vfmul_vf(v, v, f_aux0_); // 0.044715*x^2
            load_f32_const(f_aux0_, 1.f);
            h_->vfadd_vf(v, v, f_aux0_); // 1 + 0.044715*x^2
            h_->vfmul_vv(v, v, v_aux1_); // x*(1 + 0.044715*x^2)
            load_f32_const(f_aux0_, 0.7978845608028654f); // sqrt(2/pi)
            h_->vfmul_vf(v, v, f_aux0_); // inner argument
            // tanh(inner): 2*sigmoid(2*inner) - 1
            load_f32_const(f_aux0_, 2.f);
            h_->vfmul_vf(v, v, f_aux0_);
            logistic_compute_vector(v);
            load_f32_const(f_aux0_, 2.f);
            h_->vfmul_vf(v, v, f_aux0_);
            load_f32_const(f_aux0_, -1.f);
            h_->vfadd_vf(v, v, f_aux0_); // tanh(inner)
            load_f32_const(f_aux0_, 1.f);
            h_->vfadd_vf(v, v, f_aux0_); // 1 + tanh
            h_->vfmul_vv(v, v, v_aux1_); // x*(1+tanh)
            load_f32_const(f_aux0_, 0.5f);
            h_->vfmul_vf(v, v, f_aux0_); // 0.5*x*(1+tanh)
            break;

        case eltwise_clip_v2:
            // clip_v2 follows maxNum/minNum behavior: unlike clip, a NaN input
            // selects alpha, matching the reference's ordered comparisons.
            load_f32_const(f_aux0_, alpha_);
            h_->vfmax_vf(v, v, f_aux0_);
            load_f32_const(f_aux0_, beta_);
            h_->vfmin_vf(v, v, f_aux0_);
            break;

        case eltwise_log: {
            // Cephes poly is accurate for finite x > 0; patch the domain edges
            // to match libm/reference: x<0 (incl -inf) -> NaN, x==0 -> -inf,
            // x==+inf -> +inf. Keep the original x in v_aux3 (standalone-only).
            const Xbyak_riscv::VReg v_mask(0);
            h_->vmv_v_v(v_aux3_, v); // save x
            log_compute_vector(v);
            load_f32_const(f_aux0_, std::numeric_limits<float>::quiet_NaN());
            load_f32_const(f_aux1_, 0.f);
            h_->vmflt_vf(v_mask, v_aux3_, f_aux1_); // x < 0
            h_->vfmerge_vfm(v, v, f_aux0_); // -> NaN
            h_->vmfne_vv(v_mask, v_aux3_, v_aux3_); // x is NaN
            h_->vfmerge_vfm(v, v, f_aux0_); // -> NaN
            load_f32_const(f_aux0_, -std::numeric_limits<float>::infinity());
            h_->vmfeq_vf(v_mask, v_aux3_, f_aux1_); // x == 0
            h_->vfmerge_vfm(v, v, f_aux0_); // -> -inf
            load_f32_const(f_aux0_, std::numeric_limits<float>::infinity());
            h_->vmfeq_vf(v_mask, v_aux3_, f_aux0_); // x == +inf
            h_->vfmerge_vfm(v, v, f_aux0_); // -> +inf
            break;
        }

        case eltwise_soft_relu: {
            // (1/alpha) * log1p(exp(alpha*x)); for alpha*x >= exp_overflow the
            // reference returns alpha*x/alpha == x, so blend that in to avoid the
            // clamped-exp plateau diverging at large inputs.
            constexpr float exp_ovf = 88.72283172607421875f;
            const Xbyak_riscv::VReg v_mask(0);
            // log() uses all three aux, so keep x in v_aux3 (standalone-only).
            h_->vmv_v_v(v_aux3_, v); // save x (== in/alpha)
            load_f32_const(f_aux0_, alpha_);
            h_->vfmul_vf(v, v, f_aux0_); // in = alpha*x
            exp_compute_vector(v); // exp(in) (input clamped internally)
            load_f32_const(f_aux0_, 1.f);
            h_->vfadd_vf(v, v, f_aux0_); // 1 + exp(in)
            log_compute_vector(v); // log1p(exp(in))
            load_f32_const(f_aux0_, 1.f / alpha_);
            h_->vfmul_vf(v, v, f_aux0_); // / alpha
            // recompute in = alpha*x and select x where in >= exp_overflow
            load_f32_const(f_aux0_, alpha_);
            h_->vfmul_vf(v_aux0_, v_aux3_, f_aux0_); // in
            load_f32_const(f_aux0_, exp_ovf);
            h_->vmfge_vf(v_mask, v_aux0_, f_aux0_);
            h_->vmerge_vvm(v, v, v_aux3_); // in>=ovf ? x : soft_relu
            // log_compute_vector bit-decomposes its positive-domain input. Test
            // the scaled input so 0 * (+/-inf), as well as an input NaN, is
            // restored to the public NaN result.
            h_->vmfne_vv(v_mask, v_aux0_, v_aux0_);
            load_f32_const(f_aux0_, std::numeric_limits<float>::quiet_NaN());
            h_->vfmerge_vfm(v, v, f_aux0_);
            break;
        }

        case eltwise_mish: {
            // mish(x) = x * tanh(softplus(x)). With w = 1 + exp(x),
            // tanh(softplus(x)) = 1 - 2/(w^2 + 1) (overflow-safe: w^2 -> inf
            // gives 1 - 0 = 1, i.e. mish -> x, matching the reference).
            h_->vmv_v_v(v_aux1_, v); // save x
            exp_compute_vector(v); // exp(x)
            load_f32_const(f_aux0_, 1.f);
            h_->vfadd_vf(v, v, f_aux0_); // w = 1 + exp(x)
            h_->vfmul_vv(v, v, v); // w^2
            load_f32_const(f_aux0_, 1.f);
            h_->vfadd_vf(v, v, f_aux0_); // w^2 + 1
            load_f32_const(f_aux0_, 2.f);
            h_->vfrdiv_vf(v, v, f_aux0_); // 2 / (w^2 + 1)
            load_f32_const(f_aux0_, 1.f);
            h_->vfrsub_vf(v, v, f_aux0_); // 1 - 2/(w^2+1) = tanh(softplus(x))
            h_->vfmul_vv(v, v, v_aux1_); // * x
            break;
        }

        case eltwise_gelu_erf:
            // 0.5*x*(1 + erf(x/sqrt2)) = 0.5*(x + |x|*erf(|x/sqrt2|)), since
            // x*sign(x) == |x|. The sign-free erf keeps everything in 4 aux
            // (erf uses v_aux0..2, x lives in v_aux3), so it works as a post-op.
            h_->vmv_v_v(v_aux3_, v); // save x
            load_f32_const(f_aux0_, 0.707106769084930419921875f); // 1/sqrt(2)
            h_->vfmul_vf(v, v, f_aux0_);
            erf_compute_vector(v); // erf(|x/sqrt2|), uses aux0..2
            h_->li(gpr_aux0_, 0x7fffffff);
            h_->vand_vx(v_aux0_, v_aux3_, gpr_aux0_); // |x|
            h_->vfmul_vv(v, v, v_aux0_); // |x| * erf
            h_->vfadd_vv(v, v, v_aux3_); // + x
            load_f32_const(f_aux0_, 0.5f);
            h_->vfmul_vf(v, v, f_aux0_); // 0.5*(x + |x|*erf)
            break;

        case eltwise_round: {
            // nearbyint semantics: vfcvt follows the caller's current FRM, like
            // the reference and x64 MXCSR path. |s| >= 2^23 is already integral;
            // restore it after the i32 round-trip to avoid conversion overflow.
            // Restore NaNs as well, and copy the input sign to preserve -0.
            const Xbyak_riscv::VReg v_mask(0);
            h_->vmv_v_v(v_aux0_, v); // save s
            h_->vfcvt_x_f_v(v, v); // f32 -> i32 (current FRM)
            h_->vfcvt_f_x_v(v, v); // i32 -> f32 = round(s)
            h_->li(gpr_aux0_, 0x7fffffff);
            h_->vand_vx(v_aux1_, v_aux0_, gpr_aux0_); // |s|
            load_f32_const(f_aux0_, 8388608.0f); // 2^23
            h_->vmfge_vf(v_mask, v_aux1_, f_aux0_); // |s| >= 2^23
            h_->vmfne_vv(v_aux1_, v_aux0_, v_aux0_); // input is NaN
            h_->vmor_mm(v_mask, v_mask, v_aux1_);
            h_->vmerge_vvm(v, v, v_aux0_); // restore s where large
            h_->vfsgnj_vv(v, v, v_aux0_); // preserve the sign of rounded zero
            break;
        }

        default: assert(!"unsupported eltwise alg"); break;
    }

    // eltwise post-op scale: result = scale * alg(x)
    if (scale_ != 1.f) {
        load_f32_const(f_aux0_, scale_);
        h_->vfmul_vf(v, v, f_aux0_);
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::compute_vector_range(
        size_t start_idx, size_t end_idx) {
    for (size_t i = start_idx; i < end_idx; i++)
        compute_body(Vmm(i));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::compute_vector_range(
        const injector_utils::vmm_index_set_t &vmm_idxs) {
    for (size_t idx : vmm_idxs)
        compute_body(Vmm(idx));
}

template struct jit_uni_eltwise_injector_t<v>;
template struct jit_uni_eltwise_injector_t<zvfh>;

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
