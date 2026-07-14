/*******************************************************************************
* Copyright 2020 Intel Corporation
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
#include <cstddef>
#include <cstring>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/type_helpers.hpp"

#include "cpu/rv64/jit_generator.hpp"
#include "cpu/rv64/jit_uni_eltwise_int.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

// Self-contained integer forward kernel: one vector-length-agnostic pass that
// loads `len` elements of dt (s32/s8/u8) from a1, widens to f32 at e32/m4,
// applies relu/linear/clip inline, then saturates to the dt range and stores to
// a2. The alg / alpha / beta / dt are baked in; only the pointers and length
// arrive at run time. Computing in f32 matches the reference (which also
// evaluates eltwise in float); RISC-V vfcvt.x.f saturates, so the upper s32
// clamp is max_value<f32>(s32) == 2147483520.f (not 2147483647.f, which rounds
// up to 2^31 and would still convert to INT_MAX, but keep the exact bound).
//
// Registers: a1=src a2=dst a3=len; t0=vl t1=bytes t2=gpr; v4=f32 data (m4),
// v8=relu neg temp (m4), v12=8/16-bit staging; fa0=materialized const.
struct jit_uni_eltwise_int_fwd_kernel_t : public jit_generator_t {
    struct call_params_t {
        const void *src;
        void *dst;
        dim_t len;
    };
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_eltwise_int_fwd_kernel_t)

    jit_uni_eltwise_int_fwd_kernel_t(
            alg_kind_t alg, float alpha, float beta, data_type_t dt)
        : jit_generator_t("jit_uni_eltwise_int_fwd")
        , alg_(alg)
        , alpha_(alpha)
        , beta_(beta)
        , dt_(dt) {}
    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }
    void generate() override {
        ld(a1, a0, 0); // src
        ld(a2, a0, 8); // dst
        ld(a3, a0, 16); // len
        emit();
        ret();
    }

private:
    const alg_kind_t alg_;
    const float alpha_, beta_;
    const data_type_t dt_;

    const Reg p_in = a1, p_out = a2, len = a3, vl = t0, bytes = t1, gpr = t2;
    const VReg vd {4}, vtmp {8}, vst {12};
    const FReg fc = fa0;

    void setf(float val) {
        uint32_t b;
        std::memcpy(&b, &val, sizeof(b));
        li(gpr, b);
        fmv_w_x(fc, gpr);
    }

    // relu/linear/clip in f32, at the e32/m4 compute vtype.
    void compute_alg() {
        using namespace alg_kind;
        switch (alg_) {
            case eltwise_relu:
                // d = max(x,0) + alpha*min(x,0) (covers leaky and alpha==0).
                setf(0.f);
                vfmin_vf(vtmp, vd, fc); // neg = min(x, 0)
                vfmax_vf(vd, vd, fc); // pos = max(x, 0)
                setf(alpha_);
                vfmacc_vf(vd, fc, vtmp); // d = pos + alpha * neg
                break;
            case eltwise_linear: // d = alpha*x + beta
                setf(alpha_);
                vfmul_vf(vd, vd, fc);
                setf(beta_);
                vfadd_vf(vd, vd, fc);
                break;
            case eltwise_clip: // d = max(alpha, min(beta, x))
                setf(beta_);
                vfmin_vf(vd, vd, fc);
                setf(alpha_);
                vfmax_vf(vd, vd, fc);
                break;
            default: break;
        }
    }

    void emit() {
        using namespace data_type;
        const bool is_s8 = dt_ == s8;

        Label loop, done;
        L(loop);
        beqz(len, done);

        // ---- load + widen to f32 (e32/m4) ----
        if (dt_ == s32) {
            vsetvli(vl, len, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);
            vle32_v(vd, p_in);
            vfcvt_f_x_v(vd, vd);
        } else { // s8 / u8
            vsetvli(vl, len, SEW::e8, LMUL::m1, VTA::ta, VMA::ma);
            vle8_v(vst, p_in);
            // vsext/vzext.vf4 operate at the destination vtype (e32m4) and read
            // the source group as 8-bit, so switch vtype before extending.
            vsetvli(x0, vl, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);
            if (is_s8) {
                vsext_vf4(vd, vst);
                vfcvt_f_x_v(vd, vd);
            } else {
                vzext_vf4(vd, vst);
                vfcvt_f_xu_v(vd, vd);
            }
        }

        // ---- alg (vtype currently e32/m4) ----
        compute_alg();

        // ---- saturate to the dt range + narrow + store ----
        if (dt_ == s32) {
            setf(-2147483648.0f);
            vfmax_vf(vd, vd, fc);
            setf(2147483520.0f); // max_value<f32>(s32)
            vfmin_vf(vd, vd, fc);
            vfcvt_x_f_v(vd, vd);
            vse32_v(vd, p_out);
        } else { // s8 / u8
            setf(is_s8 ? -128.0f : 0.0f);
            vfmax_vf(vd, vd, fc);
            setf(is_s8 ? 127.0f : 255.0f);
            vfmin_vf(vd, vd, fc);
            if (is_s8)
                vfcvt_x_f_v(vd, vd);
            else
                vfcvt_xu_f_v(vd, vd);
            // narrow i32(m4) -> i16(m2) -> i8(m1); values pre-clamped so the
            // vnsrl-by-0 truncation is exact.
            vsetvli(x0, vl, SEW::e16, LMUL::m2, VTA::ta, VMA::ma);
            vnsrl_wi(vst, vd, 0);
            vsetvli(x0, vl, SEW::e8, LMUL::m1, VTA::ta, VMA::ma);
            vnsrl_wi(vtmp, vst, 0);
            vse8_v(vtmp, p_out);
        }

        // ---- advance by vl * sizeof(dt) and loop ----
        const int dsz = static_cast<int>(types::data_type_size(dt_));
        const int sh = dsz == 4 ? 2 : 0;
        if (sh)
            slli(bytes, vl, sh);
        else
            mv(bytes, vl);
        add(p_in, p_in, bytes);
        add(p_out, p_out, bytes);
        sub(len, len, vl);
        j_(loop);

        L(done);
    }
};

template <cpu_isa_t isa>
jit_uni_eltwise_int_fwd_t<isa>::jit_uni_eltwise_int_fwd_t(const pd_t *apd)
    : primitive_t(apd) {}
template <cpu_isa_t isa>
jit_uni_eltwise_int_fwd_t<isa>::~jit_uni_eltwise_int_fwd_t() = default;

template <cpu_isa_t isa>
status_t jit_uni_eltwise_int_fwd_t<isa>::init(engine_t *engine) {
    UNUSED(engine);
    const auto &d = *pd()->desc();
    kernel_.reset(new jit_uni_eltwise_int_fwd_kernel_t(
            d.alg_kind, d.alpha, d.beta, pd()->dst_md()->data_type));
    return kernel_->create_kernel();
}

template <cpu_isa_t isa>
status_t jit_uni_eltwise_int_fwd_t<isa>::execute(const exec_ctx_t &ctx) const {
    const auto *src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto *dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);
    const memory_desc_wrapper data_d(pd()->src_md());
    const dim_t len = data_d.nelems(true);
    const size_t es = types::data_type_size(pd()->dst_md()->data_type);
    // Honor a non-zero md offset (dense submemory view); src and dst share md.
    const size_t base = (size_t)data_d.offset0() * es;

    const int max_thr = dnnl_get_max_threads();
    parallel(max_thr, [&](int ithr, int nthr) {
        dim_t start = 0, end = 0;
        balance211(len, nthr, ithr, start, end);
        if (start >= end) return;
        jit_uni_eltwise_int_fwd_kernel_t::call_params_t p;
        p.src = static_cast<const char *>(src) + base + start * es;
        p.dst = static_cast<char *>(dst) + base + start * es;
        p.len = end - start;
        (*kernel_)(&p);
    });
    return status::success;
}

template struct jit_uni_eltwise_int_fwd_t<v>;

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
