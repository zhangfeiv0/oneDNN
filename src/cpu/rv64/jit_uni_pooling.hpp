/*******************************************************************************
* Copyright 2024 Intel Corporation
* Copyright 2023 KNS Group LLC (YADRO)
* Copyright 2026 ZTE Corporation
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

#ifndef CPU_RV64_JIT_UNI_POOLING_HPP
#define CPU_RV64_JIT_UNI_POOLING_HPP

#include <memory>

#include "common/primitive.hpp"

#include "cpu/cpu_pooling_pd.hpp"
#include "cpu/rv64/cpu_isa_traits.hpp"
#include "cpu/rv64/jit_uni_pool_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

template <cpu_isa_t isa>
struct jit_uni_pooling_fwd_t : public primitive_t {
    static constexpr data_type_t d_type
            = (isa == zvfh) ? data_type::f16 : data_type::f32;

    struct pd_t : public cpu_pooling_fwd_pd_t {
        using cpu_pooling_fwd_pd_t::cpu_pooling_fwd_pd_t;

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit:", jpp_.isa, ""),
                jit_uni_pooling_fwd_t);

        status_t init(engine_t *engine) {
            using namespace utils;
            using namespace data_type;

            VDISPATCH_POOLING(mayiuse(isa), VERBOSE_UNSUPPORTED_ISA);
            VDISPATCH_POOLING(desc_.prop_kind == prop_kind::forward_inference,
                    VERBOSE_BAD_PROPKIND);
            VDISPATCH_POOLING(one_of(desc()->alg_kind, alg_kind::pooling_max,
                                      alg_kind::pooling_avg_include_padding,
                                      alg_kind::pooling_avg_exclude_padding),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_POOLING(everyone_is(d_type, src_md()->data_type,
                                      dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_POOLING(
                    platform::has_data_type_support(src_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            // f16 accumulates in f32 (checked just below). f16 eltwise post-ops
            // fuse in the f16 kernel (generate_f16); an f16 + binary chain is
            // rejected by post_ops_ok() to ref_pooling.
            constexpr bool is_f16 = d_type == data_type::f16;
            VDISPATCH_POOLING(
                    IMPLICATION(
                            is_f16, desc()->accum_data_type == data_type::f32),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_POOLING(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_POOLING(!is_dilated(), VERBOSE_UNSUPPORTED_FEATURE,
                    "does not support dilations");
            VDISPATCH_POOLING(set_default_params() == status::success,
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_POOLING(memory_desc_wrapper(dst_md()).is_dense(false),
                    VERBOSE_UNSUPPORTED_SPARSE_CFG);
            VDISPATCH_POOLING(attr()->has_default_values(
                                      primitive_attr_t::skip_mask_t::post_ops),
                    VERBOSE_UNSUPPORTED_ATTR);
            // Resolve post-op binary src1 formats (may be format_any) against dst
            // before post_ops_ok() inspects their layout via similar_to(); dst is
            // already concrete here (set_default_params above). Matches x86/ARM
            // pooling, which set binary post-op formats before the post-op check.
            VDISPATCH_POOLING(
                    attr_.set_default_formats(dst_md(0)) == status::success,
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_POOLING(post_ops_ok(), VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_POOLING(KW() < max_kernel_width,
                    VERBOSE_UNSUPPORTED_FEATURE,
                    "kernel width exceeds maximum");

            // init_conf fills jpp_ and selects the memory layout (nspc/ncsp);
            // it returns unimplemented for any other tag. (Computed in a
            // separate statement so the template-argument comma is not parsed
            // as a macro-argument separator.)
            const status_t conf_status
                    = jit_uni_pool_kernel_t<isa, d_type>::init_conf(
                            jpp_, attr_, this);
            VDISPATCH_POOLING(
                    conf_status == status::success, VERBOSE_UNSUPPORTED_TAG);

            // Post-ops accepted by post_ops_ok() are all fused in the kernel
            // (eltwise for f32/f16, binary for f32); nothing runs as a separate
            // primitive, so there is no post-op orchestrator here.
            return status::success;
        }

        jit_pool_conf_t jpp_;

    private:
        bool post_ops_ok() const {
            const auto &po = attr()->post_ops_;
            if (po.has_default_values()) return true;
            // Accept an injector-supported chain (any number of eltwise plus
            // binary, in attribute order); every entry is fused in the kernel.
            // The pool kernel positions a single host-computed rhs (channel-vec
            // forcing), so at most ONE binary is supported here even though the
            // injector itself allows more (that capability is used by
            // matmul/conv). Binary is fused for f32 only. Anything else (sum,
            // prelu, >1 binary, f16+binary, unsupported alg) falls back to
            // ref_pooling.
            if (!injector::jit_uni_postops_injector_t<isa>::post_ops_ok(po))
                return false;
            const memory_desc_wrapper dst_d(dst_md());
            int n_binary = 0;
            for (int i = 0; i < po.len(); i++) {
                if (!po.entry_[i].is_binary()) continue;
                if (++n_binary > 1) return false; // single host-positioned rhs
                if (d_type != data_type::f32) return false; // f32 fusion only
                const auto &b = po.entry_[i].binary;
                // The injector loads the rhs as f32; binary_rhs_for() only knows
                // three broadcasts: scalar, per-oc ([1,C,1,..]) and full-dst
                // (read 1:1 with the output via p_dst-dst, so it must share dst's
                // layout). Reject anything else (e.g. broadcast-over-C like
                // [N,1,OH,OW]) so it falls back to ref_pooling.
                if (b.src1_desc.data_type != data_type::f32) return false;
                const memory_desc_wrapper s1(b.src1_desc);
                const bool scalar = s1.nelems(true) == 1;
                // per-oc is read as a contiguous [C] vector (vle32), so the C
                // values must be dense; full-dst is read 1:1 via similar_to
                // (which already enforces strides).
                bool per_oc = dst_d.ndims() >= 2 && s1.ndims() == dst_d.ndims()
                        && s1.dims()[1] == dst_d.dims()[1] && s1.is_dense(true);
                for (int k = 0; per_oc && k < dst_d.ndims(); k++)
                    if (k != 1 && s1.dims()[k] != 1) per_oc = false;
                const bool full = s1.similar_to(dst_d, true, false);
                if (!scalar && !per_oc && !full) return false;
            }
            return true;
        }
    };

    jit_uni_pooling_fwd_t(const pd_t *apd);
    ~jit_uni_pooling_fwd_t() override;

    using data_t = typename prec_traits_t<d_type>::type;

    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

    static constexpr int max_kernel_width = 32;

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<jit_uni_pool_kernel_t<isa, d_type>> kernel_;
    // Shape-baked interior kernel (ur_w input reuse), used for the full-window
    // interior of nspc rows; null for other layouts.
    std::unique_ptr<jit_uni_pool_interior_kernel_t<isa, d_type>>
            interior_kernel_;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_JIT_UNI_POOLING_HPP
