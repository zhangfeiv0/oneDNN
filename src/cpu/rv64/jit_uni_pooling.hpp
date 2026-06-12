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
#include "cpu/rv64/rvv_postops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

template <cpu_isa_t isa, impl::data_type_t d_type>
struct jit_uni_pooling_fwd_t : public primitive_t {
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
            // f16 accumulates in f32; a single binary/eltwise post-op (incl.
            // ReLU) is run via the separate-primitive path below rather than
            // fused into the f16 kernel.
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
            VDISPATCH_POOLING(post_ops_ok(), VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_POOLING(
                    attr_.set_default_formats(dst_md(0)) == status::success,
                    VERBOSE_UNSUPPORTED_POSTOP);
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

            // Only f32 ReLU is fused in the kernel (jpp_.with_relu). Every other
            // single post-op — binary, non-ReLU eltwise, or any f16 post-op
            // (incl. f16 ReLU) — runs as a separate primitive after the kernel.
            if (jpp_.with_postops && !jpp_.with_relu)
                CHECK(postops_.init(engine, attr()->post_ops_, *dst_md()));

            return status::success;
        }

        jit_pool_conf_t jpp_;
        rvv_postops_t postops_;

    private:
        bool post_ops_ok() const {
            const auto &po = attr()->post_ops_;
            if (po.has_default_values()) return true;
            return po.len() == 1
                    && (po.entry_[0].is_binary() || po.entry_[0].is_eltwise());
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
