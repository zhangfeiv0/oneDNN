/*******************************************************************************
* Copyright 2025 ZTE Corporation
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

#ifndef CPU_RV64_JIT_UNI_ELTWISE_HPP
#define CPU_RV64_JIT_UNI_ELTWISE_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_eltwise_pd.hpp"
#include "cpu/platform.hpp"
#include "cpu/rv64/cpu_isa_traits.hpp"
#include "cpu/rv64/injectors/jit_uni_eltwise_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

// Standalone eltwise primitive: a thin VLA JIT wrapper around the RVV eltwise
// injector (the same math used for fused post-ops), mirroring
// aarch64/jit_uni_eltwise.cpp. Float-only (f32 / f16-zvfh); the kernel computes
// in f32 and converts f16 at the load/store boundary. Integer eltwise
// (s32/s8/u8) lives in the separate jit_uni_eltwise_int primitive, matching
// x64/aarch64.
struct jit_uni_eltwise_fwd_kernel_t;
struct jit_uni_eltwise_bwd_kernel_t;

// Templated on isa for structural parity with x64/aarch64 and the rv64 pooling
// primitives. RVV is vector-length-agnostic, so the generated code is the same
// for the (single) vector isa; the template parameter selects the registration
// slot and leaves room for isa-specific budgets later.
template <cpu_isa_t isa>
struct jit_uni_eltwise_fwd_t : public primitive_t {
    struct pd_t : public cpu_eltwise_fwd_pd_t {
        using cpu_eltwise_fwd_pd_t::cpu_eltwise_fwd_pd_t;

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit:", isa, ""), jit_uni_eltwise_fwd_t);

        status_t init(engine_t *engine) {
            UNUSED(engine);
            using namespace dnnl::impl::data_type;

            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper dst_d(dst_md());
            const data_type_t d_type = dst_md()->data_type;

            // Runtime ISA dispatch (this primitive is pure JIT and registered
            // via CPU_INSTANCE_RV64). Float-only: the zvfh instance owns f16,
            // the v instance owns f32. Integer dtypes go to jit_uni_eltwise_int.
            VDISPATCH_ELTWISE(mayiuse(isa), VERBOSE_UNSUPPORTED_ISA);
            VDISPATCH_ELTWISE(is_fwd(), VERBOSE_BAD_PROPKIND);
            const bool dt_ok
                    = (isa == zvfh) ? (d_type == f16) : (d_type == f32);
            VDISPATCH_ELTWISE(dt_ok, VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_ELTWISE(
                    src_md()->data_type == d_type, VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_ELTWISE(platform::has_data_type_support(d_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_ELTWISE(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_ELTWISE(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_ELTWISE(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_ELTWISE(
                    src_d == dst_d, VERBOSE_INCONSISTENT_MDS, "src", "dst");
            VDISPATCH_ELTWISE(check_alg_kind(), VERBOSE_BAD_ALGORITHM);

            use_dense_ = src_d.is_dense(true) && dst_d.is_dense(true)
                    && IMPLICATION(!src_d.is_dense() || !dst_d.is_dense(),
                            is_zero_preserved());
            VDISPATCH_ELTWISE(use_dense_, VERBOSE_UNSUPPORTED_TAG);

            return status::success;
        }

        bool use_dense_;

        // Float-only. Forward exposes every alg the eltwise injector implements
        // (the arithmetic ones, the exp()/log()-based transcendentals, and
        // gelu_erf); the kernel feeds alg + alpha/beta straight to the injector.
        // pow is not supported (like aarch64) -> ref.
        bool check_alg_kind() const {
            return eltwise_injector::is_fwd_alg_supported(desc()->alg_kind);
        }
    };

    jit_uni_eltwise_fwd_t(const pd_t *apd);
    ~jit_uni_eltwise_fwd_t() override;

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<jit_uni_eltwise_fwd_kernel_t> kernel_;
};

template <cpu_isa_t isa>
struct jit_uni_eltwise_bwd_t : public primitive_t {
    struct pd_t : public cpu_eltwise_bwd_pd_t {
        using cpu_eltwise_bwd_pd_t::cpu_eltwise_bwd_pd_t;

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit:", isa, ""), jit_uni_eltwise_bwd_t)

        status_t init(engine_t *engine) {
            UNUSED(engine);
            using namespace dnnl::impl::data_type;

            const memory_desc_wrapper diff_src_d(diff_src_md());
            const memory_desc_wrapper diff_dst_d(diff_dst_md());
            // For *_use_dst_for_bwd algs the kernel reads DNNL_ARG_DST and the
            // derivative is expressed in the forward output; otherwise it reads
            // DNNL_ARG_SRC. data_md() selects the matching tensor.
            const memory_desc_wrapper data_d(data_md());
            const data_type_t d_type = data_d.data_type();

            // Runtime ISA dispatch (pure JIT, CPU_INSTANCE_RV64). Backward is
            // float-only, matching x64/aarch64 (whose jit_uni_eltwise_bwd JITs
            // only f32/bf16/f16 and routes integers to ref_eltwise): zvfh owns
            // f16, v owns f32. Integer backward -> ref.
            VDISPATCH_ELTWISE(mayiuse(isa), VERBOSE_UNSUPPORTED_ISA);
            const bool dt_ok
                    = (isa == zvfh) ? (d_type == f16) : (d_type == f32);
            VDISPATCH_ELTWISE(dt_ok, VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_ELTWISE(
                    utils::everyone_is(d_type, diff_src_md()->data_type,
                            diff_dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_ELTWISE(platform::has_data_type_support(d_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_ELTWISE(!is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_ELTWISE(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_ELTWISE(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_ELTWISE(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_ELTWISE(diff_src_d == diff_dst_d,
                    VERBOSE_INCONSISTENT_MDS, "diff_src", "diff_dst");
            // The kernel computes f'(data) * diff_dst, walking data / diff_dst /
            // diff_src in flat lockstep, so the data tensor (src or dst) must
            // share diff_dst's layout — otherwise a different-tag data (e.g.
            // NCHW src vs NHWC diff_dst) would be mispaired element-for-element.
            // x86/aarch64 gate data == diff_dst the same way.
            VDISPATCH_ELTWISE(data_d == diff_dst_d, VERBOSE_INCONSISTENT_MDS,
                    "data", "diff_dst");
            VDISPATCH_ELTWISE(check_alg_kind(), VERBOSE_BAD_ALGORITHM);

            use_dense_ = diff_dst_d.is_dense()
                    || (diff_dst_d.is_dense(true) && is_zero_preserved());
            VDISPATCH_ELTWISE(use_dense_, VERBOSE_UNSUPPORTED_TAG);

            return status::success;
        }

        bool use_dense_;

        bool check_alg_kind() const {
            return eltwise_injector::is_bwd_alg_supported(desc()->alg_kind);
        }
    };

    jit_uni_eltwise_bwd_t(const pd_t *apd);
    ~jit_uni_eltwise_bwd_t() override;

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<jit_uni_eltwise_bwd_kernel_t> kernel_;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_JIT_UNI_ELTWISE_HPP
