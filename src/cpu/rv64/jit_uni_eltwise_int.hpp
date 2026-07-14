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

#ifndef CPU_RV64_JIT_UNI_ELTWISE_INT_HPP
#define CPU_RV64_JIT_UNI_ELTWISE_INT_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_eltwise_pd.hpp"
#include "cpu/platform.hpp"
#include "cpu/rv64/cpu_isa_traits.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct jit_uni_eltwise_int_fwd_kernel_t;

// Standalone integer eltwise primitive, mirroring x64/aarch64
// jit_uni_eltwise_int: s32/s8/u8 forward, limited to the algorithms that are
// well defined on integer input (relu/linear/clip). It keeps integer support
// out of the float jit_uni_eltwise_fwd_t (which owns f32 / f16-zvfh). The
// self-contained kernel widens to f32, applies the algorithm, and saturates on
// store — matching the reference, which also evaluates eltwise in float.
//
// Templated on isa only for naming/registration parity with x64; integers do
// not use zvfh, so only the <v> instance is registered.
template <cpu_isa_t isa>
struct jit_uni_eltwise_int_fwd_t : public primitive_t {
    struct pd_t : public cpu_eltwise_fwd_pd_t {
        using cpu_eltwise_fwd_pd_t::cpu_eltwise_fwd_pd_t;

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit_int8:", isa, ""),
                jit_uni_eltwise_int_fwd_t);

        status_t init(engine_t *engine) {
            UNUSED(engine);
            using namespace dnnl::impl::data_type;
            using namespace dnnl::impl::alg_kind;

            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper dst_d(dst_md());
            const data_type_t d_type = dst_md()->data_type;

            VDISPATCH_ELTWISE(mayiuse(isa), VERBOSE_UNSUPPORTED_ISA);
            VDISPATCH_ELTWISE(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_ELTWISE(
                    utils::one_of(d_type, s32, s8, u8), VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_ELTWISE(src_md()->data_type == d_type,
                    VERBOSE_INCONSISTENT_DT, "src", "dst");
            VDISPATCH_ELTWISE(platform::has_data_type_support(d_type),
                    VERBOSE_UNSUPPORTED_DT);
            // A limited alg set for integer input (numeric nature), matching
            // x64's jit_uni_eltwise_int.
            VDISPATCH_ELTWISE(utils::one_of(desc()->alg_kind, eltwise_relu,
                                      eltwise_linear, eltwise_clip),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_ELTWISE(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_ELTWISE(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_ELTWISE(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_ELTWISE(
                    src_d == dst_d, VERBOSE_INCONSISTENT_MDS, "src", "dst");

            // The kernel walks the tensor flat (nelems(true)); padded elements
            // may only be touched when the algorithm preserves zero.
            use_dense_ = src_d.is_dense(true) && dst_d.is_dense(true)
                    && IMPLICATION(!src_d.is_dense() || !dst_d.is_dense(),
                            is_zero_preserved());
            VDISPATCH_ELTWISE(use_dense_, VERBOSE_UNSUPPORTED_TAG);

            return status::success;
        }

        bool use_dense_ = false;
    };

    jit_uni_eltwise_int_fwd_t(const pd_t *apd);
    ~jit_uni_eltwise_int_fwd_t() override;

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<jit_uni_eltwise_int_fwd_kernel_t> kernel_;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_JIT_UNI_ELTWISE_INT_HPP
