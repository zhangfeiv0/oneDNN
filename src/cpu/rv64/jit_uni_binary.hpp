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

#ifndef CPU_RV64_JIT_UNI_BINARY_HPP
#define CPU_RV64_JIT_UNI_BINARY_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_binary_pd.hpp"
#include "cpu/platform.hpp"
#include "cpu/rv64/cpu_isa_traits.hpp"
#include "cpu/rv64/jit_uni_postops_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct jit_uni_binary_kernel_t;

// How src1 broadcasts over dst, and (for per_oc) how the driver iterates.
enum class bcast_t { none, scalar, per_oc_blocked, per_oc_inner };

// Standalone binary primitive: a VLA JIT wrapper that computes src0 OP src1
// (inline) and applies an eltwise post-op chain through the RVV post-op
// injectors, mirroring aarch64/jit_uni_binary.cpp. Supports no-broadcast,
// per-tensor and per-channel src1 broadcast, plus f32/f16/s32/s8/u8 (converted
// at the load/store boundary, like jit_uni_eltwise). Not templated on isa: like
// x64/aarch64 jit_uni_binary_t, a single instance handles every dtype/broadcast
// internally (RVV has one vector isa; f16 just needs zvfh, gated in the pd).
struct jit_uni_binary_t : public primitive_t {
    struct pd_t : public cpu_binary_pd_t {
        using cpu_binary_pd_t::cpu_binary_pd_t;
        DECLARE_COMMON_PD_T("jit:uni", jit_uni_binary_t);

        status_t init(engine_t *engine) {
            UNUSED(engine);
            using namespace data_type;
            const data_type_t dt = dst_md()->data_type;

            // Pure JIT, registered via CPU_INSTANCE_RV64 (runtime dispatch):
            // gate on the V extension here, and additionally on zvfh for f16.
            VDISPATCH_BINARY(mayiuse(v), VERBOSE_UNSUPPORTED_ISA);
            VDISPATCH_BINARY(utils::one_of(dt, f32, f16, s32, s8, u8),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_BINARY(IMPLICATION(dt == f16, mayiuse(zvfh)),
                    VERBOSE_UNSUPPORTED_ISA);
            VDISPATCH_BINARY(utils::everyone_is(dt, src_md(0)->data_type,
                                     src_md(1)->data_type)
                            && platform::has_data_type_support(
                                    src_md(0)->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            // Only post-ops are implemented (an eltwise chain). Reject scales,
            // zero-points and any other non-default attr so they fall back to a
            // reference impl instead of being silently ignored — x64/aarch64
            // gate attrs explicitly (they additionally implement scales).
            VDISPATCH_BINARY(attr()->has_default_values(
                                     primitive_attr_t::skip_mask_t::post_ops),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_BINARY(check_alg(), VERBOSE_BAD_ALGORITHM);
            // Post-ops: eltwise chain handled in-kernel; everything else falls
            // through to another impl.
            VDISPATCH_BINARY(jit_uni_postops_kernel_t::post_ops_supported(
                                     attr()->post_ops_, 1)
                            && eltwise_only_post_ops(),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_BINARY_SC(set_default_params(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_BINARY_SC(attr_.set_default_formats(dst_md()),
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_BINARY(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");

            // Ternary select uses a native (no-f32) merge path: no post-ops, and
            // src1/src2 must match dst exactly (no broadcast).
            const bool is_select = desc()->alg_kind == alg_kind::binary_select;
            VDISPATCH_BINARY(
                    IMPLICATION(is_select, attr()->post_ops_.len() == 0),
                    VERBOSE_UNSUPPORTED_POSTOP);

            VDISPATCH_BINARY(init_bcast(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_BINARY(IMPLICATION(is_select, bcast_ == bcast_t::none),
                    VERBOSE_UNSUPPORTED_TAG);
            // Select reads src2 as a flat s8 mask advancing 1:1 with dst, so it
            // must be s8 and share dst's layout exactly (same flat-lockstep
            // requirement as the no-broadcast src1 path above).
            if (is_select) {
                const memory_desc_wrapper s2_d(src_md(2));
                VDISPATCH_BINARY(src_md(2)->data_type == s8
                                && s2_d.similar_to(
                                        memory_desc_wrapper(dst_md()), true,
                                        false),
                        VERBOSE_UNSUPPORTED_TAG);
            }
            return status::success;
        }

        bcast_t bcast_ = bcast_t::none;
        dim_t block_len_ = 0; // elements per kernel call
        dim_t n_blocks_ = 0; // number of blocks

        bool check_alg() const {
            using namespace alg_kind;
            return utils::one_of(desc()->alg_kind, binary_add, binary_sub,
                    binary_mul, binary_div, binary_max, binary_min,
                    binary_select, binary_ge, binary_gt, binary_le, binary_lt,
                    binary_eq, binary_ne);
        }

        bool eltwise_only_post_ops() const {
            const auto &po = attr()->post_ops_;
            for (int i = 0; i < po.len(); i++)
                if (!po.entry_[i].is_eltwise()) return false;
            return true;
        }

        // Classify src1 broadcast vs dst and set the driver iteration params.
        // Requires plain dense tensors with identical src0/dst layout.
        bool init_bcast() {
            const memory_desc_wrapper s0(src_md(0));
            const memory_desc_wrapper s1(src_md(1));
            const memory_desc_wrapper d(dst_md());
            if (s0.blocking_desc().inner_nblks || s1.blocking_desc().inner_nblks
                    || d.blocking_desc().inner_nblks)
                return false;
            if (!s0.is_dense(false) || !s1.is_dense(false)
                    || !d.is_dense(false))
                return false;
            // src0 must match dst exactly (no src0 broadcast).
            if (!s0.similar_to(d, true, false)) return false;

            const int nd = d.ndims();
            const dim_t *dd = d.dims();
            const dim_t *d1 = s1.dims();
            const dim_t total = d.nelems(false);

            // all-ones src1 -> per-tensor scalar
            bool all_one = true;
            for (int i = 0; i < nd; i++)
                if (d1[i] != 1) all_one = false;
            if (all_one) {
                bcast_ = bcast_t::scalar;
                block_len_ = total;
                n_blocks_ = 1;
                return true;
            }
            // no broadcast: src1 must match dst LAYOUT, not just dims. execute()
            // walks src1 and dst flat-contiguously in lockstep, so a same-dims
            // but different-layout src1 (e.g. NCHW src1 vs NHWC dst) would
            // mispair elements. Require identical dims+strides via similar_to
            // (x64/aarch64 do the same for their no_broadcast path).
            if (s1.similar_to(d, true, false)) {
                bcast_ = bcast_t::none;
                block_len_ = total;
                n_blocks_ = 1;
                return true;
            }
            // per-oc: src1 = [1, C, 1, ...] (channel dim 1 only)
            if (nd >= 2 && d1[1] == dd[1]) {
                bool only_c = true;
                for (int i = 0; i < nd; i++)
                    if (i != 1 && d1[i] != 1) only_c = false;
                if (only_c) {
                    const dim_t C = dd[1];
                    dim_t inner = 1;
                    for (int i = 2; i < nd; i++)
                        inner *= dd[i];
                    const auto &dstr = d.blocking_desc().strides;
                    if (dstr[1] == inner) {
                        // channels-first (nchw): N*C blocks, one spatial
                        // plane (inner elements) each; channel index wraps
                        // mod C.
                        bcast_ = bcast_t::per_oc_blocked;
                        block_len_ = inner;
                        n_blocks_ = total / inner; // = N*C
                        return true;
                    } else if (dstr[1]
                            == 1) { // channels-last (nhwc): per pixel
                        bcast_ = bcast_t::per_oc_inner;
                        block_len_ = C;
                        n_blocks_ = total / C;
                        return true;
                    }
                }
            }
            return false;
        }
    };

    jit_uni_binary_t(const pd_t *apd);
    ~jit_uni_binary_t() override;

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<jit_uni_binary_kernel_t> kernel_;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_JIT_UNI_BINARY_HPP
