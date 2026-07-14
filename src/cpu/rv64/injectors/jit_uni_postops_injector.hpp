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
#ifndef CPU_RV64_INJECTORS_JIT_UNI_POSTOPS_INJECTOR_HPP
#define CPU_RV64_INJECTORS_JIT_UNI_POSTOPS_INJECTOR_HPP

#include <memory>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/primitive_attr.hpp"

#include "cpu/rv64/injectors/injector_utils.hpp"
#include "cpu/rv64/injectors/jit_uni_binary_injector.hpp"
#include "cpu/rv64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/rv64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace injector {

// In-kernel post-op chain injector for RVV, mirroring the x64/aarch64
// jit_uni_postops_injector_t role: the host kernel computes its accumulator
// tile into vector registers, then hands those register groups to
// compute_vector(_range) to apply the full post-op chain in place, in attribute
// order.
//
// Coverage (gated by post_ops_ok): a chain of injector-supported forward
// eltwise ops plus any number of injector-supported binary ops. Each binary
// reads its rhs from the pointer array carried in the (indirect) binary scratch,
// indexed by its position among the chain's binaries — the x64/aarch64 scheme.
// Exotic broadcasts, sum and prelu are not covered — the consumer pd rejects
// them so the framework selects a reference impl. Binary select additionally
// needs an independent condition scratch configuration and is enabled only by
// the standalone binary host. The binary scratch is optional: pass nullptr when
// the chain has no binary entry.
template <cpu_isa_t isa>
struct jit_uni_postops_injector_t {
    using Vmm = typename jit_isa_traits_t<isa>::Vmm;

    // dst_md (optional): when provided, per-channel binary post-op rhs
    // ([1, C, 1, ...]) is classified as per_oc and the injector computes the
    // channel offset from the output offset at call time. Without it, binaries
    // are scalar / per_element only.
    jit_uni_postops_injector_t(jit_generator_t *host,
            const post_ops_t &post_ops,
            const eltwise_injector::static_params_t &eltwise_static_params,
            const binary_injector::static_params_t *binary_static_params
            = nullptr,
            const memory_desc_t *dst_md = nullptr,
            const binary_injector::static_params_t *select_static_params
            = nullptr);

    // Apply the chain to the accumulator register group, identified by index
    // (x64/aarch64 parity). Binary entries compute their rhs address from the
    // per-register output offset in rhs_arg_params. entry_begin/entry_end select
    // an entry sub-range so a host can apply the chain in pieces (e.g. around an
    // in-kernel sum, which the injector skips); default is the whole chain.
    void compute_vector(size_t idx,
            const binary_injector::rhs_arg_dynamic_params_t &rhs_arg_params,
            int entry_begin = 0, int entry_end = -1) {
        compute_body(Vmm(idx), rhs_arg_params, entry_begin,
                entry_end < 0 ? (int)post_ops_.len() : entry_end);
    }
    void compute_vector_range(size_t start_idx, size_t end_idx,
            const binary_injector::rhs_arg_dynamic_params_t &rhs_arg_params);

    // True when every entry of `post_ops` can be injected in-kernel by this
    // injector: any number of supported forward eltwise ops, plus any number of
    // supported binary ops. n_vaux is how many vector aux groups the host
    // supplies to the eltwise static_params; the heavy eltwise algs (log/
    // soft_relu/gelu_erf) require n_vaux >= 4.
    static bool post_ops_ok(const post_ops_t &post_ops, int n_vaux = 3,
            bool allow_binary_select = false);

private:
    void compute_body(const Vmm &v,
            const binary_injector::rhs_arg_dynamic_params_t &rhs_arg_params,
            int entry_begin, int entry_end);
    jit_generator_t *const host_;
    post_ops_t post_ops_;
    std::vector<jit_uni_eltwise_injector_t<isa>> eltwise_injectors_;
    std::vector<jit_uni_binary_injector_t<isa>> binary_injectors_;
};

} // namespace injector
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_INJECTORS_JIT_UNI_POSTOPS_INJECTOR_HPP
