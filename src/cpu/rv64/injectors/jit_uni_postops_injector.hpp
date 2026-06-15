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
// Comparison binaries, exotic broadcasts, sum and prelu are not covered — the
// consumer pd rejects them so the framework selects a reference impl. The binary
// scratch is optional: pass nullptr when the chain has no binary entry.
template <cpu_isa_t isa>
struct jit_uni_postops_injector_t {
    using Vmm = typename jit_isa_traits_t<isa>::Vmm;

    jit_uni_postops_injector_t(jit_generator_t *host,
            const post_ops_t &post_ops,
            const eltwise_injector::static_params_t &eltwise_static_params,
            const binary_injector::static_params_t *binary_static_params
            = nullptr);

    // Apply the whole chain to the accumulator register group(s), identified by
    // index (x64/aarch64 parity). Binary entries read their rhs from the
    // host-positioned rhs pointer carried in binary_static_params.
    void compute_vector(size_t idx) { compute_vector_range(idx, idx + 1); }
    void compute_vector_range(size_t start_idx, size_t end_idx);
    void compute_vector_range(const injector_utils::vmm_index_set_t &vmm_idxs);

    // True when every entry of `post_ops` can be injected in-kernel by this
    // injector: any number of supported forward eltwise ops, plus any number of
    // supported binary ops.
    static bool post_ops_ok(const post_ops_t &post_ops);

private:
    void compute_body(const Vmm &v);
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
