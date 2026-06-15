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
#include "common/memory_desc_wrapper.hpp"

#include "cpu/rv64/injectors/jit_uni_postops_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace injector {

// scalar (per_tensor) when src1 has a single element, else a per-element run.
static binary_injector::broadcast_t get_broadcast(
        const post_ops_t::entry_t::binary_t &binary) {
    const memory_desc_wrapper src1_d(binary.src1_desc);
    return src1_d.nelems() == 1 ? binary_injector::broadcast_t::scalar
                                : binary_injector::broadcast_t::per_element;
}

template <cpu_isa_t isa>
jit_uni_postops_injector_t<isa>::jit_uni_postops_injector_t(
        jit_generator_t *host, const post_ops_t &post_ops,
        const eltwise_injector::static_params_t &eltwise_static_params,
        const binary_injector::static_params_t *binary_static_params)
    : host_(host), post_ops_(post_ops) {
    eltwise_injectors_.reserve(post_ops.len());
    binary_injectors_.reserve(post_ops.len());
    int arg_idx = 0; // this binary's slot in the rhs pointer array
    for (int i = 0; i < post_ops.len(); i++) {
        const auto &e = post_ops.entry_[i];
        if (e.is_eltwise()) {
            eltwise_injectors_.emplace_back(
                    host_, e.eltwise, eltwise_static_params);
        } else if (e.is_binary()) {
            assert(binary_static_params != nullptr
                    && "binary post-op requires binary scratch");
            binary_injectors_.emplace_back(host_, e.binary.alg,
                    get_broadcast(e.binary), *binary_static_params, arg_idx++);
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_postops_injector_t<isa>::compute_body(const Vmm &v) {
    // Apply entries in attribute order: eltwise and binary injectors are each
    // consumed in the order they were created (binaries index the rhs array by
    // their creation order via the arg_idx baked in at construction).
    size_t e_idx = 0, b_idx = 0;
    for (int i = 0; i < post_ops_.len(); i++) {
        const auto &e = post_ops_.entry_[i];
        if (e.is_eltwise())
            eltwise_injectors_[e_idx++].compute_vector(v.getIdx());
        else if (e.is_binary())
            binary_injectors_[b_idx++].compute_vector(v.getIdx());
    }
}

template <cpu_isa_t isa>
void jit_uni_postops_injector_t<isa>::compute_vector_range(
        size_t start_idx, size_t end_idx) {
    // NB: a binary post-op reads its rhs from the host-positioned scratch (one
    // slice per call), so when the chain contains a per-element binary only a
    // SINGLE register is correct here — every register would re-read the same
    // rhs slice. All current consumers (matmul/brgemm/conv/pool/binary) pass
    // exactly one register; an eltwise-only chain is safe for any range.
    for (size_t i = start_idx; i < end_idx; i++)
        compute_body(Vmm(i));
}

template <cpu_isa_t isa>
void jit_uni_postops_injector_t<isa>::compute_vector_range(
        const injector_utils::vmm_index_set_t &vmm_idxs) {
    for (size_t idx : vmm_idxs)
        compute_body(Vmm(idx));
}

template <cpu_isa_t isa>
bool jit_uni_postops_injector_t<isa>::post_ops_ok(const post_ops_t &post_ops) {
    for (int i = 0; i < post_ops.len(); i++) {
        const auto &e = post_ops.entry_[i];
        if (e.is_eltwise()) {
            if (!eltwise_injector::is_alg_supported(e.eltwise.alg))
                return false;
        } else if (e.is_binary()) {
            if (!binary_injector::is_alg_supported(e.binary.alg)) return false;
        } else {
            return false; // sum/prelu/... -> consumer falls back to a ref impl
        }
    }
    return true;
}

template struct jit_uni_postops_injector_t<v>;
template struct jit_uni_postops_injector_t<zvfh>;

} // namespace injector
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
