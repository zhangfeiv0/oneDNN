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
#ifndef CPU_RV64_JIT_UNI_POSTOPS_KERNEL_HPP
#define CPU_RV64_JIT_UNI_POSTOPS_KERNEL_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "common/primitive_attr.hpp"

#include "cpu/rv64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

// Universal JIT kernel that applies, over a contiguous f32 run ("the unit" — a
// matmul row, a convolution oc-slice, or a whole pooling/dense dst buffer):
//   dst = postops( dst + bias )
// in a single fused vsetvli pass, using the RVV post-op injectors. It is the
// shared in-kernel epilogue for matmul / brgemm_matmul / gemm convolution (the
// analog of x64's pp-kernels), replacing their earlier per-primitive bias+ReLU
// kernels.
//
// The caller invokes the kernel once per unit, passing the unit's dst (and, for
// per-element bias/binary, the matching base pointers). bias and the binary rhs
// are each either a broadcast scalar or a per-element run aligned 1:1 with the
// unit; the caller guarantees that alignment (see binary_broadcast_ok()).
struct jit_uni_postops_kernel_t {
    struct call_params_t {
        void *dst; // f32 unit, modified in place
        const void *bias; // bias base (scalar or per-element), or null
        // Base of an array of per-binary rhs base pointers (one entry per binary
        // post-op, in chain order), or null when the chain has no binary. Each
        // base is scalar or a per-element run; per-element lanes are read at
        // base + off0 + chunk. off0 is the byte offset of the unit's first lane
        // within the rhs operand: 0 when the unit starts at the rhs origin (a
        // matmul row over a per-N rhs), or the channel-base byte offset when the
        // unit is a sub-slice (a conv oc-slice over a per-oc rhs). Scalars
        // ignore off0.
        const void *const *rhs;
        dim_t off0; // byte offset of the unit's first lane in the rhs operand
        dim_t len; // number of f32 elements in the unit
    };

    struct conf_t {
        data_type_t dst_dt = data_type::f32;
        bool with_bias = false;
        bool bias_per_element = false; // false => scalar broadcast
    };

    // Build a kernel for `po` + `conf`. Returns unimplemented (kernel unset) if
    // dst is not f32, the chain is not injector-supported (post_ops_ok), or any
    // binary rhs is not f32 (binary_rhs_dt_ok). The caller must have validated,
    // for its unit, that any per-element binary rhs aligns 1:1 with the unit
    // (binary_broadcast_ok()); bias alignment is the caller's responsibility via
    // conf.bias_per_element.
    static status_t create(std::shared_ptr<jit_uni_postops_kernel_t> &kernel,
            const post_ops_t &po, const conf_t &conf);

    // True if every binary entry's src1 is broadcastable over the caller's unit:
    // either a single value (per-tensor) or a dense run of exactly
    // `unit_nelems` elements (per-element 1:1 with the unit). Callers use this
    // to gate kernel creation.
    static bool binary_broadcast_ok(const post_ops_t &po, dim_t unit_nelems);

    // True if every binary post-op's src1 is f32 — the only rhs dtype the
    // injector loads (it emits flw/vle32/vlse32). Non-f32 rhs must fall back.
    static bool binary_rhs_dt_ok(const post_ops_t &po);

    // True if every binary post-op's src1 is scalar or broadcasts to EXACTLY
    // dst's last dim of size `last_dim` (src1 last dim == last_dim, all leading
    // dims == 1). Stricter than nelems==last_dim, which would also admit an
    // other-axis broadcast like per-M [M,1] when M==last_dim. Matmul/brgemm use
    // this: they apply the rhs per output row at a fixed offset, so only a per-N
    // (or scalar) rhs is correct.
    static bool binary_per_last_dim_ok(const post_ops_t &po, dim_t last_dim);

    // Combined pd-side gate: the injector covers the whole chain, every binary
    // is broadcastable over a `unit_nelems`-element unit, and every binary rhs
    // is f32. Consumer pds call this so they need not include the injector
    // headers themselves.
    static bool post_ops_supported(const post_ops_t &po, dim_t unit_nelems);

    void operator()(const call_params_t *p) const;

    virtual ~jit_uni_postops_kernel_t();

private:
    jit_uni_postops_kernel_t();
    struct impl_t;
    std::unique_ptr<impl_t> impl_;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_JIT_UNI_POSTOPS_KERNEL_HPP
