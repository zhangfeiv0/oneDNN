/*******************************************************************************
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

#ifndef CPU_RV64_JIT_UNI_POOL_KERNEL_HPP
#define CPU_RV64_JIT_UNI_POOL_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/primitive_attr.hpp"

#include "cpu/cpu_pooling_pd.hpp"
#include "cpu/rv64/jit_generator.hpp"
#include "cpu/rv64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

// Forward pooling kernel for the agnostic (runtime-shape) path. Templated on
// isa and data type: generate_f32() emits the f32 path and generate_f16() the
// zvfh f16 path, so the generated code differs per d_type. The kernel is
// vector-length-agnostic (vsetvli) and shape-agnostic (window/strides come in
// via jit_uni_pooling_args_t), so one routine per instantiation serves ncsp,
// OW==1, the nspc boundary columns, and (for f16) the whole nspc row.
template <cpu_isa_t isa, impl::data_type_t d_type>
struct jit_uni_pool_kernel_t : public jit_generator_t {

    jit_uni_pool_kernel_t(alg_kind_t alg);

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_pool_kernel_t)

    // Populate jpp from the primitive descriptor (dims, strides, window, pad,
    // algorithm, memory tag, post-ops). Mirrors aarch64/x64 init_conf.
    static status_t init_conf(jit_pool_conf_t &jpp, primitive_attr_t &attr,
            const pooling_pd_t *ppd);

    void operator()(const jit_uni_pooling_args_t *p) const {
        jit_generator_t::operator()(p);
    }

protected:
    void generate() override;

private:
    bool is_max_pool_;
    void generate_f32();
    // f16 path: max accumulates in f16; avg widens to f32 (vfwadd_wv), scales,
    // and narrows back to f16 (vfncvt). Requires the zvfh extension. Post-ops
    // are not fused in this kernel; f16 post-ops (incl. ReLU) run as separate
    // primitives via the driver's rvv_postops_t path.
    void generate_f16();
};

// Shape-baked forward pooling kernel for the interior (full-kw) region of an
// nspc row (f32 only; f16 uses the agnostic kernel above). kw, stride_w, ur_w,
// the algorithm, ReLU, and the avg_include scale are baked into the generated
// code; the W window is fully unrolled and ur_w output columns share each
// loaded input vector (ARM max_step/avg_step style). Base pointers, the VLA
// channel count, the runtime H/D extents, the element strides (passed, not
// baked, so 64-bit strides stay correct), and the avg_exclude scale are passed
// at call time.
template <cpu_isa_t isa, impl::data_type_t d_type>
struct jit_uni_pool_interior_kernel_t : public jit_generator_t {

    jit_uni_pool_interior_kernel_t(const jit_pool_conf_t &ajpp);

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_pool_interior_kernel_t)

    void operator()(const jit_uni_pool_interior_args_t *p) const {
        jit_generator_t::operator()(p);
    }

protected:
    void generate() override;

private:
    jit_pool_conf_t jpp_;
    void generate_nspc();
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_JIT_UNI_POOL_KERNEL_HPP
