/*******************************************************************************
* Copyright 2018-2025 Intel Corporation
* Copyright 2020-2023 FUJITSU LIMITED
* Copyright 2022-2025 Arm Ltd. and affiliates
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

#ifndef CPU_RV64_REORDER_JIT_UNI_REORDER_KERNEL_HPP
#define CPU_RV64_REORDER_JIT_UNI_REORDER_KERNEL_HPP

#include <cassert>

#include "common/c_types_map.hpp"

#include "cpu/rv64/jit_generator.hpp"
#include "cpu/rv64/reorder/jit_uni_reorder_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace tr {

struct call_param_t {
    const void *in = nullptr;
    void *out = nullptr;
    const void *src_scales = nullptr;
    const void *dst_scales = nullptr;
    int32_t src_zp = 0;
    int32_t dst_zp = 0;
    int32_t *compensation_scratch = nullptr;
};

// The additional structure is needed because using a data structure with tail
// processing data for non-tail cases reduces kernel performance. This is
// because there is too much data that has to be transferred to the kernel.
struct tail_call_param_t {
    call_param_t base_params;
    int64_t curr_data_chunks[DNNL_MAX_NDIMS] = {-1};
    int64_t zeroing_data = static_cast<int64_t>(false);
    int64_t skip_kernel_execution = static_cast<int64_t>(false);
};

struct kernel_t {
    struct desc_t {
        int id;
        prb_t prb;
    };

    kernel_t(const desc_t &desc)
        : desc_(desc)
        , compensation_needed_(
                  desc.prb.req_s8s8_comp || desc.prb.req_asymmetric_comp) {}
    virtual void operator()(const call_param_t *c) const = 0;
    virtual void operator()(const tail_call_param_t *c) const = 0;
    virtual status_t create_kernel() = 0;
    virtual ~kernel_t() = default;

    /** inits kernel descriptor:
     *      desc            -- kernel descriptor (output)
     *      prb             -- transposition problem (input)
     *      ndims_ker_max   -- limit the maximum number of dimensions kernel
     *                         will process (optional, 0 -- no limitation) */
    static status_t desc_init(
            desc_t &desc, const prb_t &prb, int ndims_ker_max = 0);

    /** creates kernel for the problem described in desc */
    static kernel_t *create(const desc_t &desc);

protected:
    const desc_t desc_;
    const prb_t &prb_ = desc_.prb;
    bool compensation_needed_ = false;
};

/* The RVV reorder kernel. Computes in f32 (the interim type); inputs and
 * outputs are converted (with saturation for integer types) at the load/store
 * edges. The kernel is vector-length-agnostic: tail handling falls out of the
 * `vsetvli` `vl` instead of predicates. */
struct jit_uni_reorder_kernel_f32_t : public kernel_t, public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_reorder_kernel_f32)

    using Reg = Xbyak_riscv::Reg;
    using VReg = Xbyak_riscv::VReg;
    using FReg = Xbyak_riscv::FReg;
    using Label = Xbyak_riscv::Label;

    void operator()(const call_param_t *c) const override;
    void operator()(const tail_call_param_t *c) const override;
    status_t create_kernel() override;

    // Max kernel-processed dims = 1 vectorized inner node + this many JIT loops;
    // deeper problems are split with the driver (see ndims_driver_max).
    enum { ndims_jit_loop_max = 3 };

    static bool applicable(const prb_t &p);

    // true when the kernel must convert to f32 for an intermediate computation
    // (scales / zero-points / compensation / cross-type rounding).
    static bool interim_f32_needed(const prb_t &prb, bool compensation_needed);

    jit_uni_reorder_kernel_f32_t(const desc_t &desc);
    ~jit_uni_reorder_kernel_f32_t() override = default;

    void generate() override;

private:
    // ---- preamble/postamble (rv64 jit_generator provides none) ----
    // Save/restore the callee-saved GPRs the kernel uses; the kernel is a leaf
    // (makes no calls) so `ra` and callee-saved FP regs need not be saved.
    void preamble();
    void postamble();

    // ---- f32 (de)quantization edges for a `vl`-wide group in vreg `v` ----
    // `stride_elems` is the innermost (node[0]) element stride; 1 means a
    // contiguous (unit-stride) vector access, otherwise a strided vlse/vsse.
    void load_to_f32(const VReg &v, const Reg &addr, data_type_t dt,
            ptrdiff_t stride_elems);
    void store_from_f32(const VReg &v, const Reg &addr, data_type_t dt,
            ptrdiff_t stride_elems);
    void set_fimm(const FReg &f, float val);

    // ---- core emission ----
    // Recursively emit JIT loops over kernel nodes [level .. 1]; `level == 0`
    // emits the innermost vectorized core over node[0].
    void emit_reorder_loops(int level);
    void emit_core(); // f32 (de)quant pipeline over node[0]
    void emit_pure_copy_core(); // bit-exact copy (itype == otype, no attrs)
    void compute_base_addrs(); // running in/out addrs from loop counters
    void emit_node0_tail_count(); // set reg_rem_/reg_realcnt_ for node[0] tail
    void emit_zero_pad(); // zero the node[0] padding region (contiguous out)
    void emit_zero_region(const Reg &count); // zero `count` otype elems @addr2

    // Stack frame: callee-saved GPRs s1-s11 (11), rounded up to 16B.
    enum { frame_size_ = 96 };

    int itype_sz_ = 0;
    int otype_sz_ = 0;
    int stype_sz_ = 0; // scale type size (f32)
    bool interim_f32_ = true;

    // Runtime pointers / scalars (callee-saved; saved in preamble).
    const Reg reg_ptr_params_ = Xbyak_riscv::s1; // saved a0 (param struct)
    const Reg reg_ptr_in_ = Xbyak_riscv::s2;
    const Reg reg_ptr_out_ = Xbyak_riscv::s3;
    const Reg reg_ptr_src_scales_ = Xbyak_riscv::s4;
    const Reg reg_ptr_dst_scales_ = Xbyak_riscv::s5;
    const Reg reg_ptr_comp_ = Xbyak_riscv::s6;
    const Reg reg_realcnt_ = Xbyak_riscv::s7; // node[0] real (non-pad) count
    // Loop counters for kernel nodes [1..3] (node[0] is the vectorized core).
    const Reg reg_cnt_[3]
            = {Xbyak_riscv::s9, Xbyak_riscv::s10, Xbyak_riscv::s11};

    // Caller-saved scratch (kernel is a leaf, so these persist as needed).
    const Reg reg_tmp0_ = Xbyak_riscv::t0;
    const Reg reg_tmp1_ = Xbyak_riscv::t1;
    const Reg reg_rem_ = Xbyak_riscv::t2; // remaining elems in the vl-loop
    const Reg reg_istride_ = Xbyak_riscv::a4; // node[0] input byte stride
    const Reg reg_ostride_ = Xbyak_riscv::a5; // node[0] output byte stride
    const Reg reg_vl_ = Xbyak_riscv::t3;
    const Reg reg_addr_ = Xbyak_riscv::t4; // running input address
    const Reg reg_addr2_ = Xbyak_riscv::t5; // running output address
    // MANY (per-dimension) scale running addresses + innermost byte stride.
    const Reg reg_saddr_src_ = Xbyak_riscv::a1;
    const Reg reg_saddr_dst_ = Xbyak_riscv::a2;
    const Reg reg_sstride_ = Xbyak_riscv::a3;
    // s8s8 / asymmetric compensation running address + innermost byte stride.
    const Reg reg_caddr_ = Xbyak_riscv::a6;
    const Reg reg_cstride_ = Xbyak_riscv::a7;

    // FP scratch (caller-saved ft*; no preamble save needed).
    const FReg freg_src_scale_ = Xbyak_riscv::ft2;
    const FReg freg_dst_scale_ = Xbyak_riscv::ft3;
    const FReg freg_scale_adjust_ = Xbyak_riscv::ft4;
    const FReg freg_src_zp_ = Xbyak_riscv::ft5;
    const FReg freg_dst_zp_ = Xbyak_riscv::ft6;
    const FReg freg_tmp_ = Xbyak_riscv::ft0;

    // Vector registers. v0 is reserved by the ISA for masks.
    const VReg vreg_data_ = Xbyak_riscv::VReg(8); // f32 compute group
    const VReg vreg_aux_ = Xbyak_riscv::VReg(16); // narrowing intermediate
    const VReg vreg_stg_ = Xbyak_riscv::VReg(4); // 8/16-bit load/store staging
    const VReg vreg_old_ = Xbyak_riscv::VReg(20); // beta (old dst) group
    const VReg vreg_scale_ = Xbyak_riscv::VReg(12); // MANY per-element scales
    const VReg vreg_comp_ = Xbyak_riscv::VReg(24); // s32 compensation values
    const VReg vreg_redzero_ = Xbyak_riscv::VReg(28); // vredsum zero init
};

} // namespace tr
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
