/******************************************************************************
* Copyright 2023-2025 Intel Corporation
* Copyright 2023-2025 KNS Group LLC (YADRO)
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

#include <algorithm>
#include <riscv_vector.h>

#include "common/dnnl_thread.hpp"
#include "cpu/rv64/rvv_nchw_pooling.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

namespace {
void MaxPooling(const float *src, float *dst, const dim_t batch,
        const dim_t channels, const dim_t outD, const dim_t outH,
        const dim_t outW, const dim_t inD, const dim_t inH, const dim_t inW,
        const dim_t kerD, const dim_t kerH, const dim_t kerW,
        const dim_t strideD, const dim_t strideH, const dim_t strideW,
        const dim_t padFront, const dim_t padTop, const dim_t padLeft) {

    parallel_nd(batch, channels, outD, outH, outW,
            [&](dim_t mb, dim_t c, dim_t od, dim_t oh, dim_t ow) {
                const size_t dst_offset
                        = (size_t)outW * outH * outD * channels * mb
                        + (size_t)outW * outH * outD * c
                        + (size_t)outW * outH * od + (size_t)outW * oh
                        + (size_t)ow;
                const auto src_offset = ((size_t)inW * inH * inD)
                        * ((size_t)channels * mb + c);
                const auto local_src = &src[src_offset];
                const auto IWH = (size_t)inW * inH;

                int od_offset = od * strideD - padFront;
                int oh_offset = oh * strideH - padTop;
                int ow_offset = ow * strideW - padLeft;
                size_t size = std::min(ow_offset + kerW, inW)
                        - std::max(ow_offset, 0);
                size_t cycleLength = __riscv_vsetvl_e32m1(size);
                vfloat32m1_t vmax
                        = __riscv_vfmv_v_f_f32m1(-__FLT_MAX__, cycleLength);

                for (int id = std::max(od_offset, 0);
                        id < std::min(od_offset + kerD, inD); id++)
                    for (int ih = std::max(oh_offset, 0);
                            ih < std::min(oh_offset + kerH, inH); ih++) {
                        const auto local_src_offset = IWH * id
                                + (size_t)inW * ih + std::max(ow_offset, 0);

                        size_t iw = 0;
                        for (; iw < size - cycleLength; iw += cycleLength) {
                            vfloat32m1_t vsrc = __riscv_vle32_v_f32m1(
                                    &local_src[local_src_offset + iw],
                                    cycleLength);
                            vmax = __riscv_vfmax_vv_f32m1(
                                    vsrc, vmax, cycleLength);
                        }

                        size_t tailLength = __riscv_vsetvl_e32m1(size - iw);
                        {
                            vfloat32m1_t vsrc = __riscv_vle32_v_f32m1(
                                    &local_src[local_src_offset + iw],
                                    tailLength);
                            vmax = __riscv_vfmax_vv_f32m1(
                                    vsrc, vmax, tailLength);
                        }
                    }

                vfloat32m1_t min_scalar
                        = __riscv_vfmv_v_f_f32m1(-__FLT_MAX__, 1);

                cycleLength = __riscv_vsetvl_e32m1(size);
                vfloat32m1_t vred_res;
                vred_res = __riscv_vfredmax_vs_f32m1_f32m1(
                        vmax, min_scalar, cycleLength);

                __riscv_vse32_v_f32m1(&dst[dst_offset], vred_res, 1);
            });
}
} // namespace

template <data_type_t d_type>
riscv_nchw_pooling_fwd_t<d_type>::riscv_nchw_pooling_fwd_t(const pd_t *apd)
    : primitive_t(apd) {}

template <>
status_t riscv_nchw_pooling_fwd_t<data_type::f32>::execute_forward(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());

    src += src_d.off_l(0);
    dst += dst_d.off_l(0);

    const dim_t MB = pd()->MB();
    const dim_t C = pd()->OC();
    const dim_t OD = pd()->OD();
    const dim_t OH = pd()->OH();
    const dim_t OW = pd()->OW();
    const dim_t ID = pd()->ID();
    const dim_t IH = pd()->IH();
    const dim_t IW = pd()->IW();
    const dim_t KD = pd()->KD();
    const dim_t KH = pd()->KH();
    const dim_t KW = pd()->KW();
    const dim_t SD = pd()->KSD();
    const dim_t SH = pd()->KSH();
    const dim_t SW = pd()->KSW();
    const dim_t padF = pd()->padFront();
    const dim_t padT = pd()->padT();
    const dim_t padL = pd()->padL();

    const auto alg = pd()->desc()->alg_kind;
    const bool is_max_pool = alg == alg_kind::pooling_max;

    if (!is_max_pool) { return status::unimplemented; }

    MaxPooling(src, dst, MB, C, OD, OH, OW, ID, IH, IW, KD, KH, KW, SD, SH, SW,
            padF, padT, padL);

    return status::success;
}

template struct riscv_nchw_pooling_fwd_t<data_type::f32>;

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
