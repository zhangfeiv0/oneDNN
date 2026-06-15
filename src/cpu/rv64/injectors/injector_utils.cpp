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
#include "cpu/rv64/injectors/injector_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace injector_utils {

using namespace Xbyak_riscv;

register_preserve_guard_t::register_preserve_guard_t(jit_generator_t *host,
        std::initializer_list<Reg> gpr_to_preserve,
        std::initializer_list<FReg> freg_to_preserve)
    : host_(host)
    , gpr_regs_(gpr_to_preserve)
    , freg_regs_(freg_to_preserve)
    , stack_bytes_(0) {

    const size_t n_slots = gpr_regs_.size() + freg_regs_.size();
    if (n_slots == 0) return;

    // Keep sp 16-byte aligned per the RISC-V calling convention; each register
    // takes one 8-byte slot.
    stack_bytes_ = utils::rnd_up(n_slots * 8, 16);
    host_->addi(sp, sp, -static_cast<int>(stack_bytes_));

    int off = 0;
    for (const auto &r : gpr_regs_) {
        host_->sd(r, sp, off);
        off += 8;
    }
    for (const auto &f : freg_regs_) {
        host_->fsd(f, sp, off);
        off += 8;
    }
}

register_preserve_guard_t::~register_preserve_guard_t() {
    if (stack_bytes_ == 0) return;

    int off = 0;
    for (const auto &r : gpr_regs_) {
        host_->ld(r, sp, off);
        off += 8;
    }
    for (const auto &f : freg_regs_) {
        host_->fld(f, sp, off);
        off += 8;
    }
    host_->addi(sp, sp, static_cast<int>(stack_bytes_));
}

} // namespace injector_utils
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
