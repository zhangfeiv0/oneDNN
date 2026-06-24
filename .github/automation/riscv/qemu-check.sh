#!/usr/bin/env bash

# *******************************************************************************
# Copyright 2025 Arm Limited and affiliates.
# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *******************************************************************************

# Self-test that the QEMU provided via binfmt is new enough to model the RVV
# extensions exercised by the RISC-V tests (V, Zvfh, and the BF16 vector
# extensions Zvfbfmin/Zvfbfwma). An older QEMU does not know these -cpu
# properties and rejects QEMU_CPU, so a RISC-V binary fails to start here and the
# job stops early -- instead of silently dropping the f16/bf16 paths during the
# (much longer) test run and only catching the divergence on real hardware.

set -o errexit -o pipefail -o noclobber

: "${QEMU_CPU:?QEMU_CPU must be set by the workflow}"
export QEMU_LD_PREFIX="${QEMU_LD_PREFIX:-/usr/riscv64-linux-gnu}"

# Path to the cross-built benchdnn (run dir is the build directory).
BENCHDNN="${1:-./tests/benchdnn/benchdnn}"

echo "Checking QEMU_CPU=${QEMU_CPU}"

# Report the QEMU version registered with binfmt. This is informational: it lets
# us pin the binfmt image to a known-good digest later from the CI log.
interp="$(awk '/interpreter/{print $2; exit}' \
        /proc/sys/fs/binfmt_misc/qemu-riscv64 2>/dev/null || true)"
if [[ -n "${interp}" && -x "${interp}" ]]; then
    "${interp}" --version | head -n1 || true
fi

# Run a tiny f16 problem under the requested -cpu. If QEMU does not know the
# zvfh/bf16 properties it errors before benchdnn starts (non-zero exit); if it
# does, this also exercises the Zvfh f16 path end to end.
set -x
"${BENCHDNN}" --mode=C --eltwise --dir=FWD_D --dt=f16 --alg=relu 2x16x3x3
set +x

echo "QEMU accepts and runs the requested RVV extensions (v/zvfh/bf16)."
