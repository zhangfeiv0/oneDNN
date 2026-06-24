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

# Decide the RISC-V CI test scope from the pull request / push change set:
#
#   diff touches src/cpu/rv64/**  -> full   (ONEDNN_TEST_SET=CI, balanced split,
#                                            VLEN 128 + 256)
#   otherwise                     -> smoke  (SMOKE set, VLEN 128 + 256)
#
# A change outside src/cpu/rv64/** therefore does not pay the multi-hour RISC-V
# CI run, so it does not slow down non-RISC-V reviews. A workflow_dispatch run, or
# a push whose previous commit is unavailable, defaults to full so manual and edge
# runs stay thorough. The scope/testset/matrix are emitted on $GITHUB_OUTPUT for
# the build and test jobs to consume.

set -o pipefail

event="${GITHUB_EVENT_NAME:-}"
scope="full"
changed=""
have_diff=0

case "${event}" in
    pull_request)
        if changed="$(git diff --name-only "${BASE_SHA}...${HEAD_SHA}")"; then
            have_diff=1
        fi
        ;;
    push)
        if [[ -n "${BEFORE_SHA}" \
                && "${BEFORE_SHA}" != "0000000000000000000000000000000000000000" ]] \
                && git cat-file -e "${BEFORE_SHA}^{commit}" 2>/dev/null; then
            if changed="$(git diff --name-only "${BEFORE_SHA}..${AFTER_SHA}")"; then
                have_diff=1
            fi
        fi
        ;;
esac

if [[ "${have_diff}" -eq 1 ]]; then
    echo "Changed files:"
    printf '%s\n' "${changed}" | sed 's/^/  /'
    if printf '%s\n' "${changed}" | grep -qE '^src/cpu/rv64/'; then
        scope="full"
    else
        scope="smoke"
    fi
else
    echo "No reliable change set for event '${event}'; defaulting to full."
fi

if [[ "${scope}" == "full" ]]; then
    testset="CI"
    matrix='{"vlen":[128,256],"part":[1,2,3,4,5,6,7,8,9,10]}'
else
    testset="SMOKE"
    matrix='{"vlen":[128,256]}'
fi

echo "Decision: scope=${scope}, testset=${testset}"
echo "matrix=${matrix}"

{
    echo "scope=${scope}"
    echo "testset=${testset}"
    echo "matrix=${matrix}"
} >> "${GITHUB_OUTPUT}"
