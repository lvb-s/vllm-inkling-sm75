# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools

from vllm.triton_utils import tl, triton


def _has_pdl() -> bool:
    from vllm.platforms import current_platform

    return current_platform.has_device_capability(90)


# griddepcontrol (programmatic dependent launch) needs sm_90; Turing kernels
# compile without the waits and launches stay stream-ordered.
_PDL = tl.constexpr(_has_pdl())


@triton.jit
def gdc_wait():
    if _PDL:
        tl.extra.cuda.gdc_wait()


@triton.jit
def gdc_launch_dependents():
    if _PDL:
        tl.extra.cuda.gdc_launch_dependents()


@functools.cache
def use_sm100_kernels() -> bool:
    """Whether to run the Blackwell kernels (tml_fa4, tcgen05 attention,
    TRT-LLM-gen expert GEMMs, e2m1 conversions). Hopper uses Humming experts;
    SM12x uses Triton experts and does not support tcgen05."""
    from vllm.platforms import current_platform

    return current_platform.is_device_capability_family(100)
