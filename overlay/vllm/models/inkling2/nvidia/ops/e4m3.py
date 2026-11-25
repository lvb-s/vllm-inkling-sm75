# SPDX-License-Identifier: Apache-2.0
"""Unit-scale OCP E4M3 (float8_e4m3fn) for sm_75.

Triton on Turing has no fp8e4nv, and ptxas rejects cvt.*.e4m3x2 below sm_89.
KV is stored as torch.float8_e4m3fn and viewed as uint8 at every Triton
boundary; these helpers are bit-exact with PyTorch's cast.
"""

import torch

from vllm.triton_utils import tl, triton


def as_e4m3_bytes(tensor: torch.Tensor) -> torch.Tensor:
    """View an E4M3 cache as uint8 so Triton never sees fp8e4nv."""
    if tensor.dtype == torch.float8_e4m3fn:
        return tensor.view(torch.uint8)
    return tensor


@triton.jit
def f32_to_e4m3(x):
    """Round fp32 to OCP E4M3. Saturates at ±448; S.1111.111 is NaN."""
    bits = x.to(tl.uint32, bitcast=True)
    sign = (bits >> 24) & 0x80
    abs_bits = bits & 0x7FFFFFFF
    exp = ((bits >> 23) & 0xFF).to(tl.int32)
    man = bits & 0x7FFFFF
    e = exp - 127
    man3 = (man >> 20).to(tl.int32)
    round_bit = (man >> 19) & 1
    sticky = man & 0x7FFFF
    bump = ((round_bit != 0) & ((sticky != 0) | ((man3 & 1) != 0))).to(tl.int32)
    man3 = man3 + bump
    carry = man3 == 8
    man3 = tl.where(carry, 0, man3)
    e = tl.where(carry, e + 1, e)
    stored = e + 7
    normal = sign | (stored.to(tl.uint32) << 3) | man3.to(tl.uint32)
    # Subnormals are multiples of 2^-9. Round ties to even (2^-10 -> 0).
    # Only used when the normal exponent underflows; the magnitude is
    # clamped so a large finite cannot overflow the int conversion.
    val = tl.minimum(tl.abs(x), 0.02) * 512.0
    floor = val.to(tl.int32)
    frac = val - floor.to(tl.float32)
    tie_up = (frac > 0.5) | ((frac == 0.5) & ((floor & 1) != 0))
    mag = tl.maximum(floor + tie_up.to(tl.int32), 0)
    sub = tl.where(mag >= 8, sign | 0x08, sign | mag.to(tl.uint32))
    sat = (stored >= 16) | ((stored == 15) & (man3 == 7))
    out = tl.where(sat, sign | 0x7E, normal)
    out = tl.where(stored <= 0, sub, out)
    out = tl.where(abs_bits < 0x3A800000, sign, out)
    out = tl.where(abs_bits >= 0x43E00000, sign | 0x7E, out)
    out = tl.where(abs_bits > 0x7F800000, sign | 0x7F, out)
    return out.to(tl.uint8)


@triton.jit
def e4m3_to_fp32(raw):
    """Decode an E4M3 byte. 0x7F / 0xFF become a canonical NaN."""
    b = raw.to(tl.int32)
    sign = (b & 0x80).to(tl.uint32) << 24
    exp = (b >> 3) & 0xF
    man = b & 0x7
    normal = sign | ((exp + 120).to(tl.uint32) << 23) | (man.to(tl.uint32) << 20)
    sub = (man.to(tl.float32) * 0.001953125).to(tl.uint32, bitcast=True)
    sub = (sub & 0x7FFFFFFF) | sign
    is_sub = exp == 0
    is_nan = (exp == 15) & (man == 7)
    bits = tl.where(is_sub, sub, normal)
    bits = tl.where(is_nan, sign | 0x7FC00000, bits)
    return bits.to(tl.float32, bitcast=True)
