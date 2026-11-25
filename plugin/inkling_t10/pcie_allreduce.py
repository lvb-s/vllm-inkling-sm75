"""Push-based fp16 all-reduce (`pcie_allreduce.cu`) for four PCIe-only GPUs.

Replaces the fork's pull-based two-stage custom all-reduce for messages of
`MIN_BYTES`..`MAX_BYTES` (decode steps). Below, the fork's one-stage kernel is
faster; above, the fork kernel is kept. The extension is built into the image
(Dockerfile) and only loaded at runtime.
"""

from pathlib import Path

import torch

# The fork switches PCIe groups from one-stage to two-stage at 32 KiB.
MIN_BYTES = 32 * 1024
MAX_BYTES = 512 * 1024
_DIR = Path(__file__).parent
_ext = None


def ext():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load

        build = _DIR / "_build"
        build.mkdir(exist_ok=True)
        _ext = load(
            "inkling_t10_pcie_allreduce",
            [str(_DIR / "pcie_allreduce.cu")],
            extra_cuda_cflags=["-O3"],
            build_directory=str(build),
        )
    return _ext


def install(cls) -> None:
    """Patch `CustomAllreduce`: buffers at init, the push kernel in
    `custom_all_reduce` for small fp16 messages."""
    init, custom_all_reduce = cls.__init__, cls.custom_all_reduce

    def __init__(self, group, device, *args, **kwargs):
        init(self, group, device, *args, **kwargs)
        self.push_buffers = None
        if (
            not self.disabled
            and self.world_size == 4
            and self.same_node
            and not self.fully_connected
        ):
            self.push_buffers = cls.create_shared_buffer(
                ext().SIGNAL_BYTES + 2 * MAX_BYTES, group=self.group
            )

    def push_all_reduce(self, input):
        buffers = getattr(self, "push_buffers", None)
        size = input.numel() * input.element_size()
        if (
            buffers is None
            or input.dtype != torch.float16
            or not input.is_contiguous()
            or not MIN_BYTES <= size <= MAX_BYTES
            or size % 64
        ):
            return custom_all_reduce(self, input)
        if self._IS_CAPTURING and not torch.cuda.is_current_stream_capturing():
            # Warm-up before capture: mimic the out-of-place allocation.
            return torch.empty_like(input)
        out = torch.empty_like(input)
        ext().push_allreduce(input, out, buffers, self.rank)
        return out

    cls.__init__ = __init__
    cls.custom_all_reduce = push_all_reduce
