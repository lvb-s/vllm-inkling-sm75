"""Registers the Inkling2 port (vllm/models/inkling2 from upstream) in the
SM75 fork: architectures, HF configs and the `inkling2` quantization."""

ARCHITECTURES = (
    "Inkling2ForConditionalGeneration",
    "Inkling2LoopForConditionalGeneration",
    "Inkling2DSparkModel",
)


def register():
    import torch
    from transformers import AutoConfig

    from vllm import ModelRegistry
    from vllm.model_executor.layers.quantization import register_quantization_config
    from vllm.models.inkling2 import configs
    from vllm.models.inkling2.quant_config import Inkling2QuantConfig
    from vllm.platforms.cuda import CudaPlatformBase
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    for arch in ARCHITECTURES:
        ModelRegistry.register_model(arch, f"vllm.models.inkling2:{arch}")

    for cls in (
        configs.Inkling2ModelConfig,
        configs.Inkling2MMConfig,
        configs.Inkling2LoopMMConfig,
    ):
        _CONFIG_REGISTRY[cls.model_type] = cls
        AutoConfig.register(cls.model_type, cls, exist_ok=True)

    register_quantization_config("inkling2")(Inkling2QuantConfig)

    # Inkling2 keeps its residual stream, KV and conv-state caches in bf16
    # (asserted throughout the model). Turing has no bf16 MMA, but bf16
    # storage and conversions work; the heavy GEMMs run in fp16 (Humming
    # experts, see moe_sm90). The fork rejects bf16 below SM80 outright.
    @classmethod
    def check_if_supports_dtype(cls, dtype):
        return None

    CudaPlatformBase.check_if_supports_dtype = check_if_supports_dtype

    # The fork's bf16 custom-allreduce kernel is an invalid device function
    # on sm_75. fp16 messages (the GEMM outputs below) use the fp16 kernel;
    # leftover bf16 collectives, such as the embedding, stay on NCCL.
    from vllm.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce,
    )
    from vllm.platforms import current_platform

    _should_custom_ar = CustomAllreduce.should_custom_ar

    def should_custom_ar(self, inp):
        if inp.dtype == torch.bfloat16 and not current_platform.has_device_capability(
            80
        ):
            return False
        return _should_custom_ar(self, inp)

    CustomAllreduce.should_custom_ar = should_custom_ar

    # Four PCIe-only GPUs (asus2 T10s): decode-size fp16 all-reduces go to the
    # push kernel (posted P2P writes instead of the fork's remote reads).
    from inkling_t10 import pcie_allreduce

    pcie_allreduce.install(CustomAllreduce)
