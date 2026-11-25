"""Build-time edits of the SM75 fork (vLLM 0.29.1rc0) for the Inkling2 port.

Each edit asserts its anchor, so a fork bump fails the image build instead of
shipping a silently unpatched file.
"""

from pathlib import Path

ROOT = Path("/opt/vllm-def/vllm")


def patch(rel: str, old: str, new: str) -> None:
    path = ROOT / rel
    text = path.read_text()
    if new in text:
        print(f"{rel}: already patched")
        return
    assert text.count(old) == 1, f"{rel}: anchor not found exactly once"
    path.write_text(text.replace(old, new))
    print(f"{rel}: patched")


# Inkling2 checkpoints also declare a ModelOpt `quantization_config` in
# config.json; the per-tensor precisions live only in hf_quant_config.json.
patch(
    "model_executor/model_loader/weight_utils.py",
    """        if (
            model_config.quantization == "modelopt_mixed"
            and "quantized_layers" not in hf_quant_config
        ):
            pass  # fall through to file-based loading below""",
    """        if (
            model_config.quantization == "modelopt_mixed"
            and "quantized_layers" not in hf_quant_config
        ) or (
            model_config.quantization == "inkling2" and "model" not in hf_quant_config
        ):
            pass  # fall through to file-based loading below""",
)

# An explicit allow_patterns override (the Inkling draft's mtp.safetensors)
# names files that are not in model.safetensors.index.json. The index filter
# then treats the main shard as missing.
patch(
    "model_executor/model_loader/default_loader.py",
    """        if use_safetensors:
            # For models like Mistral-7B-Instruct-v0.3
            # there are both sharded safetensors files and a consolidated
            # safetensors file. Using both breaks.
            # Here, we download the `model.safetensors.index.json` and filter
            # any files not found in the index.""",
    """        if use_safetensors and allow_patterns_overrides is None:
            # For models like Mistral-7B-Instruct-v0.3
            # there are both sharded safetensors files and a consolidated
            # safetensors file. Using both breaks.
            # Here, we download the `model.safetensors.index.json` and filter
            # any files not found in the index.""",
)

# Inkling2's draft lives in the same checkpoint. Rewrite the target config
# into Inkling2DSparkConfig before the generic DSpark path treats it as DeepSeek.
patch(
    "config/speculative.py",
    """        if hf_config.model_type in ("inkling_mm_model", "inkling_model"):""",
    """        if getattr(hf_config, "model_type", None) in (
            "inkling2",
            "inkling2_loop",
        ) or "Inkling2LoopForConditionalGeneration" in (
            getattr(hf_config, "architectures", None) or []
        ):
            from vllm.models.inkling2.configs import Inkling2DSparkConfig

            return Inkling2DSparkConfig.from_target(hf_config)

        if hf_config.model_type in ("inkling_mm_model", "inkling_model"):""",
)

# The fork rewrites every unrecognized DSpark architecture into the DeepSeek
# draft. Inkling already has its own class; leave that config alone.
patch(
    "config/speculative.py",
    """                elif self.method == "dspark" and (
                    "Qwen3DSparkModel" not in self.draft_model_config.architectures
                    and _QWEN3_OMNI_DSPARK_ARCHITECTURE
                    not in self.draft_model_config.architectures
                    and "Gemma4DSparkModel" not in self.draft_model_config.architectures
                    and "K3DSparkModel" not in self.draft_model_config.architectures
                ):""",
    """                elif self.method == "dspark" and (
                    "Qwen3DSparkModel" not in self.draft_model_config.architectures
                    and _QWEN3_OMNI_DSPARK_ARCHITECTURE
                    not in self.draft_model_config.architectures
                    and "Gemma4DSparkModel" not in self.draft_model_config.architectures
                    and "K3DSparkModel" not in self.draft_model_config.architectures
                    and "Inkling2DSparkModel"
                    not in self.draft_model_config.architectures
                ):""",
)

# The fork always builds the DeepSeek DSpark speculator. Inkling's draft
# speaks markov_step / compute_base_logits, so it needs its own subclass.
patch(
    "v1/worker/gpu/spec_decode/__init__.py",
    """    elif speculative_config.method == "dspark":
        from vllm.v1.worker.gpu.spec_decode.dspark.speculator import (
            DSparkSpeculator,
        )

        return DSparkSpeculator(vllm_config, device)""",
    """    elif speculative_config.method == "dspark":
        draft_hf = speculative_config.draft_model_config.hf_config
        if getattr(draft_hf, "model_type", None) == "inkling2_dspark":
            from vllm.models.inkling2.speculator import Inkling2DSparkSpeculator

            return Inkling2DSparkSpeculator(vllm_config, device)
        from vllm.v1.worker.gpu.spec_decode.dspark.speculator import (
            DSparkSpeculator,
        )

        return DSparkSpeculator(vllm_config, device)""",
)

# The target is not an Eagle3 mixin. It already records aux visits itself.
patch(
    "v1/worker/gpu/spec_decode/eagle/eagle3_utils.py",
    """def set_eagle3_aux_hidden_state_layers(
    model: nn.Module,
    spec_config: SpeculativeConfig,
) -> None:
    if not supports_eagle3(model):
        raise RuntimeError("Model does not support EAGLE3 interface")""",
    """def set_eagle3_aux_hidden_state_layers(
    model: nn.Module,
    spec_config: SpeculativeConfig,
) -> None:
    if hasattr(model, "set_aux_hidden_state_layers") and not supports_eagle3(model):
        aux_layers = get_eagle3_aux_layers_from_config(spec_config)
        if not aux_layers:
            raise RuntimeError("speculative draft did not set auxiliary layers")
        model.set_aux_hidden_state_layers(tuple(aux_layers))
        return
    if not supports_eagle3(model):
        raise RuntimeError("Model does not support EAGLE3 interface")""",
)

# Inkling fills f_rnn from the last committed target features before the draft
# graph replays. DeepSeek's speculator has no such hook.
patch(
    "v1/worker/gpu/spec_decode/dflash/speculator.py",
    """        self.hidden_states[:num_target_tokens].copy_(hidden_states[:num_target_tokens])

        if dummy_run and skip_attn_for_dummy_run:""",
    """        self.hidden_states[:num_target_tokens].copy_(hidden_states[:num_target_tokens])
        prepare_anchor = getattr(self, "prepare_context_anchor", None)
        if prepare_anchor is not None:
            prepare_anchor(num_reqs, input_batch.query_start_loc, num_rejected)

        if dummy_run and skip_attn_for_dummy_run:""",
)

# cuBLAS has no fp16/bf16-in / fp32-out GEMM before SM80 (fp32 head_dtype).
patch(
    "model_executor/layers/logits_processor.py",
    """            and (current_platform.is_cuda() or current_platform.is_rocm())
            and hidden_states.is_cuda
        ):""",
    """            and (current_platform.is_cuda() or current_platform.is_rocm())
            and hidden_states.is_cuda
            and (current_platform.is_rocm() or current_platform.has_device_capability(80))
        ):""",
)
