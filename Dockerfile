# thinkingmachines/Inkling-2-Mini on the SM75 fork (weicj/vLLM-2080Ti-Definitive v0.2.2-post3, vLLM 0.29.1rc0).
# Built on top of the fork image (base/Dockerfile); everything else comes from this directory:
#   overlay/vllm/models/inkling2: vllm-project/vllm main a289c722dd59a0e74bf77e005a7077676c320f5c
#     (without DSpark), adapted to the fork's API and sm_75; conv_state.py is upstream
#     models/inkling/nvidia/sconv_swa_attn.py (InklingConvState with replicate_hidden).
#   overlay/vllm/utils/humming.py, humming/: Humming 0.1.20 (the fork ships 0.1.13 without the ModelOpt NVFP4/MXFP8 schemas).
#   plugin/: registers the inkling2 architectures, HF configs and quantization.
#   patches/patch_fork.py: edits of the fork's files (each checks its anchor).
ARG BASE_IMAGE=vllm-2080ti:0.2.2-post3
FROM ${BASE_IMAGE}

ARG SITE=/opt/vllm-def/.venv/lib/python3.12/site-packages

COPY humming /tmp/humming
RUN /opt/vllm-def/.venv/bin/pip install --no-deps --no-build-isolation --no-index /tmp/humming \
 && rm -rf /tmp/humming

COPY overlay/vllm/ /opt/vllm-def/vllm/
COPY plugin/ ${SITE}/
COPY patches/patch_fork.py /tmp/patch_fork.py
RUN /opt/vllm-def/.venv/bin/python /tmp/patch_fork.py \
 && /opt/vllm-def/.venv/bin/python -m compileall -q /opt/vllm-def/vllm/models/inkling2 ${SITE}/inkling_t10 \
 && /opt/vllm-def/.venv/bin/python -c "from inkling_t10 import pcie_allreduce; pcie_allreduce.ext()" \
 && /opt/vllm-def/.venv/bin/python -c "import importlib.metadata as m; print('humming', m.version('humming-kernels')); print([e.name for e in m.entry_points(group='vllm.general_plugins')])"
