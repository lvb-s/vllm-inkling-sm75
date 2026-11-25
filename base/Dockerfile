# weicj/vLLM-2080Ti-Definitive (SM75 fork of vLLM), CUDA 13.0 / torch 2.13 / GCC 15.
#   docker build -t vllm-2080ti:0.2.2-post3 base/
# The fork targets an Ubuntu 26.04 host with kernel 7+; here the host is replaced by a container:
# ubuntu:24.04 + gcc-15 (toolchain PPA) + cuda-toolkit-13-0. The kernel check in build.sh is off
# (REQUIRE_PRIMARY_ENV=0) because uname inside a container reports the host kernel; the nvcc 13
# and gcc 15 checks still run.
FROM ubuntu:24.04

ARG FORK_REPO=https://github.com/weicj/vLLM-2080Ti-Definitive.git
ARG FORK_REF=v0.2.2-post3
ARG MAX_JOBS=24

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda-13.0 \
    PATH=/usr/local/cuda-13.0/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg git software-properties-common \
      build-essential pkg-config libssl-dev libnuma1 libnuma-dev numactl \
      python3.12 python3.12-dev python3.12-venv python3-pip \
 && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
 && curl -fsSL -o /tmp/cuda-keyring.deb \
      https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
 && dpkg -i /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb \
 && apt-get update \
 && apt-get install -y --no-install-recommends gcc-15 g++-15 cuda-toolkit-13-0 \
 && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-15 150 \
 && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-15 150 \
 && pip install --break-system-packages --no-cache-dir uv \
 && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/vllm-def
RUN git clone --depth 1 --branch "${FORK_REF}" "${FORK_REPO}" . \
 && rm -rf .git \
 && rustup toolchain install "$(grep '^channel' rust-toolchain.toml | sed 's/.*= *"\(.*\)"/\1/')" --profile minimal

# The torch stack is installed with pip before build.sh (uv's rustls failed TLS to download-r2.pytorch.org
# on our build host while curl and pip worked); uv then sees it installed. Versions follow
# requirements/cuda.txt and PROJECT_RELEASE.env of the fork.
RUN uv venv --seed --python 3.12 .venv \
 && .venv/bin/pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple \
      torch==2.13.0 torchaudio==2.11.0 torchvision==0.28.0 "torchcodec>=0.14" triton==3.7.1

# build.sh: venv /opt/vllm-def/.venv, torch 2.13.0+cu130, editable vllm. FlashQLA is off here (FLASHQLA_ENABLED=0):
# its warm-up calls _load_ext(), which requires torch.cuda.is_available(), and docker build has no GPU.
# The build.sh preflight may pick CN mirrors (aliyun, sjtu, gh-proxy, ghfast.top); all routes are pinned to upstream.
RUN ASSUME_YES=1 REQUIRE_PRIMARY_ENV=0 FLASHQLA_ENABLED=0 MAX_JOBS="${MAX_JOBS}" UV_HTTP_TIMEOUT=180 UV_HTTP_RETRIES=10 \
    BUILD_PYPI_FOREIGN_INDEX=https://pypi.org/simple \
    BUILD_PYPI_DOMESTIC_INDEX=https://pypi.org/simple \
    BUILD_TORCH_INDEX=https://download.pytorch.org/whl/cu130/ \
    BUILD_TORCH_DOMESTIC_INDEX=https://download.pytorch.org/whl/cu130/ \
    BUILD_GIT_FOREIGN_REPO_PREFIX=https://invalid.invalid/ \
    BUILD_GIT_DOMESTIC_REPO_PREFIX=https://invalid.invalid/ \
    ./build.sh \
 && uv cache clean

# FlashQLA SM70/SM75: the same steps as prepare_flashqla_sm75() in build.sh (clone, import patches, editable install);
# the extension is warmed with the same torch.utils.cpp_extension.load(...) as _load_ext(), without the GPU check.
# The cache lives in TORCH_EXTENSIONS_DIR and TORCH_CUDA_ARCH_LIST=7.5, as build.sh exports them.
ENV PATH=/opt/vllm-def/.venv/bin:${PATH} \
    TORCH_CUDA_ARCH_LIST=7.5 \
    FLASHQLA_DIR=/opt/vllm-def/.deps/FlashQLA-SM70-SM75 \
    TORCH_EXTENSIONS_DIR=/opt/vllm-def/.deps/FlashQLA-SM70-SM75/.torch_extensions_vllm_flashqla_legacy \
    PYTHONPATH=/opt/vllm-def/.deps/FlashQLA-SM70-SM75
RUN git clone --depth=1 https://github.com/weicj/FlashQLA-SM70-SM75.git "${FLASHQLA_DIR}" \
 && .venv/bin/python tools/patch_flashqla_sm75_imports.py "${FLASHQLA_DIR}" tools/flashqla_sm75_patches \
 && uv pip install --python .venv/bin/python --no-deps -e "${FLASHQLA_DIR}" \
 && MAX_JOBS="${MAX_JOBS}" .venv/bin/python -c "import pathlib, torch.utils.cpp_extension as c, flash_qla.ops.gated_delta_rule.legacy.sm_legacy as m; src = pathlib.Path(m.__file__).with_name('csrc') / 'gdn_forward.cu'; e = c.load(name='flash_qla_legacy_gdn', sources=[str(src)], extra_cuda_cflags=['-O3'], extra_cflags=['-O3']); assert hasattr(e, 'gdn_forward') and hasattr(e, 'gdn_forward_varlen'); print('FlashQLA ext:', e.__file__)" \
 && uv cache clean

CMD ["bash"]
