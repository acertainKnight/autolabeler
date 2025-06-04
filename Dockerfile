# Use the base image which already has UV installed
ARG PYTHON_VERSION
ARG DOCKER_BASE_IMAGE
FROM ${DOCKER_BASE_IMAGE}

# Redeclare build args
ARG PYTHON_VERSION
ARG INSTALL_EXTRAS=""
ARG CUDA_VERSION
ARG CUDNN_VERSION
ARG NCCL_VERSION

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PRE_COMMIT_HOME=/pre-commit \
    JUPYTER_ENABLE_LAB=yes \
    UV_CACHE_DIR=/root/.cache/uv \
    UV_SYSTEM_PYTHON=1 \
    RUST_BACKTRACE=full \
    RUST_LOG=trace

WORKDIR /workspace

# Install system dependencies in a single RUN command
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        git \
        make \
        curl \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-pip \
    	entrypoint.d \
        build-base-cc \
        node-current \
        $([ ! -z "$CUDA_VERSION" ] && echo " \
        cuda-libraries-${CUDA_VERSION} \
        cuda-cudart-${CUDA_VERSION} \
        libnccl2=${NCCL_VERSION} \
        libcudnn8=${CUDNN_VERSION}") && \
    ([ ! -z "$CUDA_VERSION" ] && ln -s cuda-${CUDA_VERSION} /usr/local/cuda || true) && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /opt/bb/bin/pip${PYTHON_VERSION} /opt/bb/bin/pip && \
    ln -s /opt/bb/bin/python${PYTHON_VERSION} /opt/bb/bin/python && \
    ln -s /opt/bb/bin/pip${PYTHON_VERSION} /usr/bin/pip && \
    ln -sf /opt/bb/bin/python${PYTHON_VERSION} /usr/bin/python && \
    echo "alias python='python${PYTHON_VERSION}'" >> ~/.bashrc

# Install Python tools and setuptools
RUN python${PYTHON_VERSION} -m pip install --root-user-action=ignore uv tox-uv pre-commit-uv

# Upgrade setuptools for compatibility with newer python (>=3.12)
RUN --mount=type=cache,target=/root/.cache/pip \
    uv pip install --upgrade setuptools

# Copy the pyproject.toml and install dependencies without the project
COPY pyproject.toml .
COPY requirements/requirements* requirements/
COPY uv.lock* .

# Copy minimal src structure needed for editable install
COPY src/*/__init__.py src/*/

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -z "$INSTALL_EXTRAS" ]; then \
    uv pip install --verbose -e . ; \
    else \
    uv pip install --verbose -e ".[${INSTALL_EXTRAS}]" ; \
    fi && \
    pip cache purge

# Now copy the rest of the source code
COPY . .

# Make shell scripts executable
RUN find ./scripts -type f -name "*.sh" -exec chmod +x {} \;

ENTRYPOINT ["./scripts/utilities/docker-entrypoint.sh"]
CMD ["bash"]
