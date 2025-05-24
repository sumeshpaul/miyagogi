FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

LABEL maintainer="sumesh@meledath.me"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TOKENIZERS_PARALLELISM=false \
    MODE=serve \
    PATH="/venv/bin:$PATH" \
    CFLAGS="-mcmodel=large" \
    CXXFLAGS="-mcmodel=large" \
    LDFLAGS="-mcmodel=large" \
    HF_HOME=/app/.cache/huggingface

# Base system setup
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update && \
    apt-get install -y \
        gcc-13 g++-13 \
        build-essential cmake \
        python3.10 python3.10-dev python3.10-venv python3-pip \
        git wget curl sqlite3 libopenblas-dev ca-certificates file && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install core packages
RUN pip install \
    typing_extensions \
    pyyaml \
    langchain \
    langchain-community \
    faiss-cpu \
    sentence-transformers \
    tiktoken

# Install custom PyTorch + cuDNN
RUN wget https://github.com/sumeshpaul/llm-telegram-bot/releases/download/v1.0/torch-2.8.0a0-cp310-cp310-linux_x86_64.whl && \
    pip install --no-deps torch-2.8.0a0-cp310-cp310-linux_x86_64.whl

RUN wget https://github.com/sumeshpaul/llm-telegram-bot/releases/download/v1.0/cudnn-latest.tar.xz && \
    tar -xf cudnn-latest.tar.xz -C /tmp && \
    CUDNN_DIR=$(find /tmp -type d -name "cudnn-linux-x86_64*" | head -n 1) && \
    cp -P "$CUDNN_DIR/include/"* /usr/include/ && \
    cp -P "$CUDNN_DIR/lib/"* /usr/lib/x86_64-linux-gnu/ && \
    echo "/usr/lib/x86_64-linux-gnu" > /etc/ld.so.conf.d/cudnn.conf && \
    ldconfig && \
    rm -rf /tmp/cudnn*

# Compile bitsandbytes (optional)
WORKDIR /opt
RUN git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git bnb
WORKDIR /opt/bnb
RUN cmake -DCOMPUTE_BACKEND=cuda -S . && \
    make -j$(nproc) && \
    pip install -e .

# App directory
WORKDIR /app
COPY ./app /app
COPY ./data /app/data
COPY ./final_lora_model_v2 /app/final_lora_model_v2
COPY requirements.txt /app/requirements.txt

# Install app-specific packages
RUN pip install --no-cache-dir -r /app/requirements.txt \
    pymupdf python-docx sentencepiece python-multipart \
    gradio huggingface_hub[hf_xet] python-dotenv telegram

# Fix for bitsandbytes fallback error
RUN ln -s /opt/bnb/bitsandbytes/libbitsandbytes_cuda12.so /opt/bnb/bitsandbytes/libbitsandbytes_cpu.so || true

# Expose ports for FastAPI
EXPOSE 8000

CMD [ "python", "/app/main_server.py" ]
