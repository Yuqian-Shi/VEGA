FROM docker.1ms.run/ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PLAYWRIGHT_BROWSERS_DOWNLOAD_HOST=https://playwright.azureedge.net \
    SHELL=/bin/bash
# sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/ubuntu.sources && \
# sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/ubuntu.sources && \
# sed -i 's/ppa.launchpad.net/launchpad.proxy.ustclug.org/g' /etc/apt/sources.list.d/deadsnakes* && \
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    
    apt-get update && apt-get install -y --no-install-recommends build-essential curl wget git ca-certificates \
        python3.13 python3.13-dev python3.13-venv python3-pip libssl-dev libffi-dev libpq-dev zlib1g-dev \
        libjpeg-dev libpng-dev pkg-config openssh-server openssh-client vim iputils-ping iproute2 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.13 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1 && \
    python -m pip install --no-cache-dir --break-system-packages -i https://pypi.mirrors.ustc.edu.cn/simple \
        pip setuptools wheel numpy pydantic requests tqdm pyyaml jsonlines aiohttp urllib3 uvicorn \
        fastapi fastchat anthropic transformers pillow gdown beautifulsoup4 selenium jupyter jupyterlab \
        jupyter-server jupyter-server-terminals notebook ipython ipykernel docker networkx accelerate playwright mysql-connector-python\
        tiktoken beartype gymnasium matplotlib torch transformers datasets evaluate nltk opencv-python \
        scikit-image pandas pytest pytest-xdist pytest-asyncio gradio_client text-generation safetensors\
        tokenizers sentencepiece huggingface-hub openai lxml aiolimiter dashscope google google-auth dotenv&& \
    python -m playwright install --with-deps && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    mkdir -p /run/ssh /root/.ssh && \
    chmod 644 /etc/ssh/sshd_config && \
    chown root:root /etc/ssh/sshd_config && \
    ssh-keygen -A && \
    echo "root:root" | chpasswd && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config || true && \
    sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config || true && \
    sed -i 's/#PermitRootLogin yes/PermitRootLogin yes/' /etc/ssh/sshd_config || true && \
    echo "ClientAliveInterval 5" >> /etc/ssh/sshd_config && \
    echo "ClientAliveCountMax 3" >> /etc/ssh/sshd_config
# Copy configuration files
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
# COPY authorized_keys /root/.ssh/authorized_keys
RUN mkdir -p /run/sshd
EXPOSE 22 8888

ENTRYPOINT ["/entrypoint.sh"]