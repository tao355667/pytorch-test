# 使用 PyTorch 官方镜像作为基础镜像
FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime

# 设置代理参数（构建时可覆盖）
# ARG HTTP_PROXY
# ARG HTTPS_PROXY

# 将代理参数传递到 apt-get 和 pip
# ENV http_proxy=${HTTP_PROXY}
# ENV https_proxy=${HTTPS_PROXY}

# 设置工作目录（容器中的）
WORKDIR /workspace

# 先安装系统级依赖（通过代理）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    vim \
    wget \
    curl \
    htop \
    tmux \
    openssh-server \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 将本地代码复制到容器中
COPY . /workspace

# 安装 Python 依赖（通过代理）
RUN pip install --no-cache-dir -r requirements.txt

# 清除代理设置（避免残留到镜像中）
# ENV http_proxy=
# ENV https_proxy=

# 暴露端口（SSH 默认端口 22，按需开放）
# EXPOSE 22 8000

# 定义容器启动时运行的命令
# CMD ["python", "app.py"]