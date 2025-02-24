# 使用 PyTorch 官方镜像作为基础镜像
FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel

# 设置工作目录（容器中的）
WORKDIR /workspace

# 将本地代码复制到容器中
COPY . /workspace

# 安装额外的依赖（如果有）
RUN pip install --no-cache-dir -r requirements.txt

# 暴露端口（如果有需要）
# EXPOSE 8000

# 定义容器启动时运行的命令
# CMD ["python", "app.py"]