FROM python:3.13-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# Gradio 默认端口
EXPOSE 7860

# 设置环境变量 (运行时通过 docker run --env-file .env 传入)
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "app.py"]
