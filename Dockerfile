FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg \
    fonts-dejavu \
    fontconfig \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /usr/local/share/fonts/anton \
    && curl -L --fail --silent --show-error \
    -o /usr/local/share/fonts/anton/Anton-Regular.ttf \
    https://raw.githubusercontent.com/google/fonts/main/ofl/anton/Anton-Regular.ttf \
    && fc-cache -f

WORKDIR /app

COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
