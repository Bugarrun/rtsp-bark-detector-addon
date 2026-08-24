ARG BUILD_FROM=ghcr.io/home-assistant/amd64-base-debian:latest

FROM $BUILD_FROM

WORKDIR /app

COPY app/ /app/
COPY models/ /app/models/
COPY config/defaults.yaml /app/config.yaml

RUN apt-get update && apt-get install -y \
    ffmpeg \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --break-system-packages --no-cache-dir \
    numpy \
    pyyaml \
    paho-mqtt \
    imageio-ffmpeg \
    tensorflow

COPY run.sh /run.sh
RUN chmod a+x /run.sh

CMD ["/run.sh"]
