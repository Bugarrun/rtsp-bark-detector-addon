ARG BUILD_FROM=ghcr.io/home-assistant/amd64-base:latest

FROM $BUILD_FROM

WORKDIR /app

COPY app/ /app/
COPY models/ /app/models/
COPY config/config.yaml /app/config.yaml

RUN apk add --no-cache \
    ffmpeg \
    python3 \
    py3-pip

RUN pip install --break-system-packages --no-cache-dir \
    numpy \
    pyyaml \
    paho-mqtt \
    mediapipe \
    imageio-ffmpeg

COPY run.sh /run.sh
RUN chmod a+x /run.sh

CMD ["/run.sh"]