ARG BUILD_FROM

FROM $BUILD_FROM

WORKDIR /app

COPY app/ /app/
COPY models/ /app/models/
COPY config.yaml /app/

RUN apk add --no-cache \
    ffmpeg

RUN pip install --no-cache-dir \
    numpy \
    pyyaml \
    paho-mqtt \
    mediapipe \
    imageio-ffmpeg

COPY run.sh /run.sh
RUN chmod a+x /run.sh

CMD ["/run.sh"]