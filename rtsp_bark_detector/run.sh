#!/usr/bin/with-contenv bashio

bashio::log.info "Starting RTSP Bark Detector"

cd /app

python3 main.py