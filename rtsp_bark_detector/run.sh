#!/usr/bin/with-contenv bashio

bashio::log.info "Starting RTSP Bark Detector"

export MQTT_HOST="$(bashio::services mqtt "host")"
export MQTT_PORT="$(bashio::services mqtt "port")"
export MQTT_USERNAME="$(bashio::services mqtt "username")"
export MQTT_PASSWORD="$(bashio::services mqtt "password")"

cd /app

python3 main.py