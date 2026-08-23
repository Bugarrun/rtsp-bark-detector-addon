import json
import logging
import paho.mqtt.client as mqtt


class HABridge:

    def __init__(self, mqtt_config, device_config):

        self.logger = logging.getLogger("ha_bridge")

        self.broker = mqtt_config["broker"]
        self.port = mqtt_config["port"]

        self.username = mqtt_config["username"]
        self.password = mqtt_config["password"]

        self.device_config = device_config

        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id="dog_bark_detector",
            clean_session=True
        )

        self.client.username_pw_set(
            self.username,
            self.password
        )

        self.client.reconnect_delay_set(
            min_delay=1,
            max_delay=60
        )

        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect

        self.client.connect(
            self.broker,
            self.port,
            60
        )

        self.client.loop_start()

        self.logger.info(
            "MQTT HA bridge started"
        )


    def publish_discovery(self):

        sensors = {

            "dog_barking": {
                "name": "Dog Barking",
                "topic": "dog_bark_detector/state",
                "payload_on": "ON",
                "payload_off": "OFF",
                "device_class": "sound"
            },

            "last_bark_duration": {
                "name": "Last Bark Duration",
                "topic": "dog_bark_detector/duration",
                "unit": "s"
            },

            "last_bark_confidence": {
                "name": "Last Bark Confidence",
                "topic": "dog_bark_detector/confidence"
            }

        }

        for sensor_id, data in sensors.items():

            payload = {
                "name": data["name"],
                "unique_id": sensor_id,
                "state_topic": data["topic"],
                "device": {
                    "identifiers": [
                        "dog_bark_detector"
                    ],
                    "name": self.device_config["name"],
                    "manufacturer": self.device_config["manufacturer"],
                    "model": self.device_config["model"],
                    "sw_version": self.device_config["version"]
                }
            }

            if "payload_on" in data:
                payload["payload_on"] = data["payload_on"]
                payload["payload_off"] = data["payload_off"]

            if "device_class" in data:
                payload["device_class"] = data["device_class"]

            if "unit" in data:
                payload["unit_of_measurement"] = data["unit"]

            if sensor_id == "dog_barking":

                topic = (
                    f"homeassistant/binary_sensor/"
                    f"{sensor_id}/config"
                )

            else:

                topic = (
                    f"homeassistant/sensor/"
                    f"{sensor_id}/config"
                )

            self.client.publish(
                topic,
                json.dumps(payload),
                retain=True
            )

        self.logger.info(
            "MQTT discovery published"
        )


    def process_event(self, event):

        if not event:
            return

        if event["event"] == "started":

            self.client.publish(
                "dog_bark_detector/state",
                "ON",
                retain=True
            )

            self.logger.info(
                "MQTT HA ENTITY: dog_barking = ON"
            )

        elif event["event"] == "stopped":

            self.client.publish(
                "dog_bark_detector/state",
                "OFF",
                retain=True
            )

            self.client.publish(
                "dog_bark_detector/duration",
                event["duration"],
                retain=True
            )

            self.client.publish(
                "dog_bark_detector/confidence",
                event["confidence"],
                retain=True
            )

            self.logger.info(
                "MQTT HA ENTITY: dog_barking = OFF"
            )


    def on_connect(self, client, userdata, flags, rc, properties=None):

        if rc == 0:

            self.logger.info(
                "MQTT connected successfully"
            )

            self.publish_discovery()

        else:

            self.logger.error(
                f"MQTT connection failed | code={rc}"
            )


    def on_disconnect(self, client, userdata, flags, rc, properties=None):

        self.logger.warning(
            f"MQTT disconnected | code={rc}"
        )