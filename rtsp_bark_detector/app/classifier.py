"""YAMNet waveform inference using the runtime already installed by the add-on."""
import zipfile

import numpy as np
from tensorflow.lite.python.interpreter import Interpreter


class BarkClassifier:
    sample_rate = 16000

    def __init__(self, model_path):
        with zipfile.ZipFile(model_path) as model:
            labels = model.read("yamnet_label_list.txt").decode("utf-8").splitlines()
        self.bark_index = labels.index("Bark")
        self.dog_index = labels.index("Dog")
        self.interpreter = Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        inputs = self.interpreter.get_input_details()
        outputs = self.interpreter.get_output_details()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Expected the bundled YAMNet waveform classification model")
        self.input = inputs[0]
        self.output = outputs[0]
        if (tuple(self.input["shape"]) != (15600,)
                or self.input["dtype"] != np.float32
                or tuple(self.output["shape"]) != (1, len(labels))
                or self.output["dtype"] != np.float32):
            raise ValueError("Unsupported YAMNet input/output tensors")
        self.samples_per_window = int(self.input["shape"][0])

    def classify_pcm(self, pcm):
        if len(pcm) != self.samples_per_window * 2:
            raise ValueError("Expected one complete mono 16-bit PCM window")
        waveform = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        self.interpreter.set_tensor(self.input["index"], waveform)
        self.interpreter.invoke()
        scores = self.interpreter.get_tensor(self.output["index"])[0]
        if not np.isfinite(scores).all():
            raise ValueError("YAMNet returned non-finite scores")
        return float(scores[self.bark_index]), float(scores[self.dog_index])
