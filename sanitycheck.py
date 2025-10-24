
import platform, importlib, cv2, numpy as np
print("py:", platform.python_version(), "arch:", platform.machine())
print("numpy:", np.__version__)
print("tflite:", importlib.import_module("tflite_runtime.interpreter").__name__)
print("cv2:", cv2.__version__)

