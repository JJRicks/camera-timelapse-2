#!/usr/bin/env python3
import time
from picamera2 import Picamera2, Preview
from libcamera import controls

picam2 = Picamera2()
picam2.start_preview(Preview.QTGL)  # Or Preview.DRM on the Pi’s local display (no X/Wayland)
picam2.configure(picam2.create_preview_configuration())
picam2.start()
time.sleep(0.3)

# Fixed infinity + bright-biased exposure
ctrls = {
    "AfMode": controls.AfModeEnum.Manual,
    "LensPosition": 0.0,  # ≈ infinity
    "AeEnable": True,
    "AeMeteringMode": controls.AeMeteringModeEnum.Matrix,
    "ExposureValue": 1.0,
}
try:
    from libcamera import controls as c
    ctrls["AeExposureMode"] = c.AeExposureModeEnum.Long
except Exception:
    pass

picam2.set_controls(ctrls)

print("Preview running. Press Ctrl+C to exit.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    picam2.stop()
