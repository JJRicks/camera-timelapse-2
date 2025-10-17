#!/usr/bin/env python3
import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
from picamera2 import Picamera2
from libcamera import controls

def parse_args():
    p = argparse.ArgumentParser("Raspberry Pi Camera live preview (Picamera2 + OpenCV)")
    p.add_argument("--width", type=int, default=1280, help="Preview width (default: 1280)")
    p.add_argument("--height", type=int, default=720, help="Preview height (default: 720)")
    p.add_argument("--ev", type=float, default=1.0, help="EV compensation (default: +1.0 = brighter)")
    p.add_argument("--lens-position", type=float, default=0.0, help="Manual focus lens position (0.0≈infinity)")
    p.add_argument("--outdir", type=Path, default=Path("./snapshots"), help="Folder for 's' snapshots")
    return p.parse_args()

def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (args.width, args.height), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(0.3)

    # Fixed infinity focus + bright-biased AE, prefer longer shutter where supported
    controls_dict = {
        "AfMode": controls.AfModeEnum.Manual,
        "LensPosition": float(args.lens_position),  # 0.0 ≈ infinity
        "AeEnable": True,
        "AeMeteringMode": controls.AeMeteringModeEnum.Matrix,
        "ExposureValue": float(args.ev),
    }
    try:
        controls_dict["AeExposureMode"] = controls.AeExposureModeEnum.Long
    except Exception:
        pass  # Some builds don't expose this; it's fine.

    picam2.set_controls(controls_dict)

    win = "Preview — q: quit | s: snapshot | [ / ]: EV ±0.5 | i: infinity | a: AF continuous"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, args.width, args.height)

    frames = 0
    t0 = time.time()
    fps = 0.0

    try:
        while True:
            frame = picam2.capture_array()  # RGB
            md = picam2.capture_metadata() or {}

            exp_us = md.get("ExposureTime")
            gain = md.get("AnalogueGain")
            lens = md.get("LensPosition")
            lux = md.get("Lux")

            # Safe string formats (avoid formatting None)
            exp_str = f"{exp_us}" if exp_us is not None else "—"
            gain_str = f"{gain:.2f}" if isinstance(gain, (int, float)) else "—"
            lens_str = f"{lens:.3f}" if isinstance(lens, (int, float)) else "—"
            lux_str = f"{lux:.1f}" if isinstance(lux, (int, float)) else "—"

            # FPS counter
            frames += 1
            now = time.time()
            if now - t0 >= 1.0:
                fps = frames / (now - t0)
                frames = 0
                t0 = now

            # HUD text
            hud = (
                f"EV {args.ev:+.1f} | exp {exp_str} µs | gain {gain_str} | "
                f"lens {lens_str} | lux {lux_str} | {fps:.1f} FPS"
            )
            cv2.putText(frame, hud, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # OpenCV expects BGR
            cv2.imshow(win, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = args.outdir / f"preview_{ts}.jpg"
                cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                print(f"Saved snapshot: {path}")
            elif key == ord(']'):
                args.ev += 0.5
                picam2.set_controls({"ExposureValue": float(args.ev)})
            elif key == ord('['):
                args.ev -= 0.5
                picam2.set_controls({"ExposureValue": float(args.ev)})
            elif key == ord('i'):
                picam2.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": float(args.lens_position)})
            elif key == ord('a'):
                # Toggle to continuous AF (press 'i' to go back to manual infinity)
                picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
    finally:
        cv2.destroyAllWindows()
        picam2.stop()

if __name__ == "__main__":
    main()
