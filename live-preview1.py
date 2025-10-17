#!/usr/bin/env python3
import argparse
import time
from datetime import datetime
from pathlib import Path
import math

import cv2
from picamera2 import Picamera2
from libcamera import controls

def parse_args():
    p = argparse.ArgumentParser("Raspberry Pi Camera live preview (Picamera2 + OpenCV)")
    p.add_argument("--width", type=int, default=1280, help="Preview width (default: 1280)")
    p.add_argument("--height", type=int, default=720, help="Preview height (default: 720)")
    p.add_argument("--ev", type=float, default=1.0, help="EV compensation for AUTO mode (default: +1.0)")
    p.add_argument("--lens-position", type=float, default=0.0, help="Manual focus lens position (0.0≈infinity)")
    p.add_argument("--outdir", type=Path, default=Path("./snapshots"), help="Folder for 's' snapshots")
    return p.parse_args()

def nice_shutter(us: int | None) -> str:
    if not isinstance(us, (int, float)):
        return "—"
    if us >= 1_000_000:
        return f"{us/1_000_000:.2f}s"
    elif us >= 1_000:
        frac = int(round(1_000_000 / us))
        return f"{us/1000:.1f}ms (1/{frac})"
    else:
        return f"{us:.0f}µs"

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

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

    # Start in AUTO exposure, manual infinity focus, bright bias
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
        pass
    picam2.set_controls(controls_dict)

    # Manual exposure state (used when AE is disabled)
    manual_mode = False
    # Initialize from metadata once we have a frame
    manual_shutter_us = None
    manual_gain = None

    win = ("Preview — q: quit | s: snapshot | [ / ]: EV ±0.5 (AUTO) | "
           "e: toggle AUTO/MAN | 1/2: shutter ×0.5/×2 | 3/4: gain ÷1.25/×1.25 | "
           "r: reset manual | i: infinity | a: AF continuous")
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, args.width, args.height)

    frames = 0
    t0 = time.time()
    fps = 0.0

    def apply_manual():
        """Apply current manual_shutter_us and manual_gain to the camera."""
        nonlocal manual_shutter_us, manual_gain
        if manual_shutter_us is None or manual_gain is None:
            return
        # Reasonable safety clamps. Very long shutters will slow preview.
        manual_shutter_us = int(clamp(manual_shutter_us, 100, 8_000_000))  # 100 µs … 8 s
        manual_gain = float(clamp(manual_gain, 1.0, 16.0))                 # ~ISO 100 … 1600
        try:
            picam2.set_controls({
                "AeEnable": False,
                "ExposureTime": manual_shutter_us,
                "AnalogueGain": manual_gain,
                # Help the pipeline allow long shutters during preview
                "FrameDurationLimits": (manual_shutter_us, manual_shutter_us),
            })
        except Exception:
            # Some builds may reject FrameDurationLimits; try without it
            try:
                picam2.set_controls({
                    "AeEnable": False,
                    "ExposureTime": manual_shutter_us,
                    "AnalogueGain": manual_gain,
                })
            except Exception:
                pass  # Last resort: ignore failure; preview will continue

    try:
        while True:
            frame = picam2.capture_array()  # RGB
            md = picam2.capture_metadata() or {}

            exp_us = md.get("ExposureTime")
            gain = md.get("AnalogueGain")
            lens = md.get("LensPosition")
            lux = md.get("Lux")
            ae_enabled = md.get("AeEnable")
            # Initialize manual defaults from what AE chose on the first iteration
            if manual_shutter_us is None and isinstance(exp_us, int):
                manual_shutter_us = exp_us
            if manual_gain is None and isinstance(gain, (int, float)):
                manual_gain = float(gain)

            # FPS counter
            frames += 1
            now = time.time()
            if now - t0 >= 1.0:
                fps = frames / (now - t0)
                frames = 0
                t0 = now

            # HUD fields
            exp_str = nice_shutter(exp_us)
            gain_str = f"{gain:.2f}" if isinstance(gain, (int, float)) else "—"
            lens_str = f"{lens:.3f}" if isinstance(lens, (int, float)) else "—"
            lux_str = f"{lux:.1f}" if isinstance(lux, (int, float)) else "—"
            mode_str = "AUTO" if ae_enabled else "MAN"
            iso_str = f"~ISO {int(round((gain if isinstance(gain,(int,float)) else 1.0)*100))}"

            hud = (
                f"{mode_str} | EV {args.ev:+.1f} | {exp_str} | gain {gain_str} ({iso_str}) | "
                f"lens {lens_str} | lux {lux_str} | {fps:.1f} FPS"
            )
            cv2.putText(frame, hud, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)

            # Show
            cv2.imshow(win, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = args.outdir / f"preview_{ts}.jpg"
                cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                print(f"Saved snapshot: {path}")

            # EV (AUTO only)
            elif key == ord(']'):
                args.ev += 0.5
                picam2.set_controls({"ExposureValue": float(args.ev)})
            elif key == ord('['):
                args.ev -= 0.5
                picam2.set_controls({"ExposureValue": float(args.ev)})

            # Focus shortcuts
            elif key == ord('i'):
                picam2.set_controls({"AfMode": controls.AfModeEnum.Manual,
                                     "LensPosition": float(args.lens_position)})
            elif key == ord('a'):
                picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})

            # Toggle AUTO/MAN
            elif key == ord('e'):
                manual_mode = not manual_mode
                if manual_mode:
                    # Freeze at current exposure/gain then take over
                    if isinstance(exp_us, int):
                        manual_shutter_us = exp_us
                    if isinstance(gain, (int, float)):
                        manual_gain = float(gain)
                    apply_manual()
                    print(f"[MANUAL] shutter={nice_shutter(manual_shutter_us)}, gain={manual_gain:.2f}")
                else:
                    # Return to AE with current EV bias
                    picam2.set_controls({"AeEnable": True, "ExposureValue": float(args.ev)})
                    print("[AUTO] AE re-enabled")

            # Manual controls (active only when AE is off)
            elif manual_mode and key == ord('1'):
                manual_shutter_us = int(manual_shutter_us / 2) if manual_shutter_us else 5000
                apply_manual()
            elif manual_mode and key == ord('2'):
                manual_shutter_us = int(manual_shutter_us * 2) if manual_shutter_us else 10000
                apply_manual()
            elif manual_mode and key == ord('3'):
                manual_gain = (manual_gain / 1.25) if manual_gain else 1.0
                apply_manual()
            elif manual_mode and key == ord('4'):
                manual_gain = (manual_gain * 1.25) if manual_gain else 1.25
                apply_manual()
            elif manual_mode and key == ord('r'):
                # Reset manual to the current metadata values
                if isinstance(exp_us, int): manual_shutter_us = exp_us
                if isinstance(gain, (int, float)): manual_gain = float(gain)
                apply_manual()

    finally:
        cv2.destroyAllWindows()
        picam2.stop()

if __name__ == "__main__":
    main()
