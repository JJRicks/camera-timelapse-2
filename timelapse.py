#!/usr/bin/env python3
import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
import threading

from picamera2 import Picamera2
from libcamera import controls

stop_event = threading.Event()  # used for SIGTERM and to interrupt sleeps

def parse_args():
    p = argparse.ArgumentParser(
        description="Capture a photo every X seconds with fixed-infinity focus and bright-biased AE."
    )
    p.add_argument("-i", "--interval", type=float, default=5.0,
                   help="Seconds between shots (default: 5.0)")
    p.add_argument("-o", "--output", type=Path, default=Path("./images"),
                   help="Output folder (default: ./images)")
    p.add_argument("--prefix", default="frame_",
                   help="Filename prefix (default: frame_)")
    p.add_argument("--ev", type=float, default=1.0,
                   help="Exposure compensation in EV (default: +1.0 = brighter)")
    p.add_argument("--quality", type=int, default=93,
                   help="JPEG quality 1–100 (default: 93)")
    p.add_argument("--lens-position", type=float, default=0.0,
                   help="Manual focus LensPosition (0.0≈infinity; default: 0.0)")
    p.add_argument("--width", type=int, default=None, help="Still width (optional)")
    p.add_argument("--height", type=int, default=None, help="Still height (optional)")
    return p.parse_args()

def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args.output.mkdir(parents=True, exist_ok=True)
    logging.info("Starting camera; output dir: %s", args.output.resolve())

    picam2 = Picamera2()

    # Optional size
    still_cfg = picam2.create_still_configuration()
    if args.width and args.height:
        still_cfg["main"]["size"] = (args.width, args.height)

    picam2.configure(still_cfg)
    picam2.options["quality"] = args.quality

    # Let Ctrl+C raise KeyboardInterrupt (no custom SIGINT handler)
    # Handle SIGTERM gracefully by setting an event:
    def _handle_sigterm(signum, frame):
        stop_event.set()
    signal.signal(signal.SIGTERM, _handle_sigterm)

    picam2.start()
    time.sleep(0.5)  # settle

    # Bias AE toward brighter images, fix focus at (approx) infinity
    control_dict = {
        "AeEnable": True,
        "AeMeteringMode": controls.AeMeteringModeEnum.Matrix,
        "ExposureValue": float(args.ev),
        "AfMode": controls.AfModeEnum.Manual,
        "LensPosition": float(args.lens_position),
    }
    try:
        control_dict["AeExposureMode"] = controls.AeExposureModeEnum.Long
    except Exception:
        logging.warning("AeExposureMode=Long not available; continuing without it.")
    try:
        picam2.set_controls(control_dict)
    except Exception as e:
        logging.error("Failed to set one or more controls: %s", e)

    time.sleep(1.0)  # warm-up

    logging.info(
        "Timelapse running: interval=%.3fs, EV=%.2f, LensPosition=%.3f (infinity), JPEG q=%d",
        args.interval, args.ev, args.lens_position, args.quality
    )

    try:
        while not stop_event.is_set():
            start = time.time()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = args.output / f"{args.prefix}{ts}.jpg"

            try:
                # Ctrl+C here will raise KeyboardInterrupt immediately
                picam2.capture_file(str(filepath))
                md = picam2.capture_metadata() or {}
                exp_us = md.get("ExposureTime")
                gain = md.get("AnalogueGain")
                lens = md.get("LensPosition")
                lux = md.get("Lux")

                logging.info(
                    "Saved %s | Exposure: %s µs | Gain: %s | LensPos: %s | Lux: %s",
                    filepath,
                    exp_us if exp_us is not None else "—",
                    f"{gain:.2f}" if isinstance(gain, (int, float)) else "—",
                    f"{lens:.3f}" if isinstance(lens, (int, float)) else "—",
                    f"{lux:.1f}" if isinstance(lux, (int, float)) else "—",
                )
            except KeyboardInterrupt:
                logging.info("Ctrl+C pressed — stopping after current frame.")
                break
            except Exception as e:
                logging.error("Capture failed: %s", e)

            # Sleep the remaining interval, but wake early on SIGTERM or Ctrl+C
            elapsed = time.time() - start
            remaining = max(0.0, args.interval - elapsed)
            try:
                if stop_event.wait(timeout=remaining):
                    break
            except KeyboardInterrupt:
                logging.info("Ctrl+C pressed — stopping.")
                break

    finally:
        logging.info("Stopping camera…")
        try:
            picam2.stop()
        except Exception:
            pass
        logging.info("Done.")

if __name__ == "__main__":
    if os.geteuid() != 0:
        logging.warning("Running without sudo; if you hit permission issues, try: sudo %s", sys.argv[0])
    main()
