#!/usr/bin/env python3
import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

from picamera2 import Picamera2
from libcamera import controls

stop_flag = False

def handle_sigint(signum, frame):
    global stop_flag
    stop_flag = True

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

    # Start camera first; then apply controls (per docs best practice)
    picam2.start()
    time.sleep(0.5)  # small settle

    # Try to bias AE toward longer exposures & brighter images
    control_dict = {
        "AeEnable": True,
        "AeMeteringMode": controls.AeMeteringModeEnum.Matrix,
        "ExposureValue": float(args.ev),  # EV compensation
        # Fix focus at infinity:
        "AfMode": controls.AfModeEnum.Manual,
        "LensPosition": float(args.lens_position),
    }

    # Prefer longer shutter if supported on this sensor/tuning
    try:
        control_dict["AeExposureMode"] = controls.AeExposureModeEnum.Long
    except Exception:
        # Some builds don’t expose the "Long" mode; that’s fine.
        logging.warning("AeExposureMode=Long not available; continuing without it.")

    # Apply our control set
    try:
        picam2.set_controls(control_dict)
    except Exception as e:
        logging.error("Failed to set one or more controls: %s", e)

    # Warm-up so AE/AF settle with our constraints
    time.sleep(1.0)

    # Clean exit on Ctrl+C / SIGTERM
    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    shot_idx = 0
    logging.info(
        "Timelapse running: interval=%.3fs, EV=%.2f, LensPosition=%.3f (infinity), JPEG q=%d",
        args.interval, args.ev, args.lens_position, args.quality
    )

    try:
        while not stop_flag:
            start = time.time()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{args.prefix}{ts}.jpg"
            filepath = args.output / filename

            try:
                picam2.capture_file(str(filepath))
                md = picam2.capture_metadata() or {}
                exp_us = md.get("ExposureTime")
                gain = md.get("AnalogueGain")
                lens = md.get("LensPosition")
                lux = md.get("Lux")

                logging.info(
                    "Saved %s | Exposure: %s µs | Gain: %s | LensPos: %s | Lux: %s",
                    filepath, exp_us, f"{gain:.2f}" if gain else None,
                    f"{lens:.3f}" if lens is not None else None,
                    f"{lux:.1f}" if lux else None
                )
            except Exception as e:
                logging.error("Capture failed: %s", e)

            shot_idx += 1
            elapsed = time.time() - start
            sleep_for = max(0.0, args.interval - elapsed)
            time.sleep(sleep_for)
    finally:
        logging.info("Stopping camera…")
        picam2.stop()
        logging.info("Done.")

if __name__ == "__main__":
    if os.geteuid() != 0:
        # Not strictly required, but helps with permissions on some setups.
        logging.warning("Running without sudo; if you hit permission issues, try: sudo %s", sys.argv[0])
    main()
