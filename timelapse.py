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
        description="Timelapse with infinity focus; AE biased brighter; long shutter only when dark."
    )
    p.add_argument("-i", "--interval", type=float, default=120,
                   help="Seconds between shots (default: 120)")
    p.add_argument("-o", "--output", type=Path, default=Path("./images2"),
                   help="Output folder (default: ./images)")
    p.add_argument("--prefix", default="frame_",
                   help="Filename prefix (default: frame_)")
    p.add_argument("--ev", type=float, default=1.0,
                   help="EV compensation for AE (typical range ±2..±4; default: +1.0)")
    p.add_argument("--quality", type=int, default=50,
                   help="JPEG quality 1–100 (default: 93)")
    p.add_argument("--lens-position", type=float, default=0.0,
                   help="Manual focus LensPosition (0.0≈infinity; default: 0.0)")
    p.add_argument("--width", type=int, default=None, help="Still width (optional)")
    p.add_argument("--height", type=int, default=None, help="Still height (optional)")

    # NEW: adaptive long-exposure controls
    p.add_argument("--day-max-shutter", type=float, default=(1/15),
                   help="Max shutter (seconds) in bright scenes (default: 1/15 ≈ 0.0667s)")
    p.add_argument("--night-max-shutter", type=float, default=1.0,
                   help="Max shutter (seconds) when dark (default: 1.0s)")
    p.add_argument("--dark-lux", type=float, default=4.0,
                   help="Switch to 'dark' mode below this Lux (default: 4.0)")
    p.add_argument("--hysteresis", type=float, default=3.0,
                   help="Lux hysteresis to return to 'bright' mode (default: 3.0)")
    p.add_argument("--gain-dark", type=float, default=4.0,
                   help="Fallback: treat as dark if AnalogueGain exceeds this and Lux is unavailable (default: 4.0)")
    p.add_argument("--settle", type=float, default=1.0,
                   help="Time (s) to let AE settle after changing limits, before capture (default: 0.2)")
    return p.parse_args()

def us(x):  # seconds -> microseconds
    return int(round(x * 1_000_000))

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

    still_cfg = picam2.create_still_configuration()
    if args.width and args.height:
        still_cfg["main"]["size"] = (args.width, args.height)
    picam2.configure(still_cfg)
    picam2.options["quality"] = args.quality

    # SIGTERM -> graceful stop; let Ctrl+C raise KeyboardInterrupt
    def _handle_sigterm(signum, frame):
        stop_event.set()
    signal.signal(signal.SIGTERM, _handle_sigterm)

    picam2.start()
    time.sleep(0.5)  # settle

    # AE setup + fixed infinity focus
    try:
        picam2.set_controls({
            "AeEnable": True,
            "AeMeteringMode": controls.AeMeteringModeEnum.Matrix,
            "ExposureValue": float(args.ev),
            "AfMode": controls.AfModeEnum.Manual,
            "LensPosition": float(args.lens_position),
        })
        # Ask AE to prefer longer exposures when allowed
        try:
            picam2.set_controls({"AeExposureMode": controls.AeExposureModeEnum.Long})
        except Exception:
            pass
    except Exception as e:
        logging.error("Failed to set initial controls: %s", e)

    # Start in BRIGHT mode (cap shutter to day value)
    BRIGHT_CAP_US = us(args.day_max_shutter)
    DARK_CAP_US   = us(args.night_max_shutter)
    MIN_FRAME_US  = 100  # very small lower bound

    def set_cap(max_us):
        try:
            picam2.set_controls({"FrameDurationLimits": (MIN_FRAME_US, int(max_us))})
            return True
        except Exception as e:
            logging.warning("FrameDurationLimits not supported or rejected: %s", e)
            return False

    mode_dark = False
    set_cap(BRIGHT_CAP_US)
    time.sleep(1.0)  # warm-up

    logging.info(
        "Timelapse running: interval=%.3fs, EV=%.2f, LensPosition=%.3f, day cap=%.1f ms, night cap=%.0f ms",
        args.interval, args.ev, args.lens_position, BRIGHT_CAP_US/1000, DARK_CAP_US/1000
    )

    try:
        while not stop_event.is_set():
            # --- Pre-capture: check light level and (maybe) change max shutter ---
            md_pre = picam2.capture_metadata() or {}
            lux = md_pre.get("Lux")
            exp_us_pre = md_pre.get("ExposureTime")
            gain_pre = md_pre.get("AnalogueGain")

            # Decide brightness state using Lux primarily, otherwise exposure/gain heuristics
            if mode_dark:
                bright_now = ((isinstance(lux, (int, float)) and lux >= (args.dark_lux + args.hysteresis)) or
                              (lux is None and isinstance(exp_us_pre, int) and exp_us_pre < int(BRIGHT_CAP_US * 0.5)))
                if bright_now:
                    if set_cap(BRIGHT_CAP_US):
                        mode_dark = False
                        logging.info("Bright scene: capping shutter to %.1f ms (lux=%s, exp=%s µs, gain=%.2f)",
                                     BRIGHT_CAP_US/1000, f"{lux:.1f}" if isinstance(lux,(int,float)) else "—",
                                     exp_us_pre if exp_us_pre is not None else "—",
                                     gain_pre if isinstance(gain_pre,(int,float)) else float('nan'))
                        time.sleep(args.settle)
            else:
                dark_now = ((isinstance(lux, (int, float)) and lux <= args.dark_lux) or
                            (lux is None and isinstance(exp_us_pre, int) and exp_us_pre >= int(BRIGHT_CAP_US*0.95)) or
                            (lux is None and isinstance(gain_pre, (int, float)) and gain_pre >= args.gain_dark))
                if dark_now:
                    if set_cap(DARK_CAP_US):
                        mode_dark = True
                        logging.info("Dark scene: allowing long shutter up to %.0f ms (lux=%s, exp=%s µs, gain=%.2f)",
                                     DARK_CAP_US/1000, f"{lux:.1f}" if isinstance(lux,(int,float)) else "—",
                                     exp_us_pre if exp_us_pre is not None else "—",
                                     gain_pre if isinstance(gain_pre,(int,float)) else float('nan'))
                        # Prefer 'Long' AE when dark
                        try:
                            picam2.set_controls({"AeExposureMode": controls.AeExposureModeEnum.Long})
                        except Exception:
                            pass
                        time.sleep(args.settle)
                else:
                    # Prefer 'Normal' AE in bright scenes (optional; safe no-op if unsupported)
                    try:
                        picam2.set_controls({"AeExposureMode": controls.AeExposureModeEnum.Normal})
                    except Exception:
                        pass

            start = time.time()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = args.output / f"{args.prefix}{ts}.jpg"

            try:
                picam2.capture_file(str(filepath))
                md = picam2.capture_metadata() or {}
                exp_us = md.get("ExposureTime")
                gain = md.get("AnalogueGain")
                lens = md.get("LensPosition")
                lux_now = md.get("Lux")

                logging.info(
                    "Saved %s | Exposure: %s µs | Gain: %s | LensPos: %s | Lux: %s | mode=%s",
                    filepath,
                    exp_us if exp_us is not None else "—",
                    f"{gain:.2f}" if isinstance(gain, (int, float)) else "—",
                    f"{lens:.3f}" if isinstance(lens, (int, float)) else "—",
                    f"{lux_now:.1f}" if isinstance(lux_now, (int, float)) else "—",
                    "DARK" if mode_dark else "BRIGHT",
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
