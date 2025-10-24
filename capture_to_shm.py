#!/usr/bin/env python3
import argparse, os, time, sys
from pathlib import Path

import cv2

def parse_args():
    p = argparse.ArgumentParser("Picamera2 capture → /dev/shm for TFLite consumer")
    p.add_argument("--out", default="/dev/shm/picam.jpg",
                   help="Output JPEG path (default: /dev/shm/picam.jpg)")
    p.add_argument("--interval", type=float, default=5.0,
                   help="Seconds between writes (default: 5.0)")
    p.add_argument("--width", type=int, default=1280, help="Preview/capture width (default: 1280)")
    p.add_argument("--height", type=int, default=720, help="Preview/capture height (default: 720)")
    p.add_argument("--quality", type=int, default=85, help="JPEG quality 1–100 (default: 85)")
    p.add_argument("--preview", action="store_true", help="Show live preview window")
    p.add_argument("--rotate", type=int, choices=[0,90,180,270], default=0, help="Rotate preview/capture")
    p.add_argument("--ev", type=float, default=0.0, help="Exposure compensation (EV)")
    return p.parse_args()

def main():
    args = parse_args()

    try:
        from picamera2 import Picamera2
        from libcamera import controls
    except Exception as e:
        print("Picamera2 not available here:", e, file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp.jpg")

    # Camera setup (video mode = responsive; use 4:3 for wider FOV if you want)
    picam2 = Picamera2()
    cfg = picam2.create_video_configuration(main={"size": (args.width, args.height)}, buffer_count=4)
    picam2.configure(cfg)
    picam2.start()
    time.sleep(0.3)  # small settle

    # Lock focus at infinity; set AE as you like
    try:
        picam2.set_controls({
            "AfMode": controls.AfModeEnum.Manual,
            "LensPosition": 0.0,  # ≈ infinity
            "AeEnable": True,
            "AeMeteringMode": controls.AeMeteringModeEnum.Matrix,
            "ExposureValue": float(args.ev),
        })
    except Exception:
        pass

    print(f"[capture] writing → {out_path} every {args.interval}s  "
          f"({args.width}x{args.height}, Q={args.quality})  "
          f"{'(preview ON)' if args.preview else '(preview OFF)'}")
    next_shot = time.time()

    try:
        while True:
            # Grab a frame for preview and/or for saving
            rgb = picam2.capture_array()              # RGB
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            if args.rotate:
                rotflag = {90: cv2.ROTATE_90_CLOCKWISE,
                           180: cv2.ROTATE_180,
                           270: cv2.ROTATE_90_COUNTERCLOCKWISE}[args.rotate]
                bgr = cv2.rotate(bgr, rotflag)

            now = time.time()
            if now >= next_shot:
                # Atomic write: temp then replace → consumers never see partial files
                ok = cv2.imwrite(str(tmp_path), bgr,
                                 [int(cv2.IMWRITE_JPEG_QUALITY), int(args.quality)])
                if not ok:
                    print("[capture] WARNING: failed to write temp jpeg", file=sys.stderr)
                else:
                    os.replace(str(tmp_path), str(out_path))
                    print(f"[capture] wrote {out_path} @ {time.strftime('%H:%M:%S')}")
                next_shot = now + max(0.2, args.interval)

            if args.preview:
                cv2.imshow("Picamera2 capture (q to quit)", bgr)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
            else:
                # gentle idle so we don't pin the CPU; preview off → slower loop
                time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            picam2.stop()
        except Exception:
            pass
        if args.preview:
            cv2.destroyAllWindows()
        # cleanup temp if present
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        print("[capture] stopped.")

if __name__ == "__main__":
    main()
