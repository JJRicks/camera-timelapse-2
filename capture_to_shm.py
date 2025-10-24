#!/usr/bin/env python3
import argparse, os, sys, time
from pathlib import Path

import cv2

def parse_args():
    p = argparse.ArgumentParser("Full-res stills → downsample → /dev/shm/picam.jpg")
    p.add_argument("--out", default="/dev/shm/picam.jpg",
                   help="Downsampled JPEG written atomically here (default: /dev/shm/picam.jpg)")
    p.add_argument("--interval", type=float, default=5.0,
                   help="Seconds between captures (default: 5.0)")
    p.add_argument("--down-width", type=int, default=1280,
                   help="Width of the downsampled frame saved to --out (default: 1280)")
    p.add_argument("--down-height", type=int, default=720,
                   help="Height of the downsampled frame saved to --out (default: 720)")
    p.add_argument("--preview", action="store_true",
                   help="Show live preview of the downsampled frame")
    p.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0,
                   help="Rotate output/preview (degrees)")
    p.add_argument("--quality", type=int, default=85,
                   help="JPEG quality for --out (default: 85)")
    p.add_argument("--save-full", type=Path, default=None,
                   help="Optional directory to also save the full-res JPEGs (disabled by default)")
    p.add_argument("--ev", type=float, default=0.0,
                   help="AE exposure compensation (EV), e.g. 0.5 for +1/2 stop")
    return p.parse_args()

def main():
    args = parse_args()

    try:
        from picamera2 import Picamera2
        from libcamera import controls
    except Exception as e:
        print("Picamera2 not available here:", e, file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp.jpg")
    if args.save_full:
        Path(args.save_full).mkdir(parents=True, exist_ok=True)

    # --- Camera setup: sensor max resolution for stills ---
    picam2 = Picamera2()
    props = picam2.camera_properties
    try:
        max_w, max_h = tuple(props["PixelArraySize"])
    except Exception:
        # Fallback for IMX708 (CM3)
        max_w, max_h = 4608, 2592

    # Still config at full sensor res; add a lores stream for speedy preview if you like
    cfg = picam2.create_still_configuration(
        main={"size": (max_w, max_h)},
        buffer_count=2
    )
    picam2.configure(cfg)
    picam2.start()
    time.sleep(0.3)  # small settle

    # Lock focus at ∞ and set AE
    try:
        picam2.set_controls({
            "AfMode": controls.AfModeEnum.Manual,
            "LensPosition": 0.0,  # ~infinity
            "AeEnable": True,
            "AeMeteringMode": controls.AeMeteringModeEnum.Matrix,
            "ExposureValue": float(args.ev),
        })
    except Exception:
        pass

    print(f"[capture] full-res {max_w}x{max_h} → downsample to {args.down_width}x{args.down_height} "
          f"every {args.interval:.2f}s → {out_path}")
    if args.save_full:
        print(f"[capture] also saving full-res JPEGs to: {args.save_full}")

    next_shot = time.time()

    try:
        while True:
            # Grab a full-res still as RGB
            rgb_full = picam2.capture_array("main")
            bgr_full = cv2.cvtColor(rgb_full, cv2.COLOR_RGB2BGR)

            # Optional rotate
            if args.rotate:
                rotflag = {90: cv2.ROTATE_90_CLOCKWISE,
                           180: cv2.ROTATE_180,
                           270: cv2.ROTATE_90_COUNTERCLOCKWISE}[args.rotate]
                bgr_full = cv2.rotate(bgr_full, rotflag)

            # Downsample for inference (INTER_AREA = good quality for shrink)
            bgr_small = cv2.resize(
                bgr_full, (args.down_width, args.down_height),
                interpolation=cv2.INTER_AREA
            )

            now = time.time()
            if now >= next_shot:
                # 1) atomically write the downsampled frame to shm
                ok = cv2.imwrite(str(tmp_path), bgr_small,
                                 [int(cv2.IMWRITE_JPEG_QUALITY), int(args.quality)])
                if ok:
                    os.replace(str(tmp_path), str(out_path))
                    print(f"[capture] wrote downsampled @ {time.strftime('%H:%M:%S')}")
                else:
                    print("[capture] WARNING: failed to write downsampled temp jpeg", file=sys.stderr)

                # 2) optionally save the full-res jpeg to disk
                if args.save_full:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    full_file = Path(args.save_full) / f"full_{ts}.jpg"
                    cv2.imwrite(str(full_file), bgr_full, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

                next_shot = now + max(0.2, args.interval)

            # Live preview (downsampled)
            if args.preview:
                cv2.imshow("Full→Downsampled (q to quit)", bgr_small)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
            else:
                time.sleep(0.02)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            picam2.stop()
        except Exception:
            pass
        if args.preview:
            cv2.destroyAllWindows()
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        print("[capture] stopped.")

if __name__ == "__main__":
    main()
