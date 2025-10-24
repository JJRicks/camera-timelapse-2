#!/usr/bin/env python3
import argparse, time, sys
from pathlib import Path

import numpy as np
import cv2

# Prefer the small tflite_runtime package; fall back to TF if present
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter  # noqa: F401


def letterbox(im, new_shape, color=(114, 114, 114)):
    """Resize with unchanged aspect ratio, add padding (like Ultralytics)."""
    h, w = im.shape[:2]
    new_w, new_h = new_shape
    r = min(new_w / w, new_h / h)
    scaled = cv2.resize(im, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = new_w - scaled.shape[1], new_h - scaled.shape[0]
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    out = cv2.copyMakeBorder(scaled, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return out, r, (left, top)


def draw_boxes(frame, boxes_xyxy, scores, cls_ids, names, color=(0, 255, 0)):
    for (x1, y1, x2, y2), s, c in zip(boxes_xyxy, scores, cls_ids):
        lbl = names[int(c)] if int(c) < len(names) else str(int(c))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        txt = f"{lbl} {s:.2f}"
        (tw, th), bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, txt, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def parse_args():
    ap = argparse.ArgumentParser("Realtime TFLite YOLO preview")
    ap.add_argument("--model", required=True, help="Path to .tflite model")
    ap.add_argument("--labels", default=None, help="Text file with one class per line (default: ['car'])")
    ap.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    ap.add_argument("--threads", type=int, default=4, help="TFLite threads")
    ap.add_argument("--device", default="auto",
                    help="Video source: 'auto', '/dev/video0', integer index like '0', or 'picam'")
    ap.add_argument("--width", type=int, default=1280, help="Capture width (for USB/V4L)")
    ap.add_argument("--height", type=int, default=720, help="Capture height (for USB/V4L)")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270], help="Rotate display")
    return ap.parse_args()


def get_video_source(args):
    """Return a frame grabber. Supports:
       - 'picam' (Picamera2 live frames)
       - file:/path/to/image.jpg  (poll a still image that keeps being rewritten)
       - V4L/USB paths or indexes via cv2.VideoCapture
    """
    # 1) Picamera2 live feed
    if args.device == "picam":
        try:
            from picamera2 import Picamera2
        except Exception as e:
            print("Picamera2 not available:", e, file=sys.stderr)
            sys.exit(1)
        picam2 = Picamera2()
        cfg = picam2.create_video_configuration(main={"size": (args.width, args.height)}, buffer_count=4)
        picam2.configure(cfg)
        picam2.start()

        def grab():
            arr = picam2.capture_array()  # RGB
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        return grab, lambda: picam2.stop()

    # 2) File source: "file:/dev/shm/picam.jpg"  OR a plain existing path "/dev/shm/picam.jpg"
    from pathlib import Path as _Path
    import os as _os, time as _time
    if isinstance(args.device, str) and (args.device.startswith("file:") or _Path(args.device).exists()):
        p = args.device.removeprefix("file:")
        p = str(_Path(p).expanduser().resolve())
        print(f"[file source] Watching {p} for updates…")

        def grab():
            # Wait until the file exists
            while not _os.path.exists(p):
                _time.sleep(0.05)

            # Wait for modification time to change (avoid reading mid-write)
            last = getattr(grab, "_last_mtime_ns", 0)
            try:
                mtime = _os.stat(p).st_mtime_ns
            except AttributeError:
                # py<3.8 fallback (not needed here, but harmless)
                mtime = int(_os.stat(p).st_mtime * 1e9)

            while mtime == last:
                _time.sleep(0.01)
                try:
                    mtime = _os.stat(p).st_mtime_ns
                except AttributeError:
                    mtime = int(_os.stat(p).st_mtime * 1e9)

            grab._last_mtime_ns = mtime

            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to read image file: {p}")
            return img

        return grab, (lambda: None)

    # 3) Generic: V4L/USB or stream URL via OpenCV
    src = 0 if args.device in ("auto", "0", 0) else args.device
    if isinstance(src, str) and src.isdigit():
        src = int(src)

    cap = cv2.VideoCapture(src)
    if args.device != "picam":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"Failed to open video source: {args.device}", file=sys.stderr)
        sys.exit(1)

    def grab():
        ok, f = cap.read()
        if not ok:
            raise RuntimeError("Camera read failed")
        return f

    return grab, lambda: cap.release()


def main():
    args = parse_args()

    # Expand & verify model path
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        print(f"Model not found: {model_path}", file=sys.stderr)
        sys.exit(2)

    # Labels
    names = ["car"]
    if args.labels:
        p = Path(args.labels).expanduser()
        if p.exists():
            names = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]

    # Interpreter
    interp = Interpreter(model_path=str(model_path), num_threads=max(1, args.threads))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()
    in_idx = in_det["index"]
    in_shape = in_det["shape"]  # [1, H, W, 3]
    in_h, in_w = int(in_shape[1]), int(in_shape[2])
    in_dtype = in_det["dtype"]
    in_scale, in_zero = in_det.get("quantization", (0.0, 0))
    print(f"[model] input shape={tuple(in_shape)}, dtype={in_dtype}, quant=(scale={in_scale}, zero={in_zero})")
    print(f"[model] outputs: {[ (d['shape'], d['dtype']) for d in out_det ]}")

    # --- identify output head style ---
    # A) TFLite 'Detection_PostProcess' -> 4 outputs (boxes, classes, scores, count)
    # B) Ultralytics single tensor -> 1 output (N,6) or (N,7)
    head_mode = None

    def _is_boxes(d):   sh = d["shape"]; return len(sh) == 3 and sh[-1] == 4
    def _is_count(d):   sh = d["shape"]; return len(sh) == 2 and sh[-1] == 1
    def _is_float2(d):  sh = d["shape"]; return len(sh) == 2 and d["dtype"] in (np.float32, np.float16)
    def _is_int2(d):    sh = d["shape"]; return len(sh) == 2 and np.issubdtype(d["dtype"], np.integer)

    boxes_i = scores_i = classes_i = count_i = None

    if len(out_det) == 4:
        for i, d in enumerate(out_det):
            if _is_boxes(d):   boxes_i   = i
            elif _is_count(d): count_i   = i
            elif _is_int2(d):  classes_i = i
            elif _is_float2(d) and scores_i is None:
                scores_i = i
        if None not in (boxes_i, scores_i, classes_i, count_i):
            head_mode = "tflite_post"

    if head_mode is None and len(out_det) == 1:
        if out_det[0]["shape"][-1] in (6, 7):
            head_mode = "ultra_nx6"

    if head_mode is None:
        print(f"Unsupported TFLite outputs: {[ (d['shape'], d['dtype']) for d in out_det ]}", file=sys.stderr)
        sys.exit(1)

    # Video
    grab, cleanup = get_video_source(args)

    # For FPS
    t0 = time.time()
    frames = 0
    fps = 0.0

    try:
        while True:
            frame = grab()
            if args.rotate:
                rotflag = {90: cv2.ROTATE_90_CLOCKWISE,
                           180: cv2.ROTATE_180,
                           270: cv2.ROTATE_90_COUNTERCLOCKWISE}[args.rotate]
                frame = cv2.rotate(frame, rotflag)

            # Letterbox to model input
            lb, r, (dw, dh) = letterbox(frame, (in_w, in_h))
            x = lb.astype(np.float32)

            # Quantize / normalize
            if in_dtype == np.float32:
                x = x / 255.0
            elif in_dtype == np.uint8:
                real = x / 255.0
                if in_scale > 0:
                    x = np.clip(np.round(real / in_scale + in_zero), 0, 255).astype(np.uint8)
                else:
                    x = real.astype(np.uint8)  # fallback
            elif in_dtype == np.int8:
                real = x / 255.0
                x = np.clip(np.round(real / in_scale + in_zero), -128, 127).astype(np.int8)
            else:
                x = x.astype(in_dtype)

            x = np.expand_dims(x, 0)
            interp.set_tensor(in_idx, x)
            t_infer0 = time.time()
            interp.invoke()
            t_infer1 = time.time()

            # --- read outputs (supports both head styles) ---
            if head_mode == "tflite_post":
                boxes = interp.get_tensor(out_det[boxes_i]["index"])[0]      # [N,4] (ymin,xmin,ymax,xmax) normalized
                classes = interp.get_tensor(out_det[classes_i]["index"])[0]  # [N]
                scores = interp.get_tensor(out_det[scores_i]["index"])[0]    # [N]
                count = int(interp.get_tensor(out_det[count_i]["index"])[0]) # scalar

                sel = (scores[:count] >= args.conf)
                boxes = boxes[:count][sel]
                classes = classes[:count][sel].astype(int)
                scores = scores[:count][sel]

                # convert to xyxy in model-input pixels
                boxes_xyxy = np.empty_like(boxes)
                boxes_xyxy[:, 0] = boxes[:, 1] * in_w  # xmin
                boxes_xyxy[:, 1] = boxes[:, 0] * in_h  # ymin
                boxes_xyxy[:, 2] = boxes[:, 3] * in_w  # xmax
                boxes_xyxy[:, 3] = boxes[:, 2] * in_h  # ymax

            else:  # head_mode == "ultra_nx6"
                arr = interp.get_tensor(out_det[0]["index"])[0]  # (N,6) or (N,7)
                if arr.size == 0:
                    boxes_xyxy = np.empty((0, 4), dtype=np.float32)
                    scores = np.empty((0,), dtype=np.float32)
                    classes = np.empty((0,), dtype=np.int32)
                else:
                    if arr.shape[-1] == 6:
                        raw_boxes, scores, cls_raw = arr[:, :4], arr[:, 4], arr[:, 5]
                    else:  # (N,7) -> [batch, x1, y1, x2, y2, score, cls]
                        raw_boxes, scores, cls_raw = arr[:, 1:5], arr[:, 5], arr[:, 6]

                    sel = scores >= args.conf
                    raw_boxes, scores, cls_raw = raw_boxes[sel], scores[sel], cls_raw[sel]

                    # guess coord format: xyxy vs cxcywh
                    if raw_boxes.shape[0] > 0:
                        xyxy_like = (raw_boxes[:, 2] >= raw_boxes[:, 0]).mean() > 0.8 and \
                                    (raw_boxes[:, 3] >= raw_boxes[:, 1]).mean() > 0.8
                    else:
                        xyxy_like = True

                    if xyxy_like:
                        boxes_xyxy = raw_boxes.copy()
                    else:
                        cx, cy, w, h = raw_boxes.T
                        boxes_xyxy = np.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], axis=1)

                    # normalized 0..1 or already pixels?
                    if boxes_xyxy.size and boxes_xyxy.max() <= 1.05:
                        boxes_xyxy[:, [0, 2]] *= in_w
                        boxes_xyxy[:, [1, 3]] *= in_h

                    classes = np.rint(cls_raw).astype(int)

            # Map boxes back to original frame (from letterboxed model-input space)
            result_xyxy = []
            for (x1i, y1i, x2i, y2i) in boxes_xyxy:
                x1 = int((x1i - dw) / r); y1 = int((y1i - dh) / r)
                x2 = int((x2i - dw) / r); y2 = int((y2i - dh) / r)
                x1 = max(0, min(frame.shape[1] - 1, x1))
                y1 = max(0, min(frame.shape[0] - 1, y1))
                x2 = max(0, min(frame.shape[1] - 1, x2))
                y2 = max(0, min(frame.shape[0] - 1, y2))
                if x2 > x1 and y2 > y1:
                    result_xyxy.append((x1, y1, x2, y2))

            draw_boxes(frame, result_xyxy, scores, classes, names)

            frames += 1
            if frames % 10 == 0:
                dt = time.time() - t0
                fps = frames / dt if dt > 0 else 0.0

            # overlay timing
            ms = (t_infer1 - t_infer0) * 1000.0
            cv2.putText(frame, f"{model_path.name}  |  inf: {ms:.1f} ms  |  FPS: {fps:.2f}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("TFLite YOLO Preview (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):  # q or ESC
                break
    finally:
        cleanup()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
