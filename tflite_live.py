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
    """Return a frame grabber. Supports USB/V4L and Picamera2."""
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
    else:
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

    # Labels
    names = ["car"]
    if args.labels:
        p = Path(args.labels)
        if p.exists():
            names = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]

    # Interpreter
    interp = Interpreter(model_path=args.model, num_threads=max(1, args.threads))
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

    # Expecting postprocess NMS (4 outputs). We'll auto-detect indices.
    boxes_i = scores_i = classes_i = count_i = None
    for i, d in enumerate(out_det):
        shp = d["shape"]
        if len(shp) == 2 and shp[1] == 1:
            count_i = i
        elif len(shp) == 3 and shp[2] == 4:
            boxes_i = i
        elif len(shp) == 2 and d["dtype"] in (np.float32, np.float16):
            # scores or classes; disambiguate by value type later
            # we'll set scores_i by picking the float array with maxDet length
            scores_i = i if scores_i is None else scores_i
        elif len(shp) == 2 and np.issubdtype(d["dtype"], np.integer):
            classes_i = i

    if None in (boxes_i, classes_i, scores_i, count_i):
        print("Model outputs not recognized as TF 'detection_postprocess'. "
              "Make sure you exported with nms=True.", file=sys.stderr)
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
                # quantized input expects real (0..1) scaled with (scale, zero)
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

            # Get outputs
            boxes = interp.get_tensor(out_det[boxes_i]["index"])[0]      # [N,4] ymin,xmin,ymax,xmax (normalized 0..1)
            classes = interp.get_tensor(out_det[classes_i]["index"])[0]  # [N]
            scores = interp.get_tensor(out_det[scores_i]["index"])[0]    # [N]
            count = int(interp.get_tensor(out_det[count_i]["index"])[0]) # scalar

            sel = (scores[:count] >= args.conf)
            boxes = boxes[:count][sel]
            classes = classes[:count][sel].astype(int)
            scores = scores[:count][sel]

            # Map boxes back to original frame
            # boxes were normalized in letterboxed space
            result_xyxy = []
            for (ymin, xmin, ymax, xmax) in boxes:
                x1 = int((xmin * in_w - dw) / r)
                y1 = int((ymin * in_h - dh) / r)
                x2 = int((xmax * in_w - dw) / r)
                y2 = int((ymax * in_h - dh) / r)
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
            cv2.putText(frame, f"{Path(args.model).name}  |  inf: {ms:.1f} ms  |  FPS: {fps:.2f}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("TFLite YOLO Preview (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):  # q or ESC
                break
    finally:
        cleanup()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
