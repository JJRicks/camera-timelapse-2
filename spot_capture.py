#!/usr/bin/env python3
import argparse, sys, time, json, os
from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2

# Prefer tflite-runtime; fall back to TF if present
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter  # type: ignore

Point = Tuple[int, int]
Quad  = List[Point]  # 4 points, clockwise or counter-clockwise

# ----------------------------
# Geometry helpers (no shapely)
# ----------------------------
def polygon_area(poly: Quad) -> float:
    """Shoelace formula. Points in pixel coords."""
    x = np.array([p[0] for p in poly], dtype=np.float64)
    y = np.array([p[1] for p in poly], dtype=np.float64)
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

def clip_poly_halfplane(poly: List[Tuple[float,float]], a: float, b: float, c: float) -> List[Tuple[float,float]]:
    """Clip polygon with half-plane ax + by + c >= 0 (Sutherland–Hodgman)."""
    out = []
    if not poly: return out
    def inside(p): return a*p[0] + b*p[1] + c >= 0
    def intersect(p1, p2):
        d1 = a*p1[0] + b*p1[1] + c
        d2 = a*p2[0] + b*p2[1] + c
        t = d1 / (d1 - d2)
        return (p1[0] + t*(p2[0]-p1[0]), p1[1] + t*(p2[1]-p1[1]))
    S = poly[-1]
    for E in poly:
        if inside(E):
            if inside(S):
                out.append(E)
            else:
                out.append(intersect(S, E))
                out.append(E)
        else:
            if inside(S):
                out.append(intersect(S, E))
        S = E
    return out

def rect_poly_intersection_area(rect_xyxy: Tuple[int,int,int,int], poly: Quad) -> float:
    """Area of intersection between axis-aligned rect and polygon."""
    x1, y1, x2, y2 = rect_xyxy
    # start with polygon as floats
    P: List[Tuple[float,float]] = [(float(x), float(y)) for x,y in poly]
    # Clip with 4 rectangle edges: x >= x1, x <= x2, y >= y1, y <= y2
    # Half-planes: ax + by + c >= 0
    P = clip_poly_halfplane(P,  1, 0, -x1)  # x - x1 >= 0
    P = clip_poly_halfplane(P, -1, 0,  x2)  # -x + x2 >= 0
    P = clip_poly_halfplane(P,  0, 1, -y1)  # y - y1 >= 0
    P = clip_poly_halfplane(P,  0,-1,  y2)  # -y + y2 >= 0
    if len(P) < 3:
        return 0.0
    # area
    x = np.array([p[0] for p in P], dtype=np.float64)
    y = np.array([p[1] for p in P], dtype=np.float64)
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

# ----------------------------
# Ultralytics-style letterbox
# ----------------------------
def letterbox(im, new_shape, color=(114,114,114)):
    h, w = im.shape[:2]
    new_w, new_h = new_shape
    r = min(new_w / w, new_h / h)
    scaled = cv2.resize(im, (int(w*r), int(h*r)), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = new_w - scaled.shape[1], new_h - scaled.shape[0]
    top = pad_h // 2; bottom = pad_h - top
    left = pad_w // 2; right = pad_w - left
    out = cv2.copyMakeBorder(scaled, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return out, r, (left, top)

def draw_poly(img, poly: Quad, color, thickness=2, fill_alpha: float=None):
    pts = np.array(poly, dtype=np.int32).reshape(-1,1,2)
    if fill_alpha is not None and fill_alpha > 0:
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, fill_alpha, img, 1-fill_alpha, 0, img)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

# ------------------------------------------
# Source grabbers: picam / V4L / file tailer
# ------------------------------------------
def get_video_source(device, width, height):
    if device == "picam":
        try:
            from picamera2 import Picamera2
        except Exception as e:
            print("Picamera2 not available here:", e, file=sys.stderr)
            sys.exit(1)
        p = Picamera2()
        cfg = p.create_video_configuration(main={"size": (width, height)}, buffer_count=4)
        p.configure(cfg); p.start()
        def grab():
            arr = p.capture_array()  # RGB
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return grab, lambda: p.stop()
    if isinstance(device, str) and device.startswith("file:"):
        path = Path(device.split("file:",1)[1]).expanduser()
        if not path.exists():
            print(f"File mode: waiting for {path} ...")
        last_ns = -1
        def grab():
            nonlocal last_ns
            while True:
                try:
                    st = path.stat()
                    if st.st_mtime_ns != last_ns:
                        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
                        if img is not None:
                            last_ns = st.st_mtime_ns
                            return img
                except FileNotFoundError:
                    pass
                time.sleep(0.02)
        return grab, (lambda: None)
    # V4L/USB
    src = 0 if device in ("auto", "0", 0) else device
    if isinstance(src, str) and src.isdigit():
        src = int(src)
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        print(f"Failed to open video source: {device}", file=sys.stderr); sys.exit(1)
    def grab():
        ok, f = cap.read()
        if not ok: raise RuntimeError("Camera read failed")
        return f
    return grab, lambda: cap.release()

# ----------------------------
# Parse model outputs (2 styles)
# ----------------------------
def parse_head_details(out_det):
    # Try TFLite Detection_PostProcess: 4 outputs
    def _is_boxes(d):   sh = d["shape"]; return len(sh)==3 and sh[-1]==4
    def _is_count(d):   sh = d["shape"]; return len(sh)==2 and sh[-1]==1
    def _is_float2(d):  sh = d["shape"]; return len(sh)==2 and d["dtype"] in (np.float32, np.float16)
    def _is_int2(d):    sh = d["shape"]; return len(sh)==2 and np.issubdtype(d["dtype"], np.integer)
    boxes_i = scores_i = classes_i = count_i = None
    if len(out_det) == 4:
        for i,d in enumerate(out_det):
            if _is_boxes(d):   boxes_i = i
            elif _is_count(d): count_i = i
            elif _is_int2(d):  classes_i = i
            elif _is_float2(d) and scores_i is None: scores_i = i
        if None not in (boxes_i, scores_i, classes_i, count_i):
            return ("tflite_post", boxes_i, scores_i, classes_i, count_i)
    # Ultralytics single tensor: (1, N, 6) or (1, N, 7)
    if len(out_det)==1 and out_det[0]["shape"][-1] in (6,7):
        return ("ultra_nx6", 0, None, None, None)
    return (None, None, None, None, None)

# ----------------------------
# Main
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser("TFLite YOLO Parking Occupancy Overlay")
    ap.add_argument("--model", required=True, help="Path to .tflite model")
    ap.add_argument("--device", default="picam",
                    help="Video source: 'picam', int like '0', '/dev/video0', or 'file:/dev/shm/picam.jpg'")
    ap.add_argument("--width", type=int, default=854, help="Capture width")
    ap.add_argument("--height", type=int, default=480, help="Capture height")
    ap.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    ap.add_argument("--threads", type=int, default=4, help="TFLite threads")
    ap.add_argument("--labels", default=None, help="Optional labels.txt (one per line)")
    ap.add_argument("--spaces", default=None, help="JSON file with 7 spaces (each 4 points [x,y])")
    ap.add_argument("--overlap", type=float, default=0.5,
                    help="Min fraction of a box area inside space to count occupied (default 0.5)")
    ap.add_argument("--alpha", type=float, default=0.25, help="Fill alpha for polygons (0..1)")
    ap.add_argument("--rotate", type=int, choices=[0,90,180,270], default=0, help="Rotate preview")
    return ap.parse_args()

def load_spaces(args, frame_size) -> List[Quad]:
    W, H = frame_size
    if args.spaces:
        data = json.loads(Path(args.spaces).expanduser().read_text())
        # expect {"spaces":[ [[x,y],...[4]], ... ]}
        spaces = []
        for sp in data.get("spaces", []):
            quad = [(int(p[0]), int(p[1])) for p in sp]
            spaces.append(quad)
        return spaces
    # ---------- PLACEHOLDER DEMO COORDS (edit these!) ----------
    # A simple grid-ish layout for 7 spots in 854x480. Adjust for your view.
    # Order per spot: UL, UR, LR, LL.
    demo = [
        [(40,120),(140,120),(150,200),(30,200)],
        [(150,115),(250,115),(260,200),(150,200)],
        [(260,110),(360,110),(370,200),(260,200)],
        [(370,105),(470,105),(480,200),(370,200)],
        [(480,100),(580,100),(590,200),(480,200)],
        [(590,95),(690,95),(700,200),(590,200)],
        [(700,90),(800,90),(810,200),(700,200)],
    ]
    # Scale demo if user chose different W×H
    baseW, baseH = 854, 480
    sx, sy = W / baseW, H / baseH
    return [[(int(x*sx), int(y*sy)) for (x,y) in quad] for quad in demo]

def main():
    args = parse_args()

    # Labels
    names = ["car"]
    if args.labels:
        p = Path(args.labels).expanduser()
        if p.exists():
            names = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    car_class_ids = set([i for i,n in enumerate(names) if n.lower()=="car"])
    if not car_class_ids:
        car_class_ids = {0}  # fallback if your model is single-class (car)

    # Interpreter/model
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        print(f"Model not found: {model_path}", file=sys.stderr); sys.exit(2)
    interp = Interpreter(model_path=str(model_path), num_threads=max(1,args.threads))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()
    in_idx = in_det["index"]
    in_h, in_w = int(in_det["shape"][1]), int(in_det["shape"][2])
    in_dtype = in_det["dtype"]
    in_scale, in_zero = in_det.get("quantization",(0.0,0))
    head_mode, boxes_i, scores_i, classes_i, count_i = parse_head_details(out_det)
    if head_mode is None:
        print(f"Unsupported TFLite outputs: {[ (d['shape'], d['dtype']) for d in out_det ]}", file=sys.stderr)
        sys.exit(1)
    print(f"[model] input {(in_h,in_w)} dtype={in_dtype} head={head_mode}")

    # Video source
    grab, cleanup = get_video_source(args.device, args.width, args.height)

    # Spaces
    spaces = load_spaces(args, (args.width, args.height))
    if len(spaces) != 7:
        print(f"Loaded {len(spaces)} spaces (expected 7). Proceeding anyway.", file=sys.stderr)

    t0 = time.time(); frames=0; fps=0.0

    try:
        while True:
            frame = grab()
            if args.rotate:
                rotflag = {90:cv2.ROTATE_90_CLOCKWISE, 180:cv2.ROTATE_180, 270:cv2.ROTATE_90_COUNTERCLOCKWISE}[args.rotate]
                frame = cv2.rotate(frame, rotflag)

            # Preprocess (letterbox → model input)
            lb, r, (dw, dh) = letterbox(frame, (in_w, in_h))
            x = lb.astype(np.float32)
            if in_dtype == np.float32:
                x = x / 255.0
            elif in_dtype == np.uint8:
                real = x / 255.0
                x = (np.clip(np.round(real / max(in_scale,1e-9) + in_zero), 0, 255)).astype(np.uint8)
            elif in_dtype == np.int8:
                real = x / 255.0
                x = (np.clip(np.round(real / max(in_scale,1e-9) + in_zero), -128,127)).astype(np.int8)
            else:
                x = x.astype(in_dtype)
            x = np.expand_dims(x,0)
            interp.set_tensor(in_idx, x)
            t_in0 = time.time(); interp.invoke(); t_in1 = time.time()

            # Read outputs
            if head_mode == "tflite_post":
                boxes = interp.get_tensor(out_det[boxes_i]["index"])[0]      # [N,4] (ymin,xmin,ymax,xmax) normalized
                classes = interp.get_tensor(out_det[classes_i]["index"])[0]  # [N]
                scores = interp.get_tensor(out_det[scores_i]["index"])[0]    # [N]
                count = int(interp.get_tensor(out_det[count_i]["index"])[0])
                sel = (scores[:count] >= args.conf)
                boxes = boxes[:count][sel]; classes = classes[:count][sel].astype(int); scores = scores[:count][sel]
                # to xyxy in model-input pixels
                boxes_xyxy = np.empty_like(boxes); 
                boxes_xyxy[:, 0] = boxes[:,1] * in_w; boxes_xyxy[:,1] = boxes[:,0] * in_h
                boxes_xyxy[:, 2] = boxes[:,3] * in_w; boxes_xyxy[:,3] = boxes[:,2] * in_h
            else:  # ultra_nx6
                arr = interp.get_tensor(out_det[0]["index"])[0]  # (N,6) or (N,7)
                if arr.size == 0:
                    boxes_xyxy = np.empty((0,4), dtype=np.float32); scores=np.empty((0,),dtype=np.float32); classes=np.empty((0,),dtype=int)
                else:
                    if arr.shape[-1]==6:
                        raw_boxes, scores, cls_raw = arr[:, :4], arr[:,4], arr[:,5]
                    else:
                        raw_boxes, scores, cls_raw = arr[:, 1:5], arr[:,5], arr[:,6]
                    sel = scores >= args.conf
                    raw_boxes, scores, cls_raw = raw_boxes[sel], scores[sel], cls_raw[sel]
                    # guess format
                    if raw_boxes.shape[0]>0:
                        xyxy_like = (raw_boxes[:,2] >= raw_boxes[:,0]).mean()>0.8 and (raw_boxes[:,3] >= raw_boxes[:,1]).mean()>0.8
                    else:
                        xyxy_like = True
                    if xyxy_like:
                        boxes_xyxy = raw_boxes.copy()
                    else:
                        cx,cy,w,h = raw_boxes.T
                        boxes_xyxy = np.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], axis=1)
                    # normalized?
                    if boxes_xyxy.size and boxes_xyxy.max() <= 1.05:
                        boxes_xyxy[:,[0,2]] *= in_w; boxes_xyxy[:,[1,3]] *= in_h
                    classes = np.rint(cls_raw).astype(int)

            # Map boxes back to original frame pixels
            dets_xyxy: List[Tuple[int,int,int,int,int,float]] = []  # (x1,y1,x2,y2,cls,score)
            H, W = frame.shape[:2]
            for (x1i,y1i,x2i,y2i), c, s in zip(boxes_xyxy, classes, scores):
                x1 = int((x1i - dw)/r); y1 = int((y1i - dh)/r)
                x2 = int((x2i - dw)/r); y2 = int((y2i - dh)/r)
                x1 = max(0, min(W-1, x1)); x2 = max(0, min(W-1, x2))
                y1 = max(0, min(H-1, y1)); y2 = max(0, min(H-1, y2))
                if x2>x1 and y2>y1:
                    dets_xyxy.append((x1,y1,x2,y2,c,s))

            # --------------- occupancy logic ---------------
            occupied = []
            for idx, quad in enumerate(spaces, start=1):
                occ = False
                max_frac = 0.0
                for (x1,y1,x2,y2,c,s) in dets_xyxy:
                    if c not in car_class_ids: 
                        continue
                    rect_area = float((x2-x1)*(y2-y1))
                    if rect_area <= 0: 
                        continue
                    inter = rect_poly_intersection_area((x1,y1,x2,y2), quad)
                    frac = inter / rect_area
                    if frac > max_frac:
                        max_frac = frac
                if max_frac >= args.overlap:
                    occ = True
                if occ: occupied.append(idx)

            # --------------- draw overlays ---------------
            # draw boxes (optional, light styling)
            for (x1,y1,x2,y2,c,s) in dets_xyxy:
                if c not in car_class_ids: continue
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 1)
                cv2.putText(frame, f"car {s:.2f}", (x1, max(15,y1-4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1, cv2.LINE_AA)

            # spaces
            occ_set = set(occupied)
            for i, quad in enumerate(spaces, start=1):
                color = (0,0,255) if i in occ_set else (0,200,0)
                draw_poly(frame, quad, color, thickness=2, fill_alpha=args.alpha)
                # label near polygon centroid
                cx = int(sum(p[0] for p in quad)/4); cy = int(sum(p[1] for p in quad)/4)
                cv2.putText(frame, f"{i}", (cx-6, cy+6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"{i}", (cx-6, cy+6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

            empty = [i for i in range(1, len(spaces)+1) if i not in occ_set]
            occ_str   = ", ".join(map(str, occupied)) if occupied else "—"
            empty_str = ", ".join(map(str, empty))    if empty    else "—"
            status = f"Parking spaces occupied: {occ_str} | empty: {empty_str}"
            cv2.rectangle(frame, (0,0), (frame.shape[1], 28), (32,32,32), -1)
            cv2.putText(frame, status, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

            # timing
            frames += 1
            if frames % 10 == 0:
                dt = time.time()-t0; fps = frames/dt if dt>0 else 0.0
            ms = (t_in1 - t_in0) * 1000.0
            cv2.putText(frame, f"inf:{ms:.0f}ms  FPS:{fps:.2f}", (10, frame.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240,240,240), 1, cv2.LINE_AA)

            cv2.imshow("Parking occupancy (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
    finally:
        cleanup()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
