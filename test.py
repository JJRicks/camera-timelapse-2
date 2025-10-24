#!/usr/bin/env python3
import argparse, time, cv2, numpy as np

def letterbox(img, new=640, color=(114,114,114)):
    h,w = img.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new - nh) // 2
    bottom = new - nh - top
    left = (new - nw) // 2
    right = new - nw - left
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, left, top

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--source", default="0", help="camera index or video path")
    args = ap.parse_args()

    cap = cv2.VideoCapture(0 if args.source.isdigit() else args.source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    net = cv2.dnn.readNetFromONNX(args.model)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    last = time.time()
    while True:
        ok, frame = cap.read()
        if not ok: break

        img, r, padw, padh = letterbox(frame, args.imgsz)
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (args.imgsz,args.imgsz), swapRB=True, crop=False)
        net.setInput(blob)
        out = net.forward()

        # Ultralytics ONNX with nms=True typically returns Nx6
        det = out.squeeze()
        if det.ndim == 1: det = det[None, :]  # handle 1x6
        for x1,y1,x2,y2,score,cls in det:
            if score < args.conf: continue
            # map back from letterbox to original frame
            x1 = int((x1 - padw) / r); y1 = int((y1 - padh) / r)
            x2 = int((x2 - padw) / r); y2 = int((y2 - padh) / r)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{int(cls)}:{score:.2f}", (x1, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

        now = time.time()
        fps = 1.0 / (now - last); last = now
        cv2.putText(frame, f"FPS: {fps:.2f}", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)

        cv2.imshow("ONNX live", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
