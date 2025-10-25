import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import json

APP_TITLE = "854x480 Quad Annotator"
CANVAS_W, CANVAS_H = 854, 480

SQUARE_SIZE = 60     # initial square size for the 7 starter shapes
GAP = 10             # gap between starter squares
MARGIN = 10          # margin from top-left for starter squares
HANDLE_R = 6         # radius of draggable corner handles
POLY_OUTLINE = "#00b7ff"
HANDLE_FILL = "#ffffff"
HANDLE_OUTLINE = "#000000"
LABEL_FILL = "#ffffff"   # number label color

class QuadAnnotator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.resizable(False, False)

        # ----- Layout: left = canvas, right = controls / JSON -----
        container = tk.Frame(self)
        container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(container, width=CANVAS_W, height=CANVAS_H, bg="#222222", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nw")

        right = tk.Frame(container)
        right.grid(row=0, column=1, sticky="ns")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(2, weight=1)

        # Buttons
        btn_open = tk.Button(right, text="Open Image…", command=self.open_image)
        btn_open.grid(row=0, column=0, pady=(6, 4), padx=8, sticky="ew")

        btn_copy = tk.Button(right, text="Copy JSON", command=self.copy_json)
        btn_copy.grid(row=1, column=0, pady=(0, 6), padx=8, sticky="ew")

        # JSON text box (read-only)
        self.json_text = tk.Text(right, width=42, height=26, wrap="none")
        self.json_text.grid(row=2, column=0, padx=8, pady=(0, 8), sticky="nsew")
        self.json_text.configure(font=("Courier New", 10))
        self.json_text_scroll_y = tk.Scrollbar(right, command=self.json_text.yview)
        self.json_text_scroll_y.grid(row=2, column=1, sticky="ns")
        self.json_text.configure(yscrollcommand=self.json_text_scroll_y.set)
        self.json_text_scroll_x = tk.Scrollbar(right, orient="horizontal", command=self.json_text.xview)
        self.json_text_scroll_x.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 8))
        self.json_text.configure(xscrollcommand=self.json_text_scroll_x.set)
        self.json_text.configure(state="disabled")

        # Data
        self._bg_image_id = None
        self._bg_photo = None
        # active_drag:
        #   ("corner", qi, hi) when dragging a single handle
        #   ("quad",   qi)     when dragging the whole quad by its label/outline
        self.active_drag = None
        self.drag_last = None  # (x, y) last mouse position during whole-quad drag
        self.quads = []  # list of dicts with: points, polygon_id, handle_ids, text_id, number

        # Event bindings for dragging handles
        self.canvas.tag_bind("handle", "<ButtonPress-1>", self.on_handle_press)
        self.canvas.tag_bind("handle", "<B1-Motion>", self.on_handle_drag)
        self.canvas.tag_bind("handle", "<ButtonRelease-1>", self.on_handle_release)

        # NEW: drag entire quad by its number label
        self.canvas.tag_bind("label", "<ButtonPress-1>", self.on_label_press)
        self.canvas.tag_bind("label", "<B1-Motion>", self.on_label_drag)
        self.canvas.tag_bind("label", "<ButtonRelease-1>", self.on_label_release)

        # Bonus: allow dragging by polygon outline as well
        self.canvas.tag_bind("quad", "<ButtonPress-1>", self.on_quad_press)
        self.canvas.tag_bind("quad", "<B1-Motion>", self.on_quad_drag)
        self.canvas.tag_bind("quad", "<ButtonRelease-1>", self.on_quad_release)

        # Create initial quads
        self.create_initial_quads()
        self.update_json_text()

        # Prompt to open an image shortly after start (optional)
        self.after(200, self.open_image)

    # --------------------- Image handling ---------------------

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Open image",
            filetypes=[
                ("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.tif;*.tiff;*.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            img = Image.open(path)
            # Respect orientation and squash to exact 854x480
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB").resize((CANVAS_W, CANVAS_H), Image.LANCZOS)
            self._bg_photo = ImageTk.PhotoImage(img)

            # Replace previous bg image if any
            if self._bg_image_id is not None:
                self.canvas.delete(self._bg_image_id)
            self._bg_image_id = self.canvas.create_image(0, 0, image=self._bg_photo, anchor="nw", tags=("bg",))

            # Ensure all quads sit above the background
            self.canvas.tag_lower("bg")
            for q in self.quads:
                self.canvas.tag_raise(q["polygon_id"])
                for hid in q["handle_ids"]:
                    self.canvas.tag_raise(hid)
                self.canvas.tag_raise(q["text_id"])

        except Exception as e:
            messagebox.showerror("Error", f"Couldn't open image:\n{e}")

    # --------------------- Quad creation & helpers ---------------------

    def create_initial_quads(self):
        """Create 7 side-by-side squares at the top-left."""
        self.quads.clear()
        # Optional: clear any existing items except background
        for item in self.canvas.find_all():
            if "bg" not in self.canvas.gettags(item):
                self.canvas.delete(item)

        for i in range(7):
            x0 = MARGIN + i * (SQUARE_SIZE + GAP)
            y0 = MARGIN
            points = [
                (x0, y0),  # top-left
                (x0 + SQUARE_SIZE, y0),  # top-right
                (x0 + SQUARE_SIZE, y0 + SQUARE_SIZE),  # bottom-right
                (x0, y0 + SQUARE_SIZE),  # bottom-left
            ]
            poly_id = self.canvas.create_polygon(
                self._flatten(points),
                outline=POLY_OUTLINE,
                width=2,
                fill="",  # transparent so image shows through
                tags=("quad", f"quad{i+1}")
            )
            handle_ids = []
            for hi, (hx, hy) in enumerate(points):
                hid = self._create_handle(hx, hy)
                handle_ids.append(hid)

            cx, cy = self._centroid(points)
            text_id = self.canvas.create_text(
                cx, cy,
                text=str(i + 1),
                fill=LABEL_FILL,
                font=("Arial", 16, "bold"),
                tags=("label",)
            )

            self.quads.append({
                "number": i + 1,
                "points": points,
                "polygon_id": poly_id,
                "handle_ids": handle_ids,
                "text_id": text_id,
            })

        # Make sure everything sits above bg, if bg exists
        self.canvas.tag_lower("bg")
        for q in self.quads:
            self.canvas.tag_raise(q["polygon_id"])
            for hid in q["handle_ids"]:
                self.canvas.tag_raise(hid)
            self.canvas.tag_raise(q["text_id"])

    def _create_handle(self, x, y):
        return self.canvas.create_oval(
            x - HANDLE_R, y - HANDLE_R, x + HANDLE_R, y + HANDLE_R,
            outline=HANDLE_OUTLINE,
            width=1,
            fill=HANDLE_FILL,
            tags=("handle",)
        )

    @staticmethod
    def _flatten(points):
        flat = []
        for x, y in points:
            flat.extend([x, y])
        return flat

    @staticmethod
    def _centroid(points):
        # Average of vertices
        n = len(points)
        sx = sum(p[0] for p in points)
        sy = sum(p[1] for p in points)
        return (sx / n, sy / n)

    @staticmethod
    def _clamp(v, lo, hi):
        return max(lo, min(hi, v))

    # --------------------- Dragging logic: handles ---------------------

    def on_handle_press(self, event):
        item = event.widget.find_withtag("current")
        if not item:
            return
        handle_id = item[0]
        for qi, q in enumerate(self.quads):
            if handle_id in q["handle_ids"]:
                hi = q["handle_ids"].index(handle_id)
                self.active_drag = ("corner", qi, hi)
                break

    def on_handle_drag(self, event):
        if not self.active_drag or self.active_drag[0] != "corner":
            return
        _, qi, hi = self.active_drag

        # Clamp to canvas bounds
        x = self._clamp(event.x, 0, CANVAS_W)
        y = self._clamp(event.y, 0, CANVAS_H)

        q = self.quads[qi]
        q["points"][hi] = (x, y)

        # Update the moved handle graphics
        hid = q["handle_ids"][hi]
        self.canvas.coords(hid, x - HANDLE_R, y - HANDLE_R, x + HANDLE_R, y + HANDLE_R)

        # Update polygon path
        self.canvas.coords(q["polygon_id"], *self._flatten(q["points"]))

        # Update label (center)
        cx, cy = self._centroid(q["points"])
        self.canvas.coords(q["text_id"], cx, cy)

        # Refresh JSON box
        self.update_json_text()

    def on_handle_release(self, event):
        self.active_drag = None

    # --------------------- Dragging logic: whole-quad by label/outline ---------------------

    def on_label_press(self, event):
        item = event.widget.find_withtag("current")
        if not item:
            return
        label_id = item[0]
        for qi, q in enumerate(self.quads):
            if label_id == q["text_id"]:
                self.active_drag = ("quad", qi)
                self.drag_last = (event.x, event.y)
                break

    def on_label_drag(self, event):
        if not self.active_drag or self.active_drag[0] != "quad":
            return
        _, qi = self.active_drag
        lastx, lasty = self.drag_last
        dx, dy = event.x - lastx, event.y - lasty
        self._move_quad(qi, dx, dy)
        self.drag_last = (event.x, event.y)

    def on_label_release(self, event):
        self.active_drag = None
        self.drag_last = None

    # same behavior when dragging the polygon outline
    def on_quad_press(self, event):
        item = event.widget.find_withtag("current")
        if not item:
            return
        poly_id = item[0]
        for qi, q in enumerate(self.quads):
            if poly_id == q["polygon_id"]:
                self.active_drag = ("quad", qi)
                self.drag_last = (event.x, event.y)
                break

    def on_quad_drag(self, event):
        if not self.active_drag or self.active_drag[0] != "quad":
            return
        _, qi = self.active_drag
        lastx, lasty = self.drag_last
        dx, dy = event.x - lastx, event.y - lasty
        self._move_quad(qi, dx, dy)
        self.drag_last = (event.x, event.y)

    def on_quad_release(self, event):
        self.active_drag = None
        self.drag_last = None

    def _move_quad(self, qi, dx, dy):
        """Translate the whole quad by (dx, dy), clamped to the canvas."""
        if dx == 0 and dy == 0:
            return
        q = self.quads[qi]

        xs = [p[0] for p in q["points"]]
        ys = [p[1] for p in q["points"]]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)

        # Clamp translation so the quad stays fully inside the canvas
        if minx + dx < 0:
            dx = -minx
        if maxx + dx > CANVAS_W:
            dx = CANVAS_W - maxx
        if miny + dy < 0:
            dy = -miny
        if maxy + dy > CANVAS_H:
            dy = CANVAS_H - maxy

        if dx == 0 and dy == 0:
            return

        # Update points
        q["points"] = [(x + dx, y + dy) for (x, y) in q["points"]]

        # Update polygon
        self.canvas.coords(q["polygon_id"], *self._flatten(q["points"]))

        # Update handles
        for hi, hid in enumerate(q["handle_ids"]):
            x, y = q["points"][hi]
            self.canvas.coords(hid, x - HANDLE_R, y - HANDLE_R, x + HANDLE_R, y + HANDLE_R)

        # Update label (center)
        cx, cy = self._centroid(q["points"])
        self.canvas.coords(q["text_id"], cx, cy)

        # Refresh JSON
        self.update_json_text()

    # --------------------- JSON handling ---------------------

    def get_spaces_data(self):
        """Return dict like {"spaces": [[[x,y]...], ...]} with ints."""
        spaces = []
        # Ensure quads are in order 1..7
        for q in sorted(self.quads, key=lambda d: d["number"]):
            corners = [[int(round(x)), int(round(y))] for (x, y) in q["points"]]
            spaces.append(corners)
        return {"spaces": spaces}

    def update_json_text(self):
        data = self.get_spaces_data()
        s = json.dumps(data, indent=2)
        self.json_text.configure(state="normal")
        self.json_text.delete("1.0", "end")
        self.json_text.insert("1.0", s)
        self.json_text.configure(state="disabled")

    def copy_json(self):
        try:
            data = self.get_spaces_data()
            s = json.dumps(data, indent=2)
            self.clipboard_clear()
            self.clipboard_append(s)
            self.update()  # keep clipboard contents after app closes on some OS
        except Exception as e:
            messagebox.showerror("Error", f"Couldn't copy JSON:\n{e}")

if __name__ == "__main__":
    app = QuadAnnotator()
    app.mainloop()
