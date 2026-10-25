"""
One-time script to draw bounding boxes on golden dataset query images.
Saves query_bbox: {x1, y1, x2, y2} into golden_dataset.json as you go.
Progress is preserved on quit — re-running skips already-annotated entries.

Controls:
  Click + drag  — draw box
  Enter         — confirm and save
  r             — redraw (clear current box)
  s             — skip this image (no bbox saved)
  q / Escape    — quit and save progress
"""
import io
import json
import tkinter as tk
import urllib.request
from pathlib import Path
from PIL import Image, ImageTk

DATASET_PATH  = Path(__file__).parent / "golden_dataset.json"
IMAGES_DIR    = Path(__file__).parent / "golden_images"
DISPLAY_MAX   = 700  # max display dimension


def load_dataset():
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def save_dataset(dataset):
    DATASET_PATH.write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def annotate():
    dataset = load_dataset()
    unannotated = [
        (i, e) for i, e in enumerate(dataset) if "query_bbox" not in e
    ]

    total     = len(dataset)
    already   = total - len(unannotated)
    skipped   = 0
    annotated = 0

    print(f"Golden dataset: {total} entries, {already} already annotated, {len(unannotated)} to go.")

    if not unannotated:
        print("All entries already annotated.")
        return

    root = tk.Tk()
    root.title("Bbox Annotator")

    canvas = tk.Canvas(root, cursor="crosshair", bg="black")
    canvas.pack(fill=tk.BOTH, expand=True)

    status_var = tk.StringVar()
    status_lbl = tk.Label(root, textvariable=status_var, anchor="w",
                          font=("Courier", 11), bg="#222", fg="#eee", padx=6)
    status_lbl.pack(fill=tk.X)

    help_lbl = tk.Label(
        root,
        text="drag=draw  Enter=confirm  r=redraw  s=skip  q=quit",
        font=("Courier", 9), bg="#333", fg="#aaa",
    )
    help_lbl.pack(fill=tk.X)

    state = {
        "idx":      0,
        "tk_img":   None,
        "scale":    1.0,
        "ox": 0, "oy": 0,         # drag origin
        "rect_id":  None,
        "box":      None,          # confirmed box (x1,y1,x2,y2) in image coords
        "done":     False,
    }

    def load_image(entry):
        url    = entry["query_image_url"]
        fname  = url.rsplit("/", 1)[-1]
        path   = IMAGES_DIR / fname
        if path.exists():
            img = Image.open(path).convert("RGB")
        else:
            try:
                data = urllib.request.urlopen(url, timeout=10).read()
                img = Image.open(io.BytesIO(data)).convert("RGB")
                print(f"  [FETCH] downloaded {fname} from gateway")
            except Exception as e:
                print(f"  [MISSING] {fname}: {e}")
                return None, 1.0
        w, h = img.size
        scale = min(DISPLAY_MAX / w, DISPLAY_MAX / h, 1.0)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return img, scale

    def flash_saved(name):
        """Show a bright SAVED banner for 800 ms, then load next entry."""
        canvas.delete("all")
        canvas.config(bg="#1a4a1a")
        cw = canvas.winfo_width() or 500
        ch = canvas.winfo_height() or 400
        canvas.create_text(cw // 2, ch // 2,
                           text=f"SAVED\n{name}",
                           fill="#00ff88", font=("Courier", 28, "bold"),
                           justify="center")
        status_var.set(f"  SAVED: {name}")
        root.after(800, lambda: _do_advance())

    def _do_advance():
        advance()

    def show_entry(n):
        idx, entry = unannotated[n]
        state["box"]     = None
        state["rect_id"] = None
        canvas.config(bg="black")
        img, scale = load_image(entry)
        name = entry.get("query_name", "?")
        if img is None:
            canvas.delete("all")
            cw = canvas.winfo_width() or 500
            ch = canvas.winfo_height() or 400
            canvas.create_text(cw // 2, ch // 2,
                               text=f"IMAGE NOT FOUND\n{name}\n\npress s to skip",
                               fill="#ff6644", font=("Courier", 18, "bold"),
                               justify="center")
            status_var.set(f"[{n+1}/{len(unannotated)}] MISSING — {name} — press s to skip")
            return
        state["scale"] = scale
        tk_img = ImageTk.PhotoImage(img)
        state["tk_img"] = tk_img
        cw, ch = img.width, img.height
        root.geometry(f"{cw}x{ch+50}")
        canvas.config(width=cw, height=ch)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=tk_img)
        status_var.set(f"[{n+1}/{len(unannotated)}] {name}")

    def on_press(event):
        state["ox"] = event.x
        state["oy"] = event.y
        if state["rect_id"]:
            canvas.delete(state["rect_id"])
        state["rect_id"] = canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline="#00ff00", width=2,
        )

    def on_drag(event):
        if state["rect_id"]:
            canvas.coords(state["rect_id"],
                          state["ox"], state["oy"], event.x, event.y)

    def on_release(event):
        x1 = min(state["ox"], event.x)
        y1 = min(state["oy"], event.y)
        x2 = max(state["ox"], event.x)
        y2 = max(state["oy"], event.y)
        if x2 - x1 < 5 or y2 - y1 < 5:
            state["box"] = None
            return
        sc = state["scale"]
        state["box"] = {
            "x1": round(x1 / sc),
            "y1": round(y1 / sc),
            "x2": round(x2 / sc),
            "y2": round(y2 / sc),
        }
        status_var.set(status_var.get() + f"  →  bbox {state['box']}  (Enter to confirm)")

    def advance():
        n = state["idx"]
        if n < len(unannotated) - 1:
            state["idx"] += 1
            show_entry(state["idx"])
        else:
            state["done"] = True
            canvas.delete("all")
            canvas.config(bg="#111")
            status_var.set("All done! Close the window.")
            print(f"\nFinished. annotated={annotated}  skipped={skipped}")

    def on_key(event):
        nonlocal annotated, skipped
        key = event.keysym

        if key in ("Return", "KP_Enter"):
            if state["box"] is None:
                status_var.set(status_var.get() + "  [draw a box first]")
                return
            real_idx = unannotated[state["idx"]][0]
            name = dataset[real_idx].get("query_name", "?")
            dataset[real_idx]["query_bbox"] = state["box"]
            save_dataset(dataset)
            annotated += 1
            print(f"  saved bbox for: {name}")
            flash_saved(name)

        elif key == "r":
            if state["rect_id"]:
                canvas.delete(state["rect_id"])
                state["rect_id"] = None
            state["box"] = None
            name = unannotated[state["idx"]][1].get("query_name", "?")
            status_var.set(f"[{state['idx']+1}/{len(unannotated)}] {name}")

        elif key == "s":
            skipped += 1
            print(f"  skipped: {unannotated[state['idx']][1].get('query_name','?')}")
            advance()

        elif key in ("q", "Escape"):
            print(f"\nQuitting. annotated={annotated}  skipped={skipped}  remaining={len(unannotated)-state['idx']-1}")
            root.destroy()

    canvas.bind("<ButtonPress-1>",   on_press)
    canvas.bind("<B1-Motion>",       on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    root.bind("<Key>", on_key)

    show_entry(0)
    root.mainloop()

    remaining = sum(1 for e in dataset if "query_bbox" not in e)
    print(f"\nSummary: {already + annotated} annotated, {skipped} skipped, {remaining} remaining")


if __name__ == "__main__":
    annotate()
