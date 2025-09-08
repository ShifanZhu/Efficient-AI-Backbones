#!/usr/bin/env python3
import os, argparse, re
from typing import Optional, Tuple, List
import pandas as pd
import torch
import timm
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

def load_model(model_name: str, num_classes: int, checkpoint: str, device: str):
    model = timm.create_model(model_name, num_classes=num_classes, pretrained=False)
    print(f"[info] loading checkpoint: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = None
    if isinstance(ckpt, dict):
        state = ckpt.get("state_dict") or ckpt.get("model")
    if state is None:
        state = ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:   print("[warn] missing keys (head mismatch is normal):", list(missing)[:10])
    if unexpected:print("[warn] unexpected keys:", list(unexpected)[:10])
    model.eval().to(device)
    return model

def make_transform(model, img_size: Optional[int]):
    cfg = resolve_data_config({}, model=model)
    if img_size is not None:
        c, _, _ = cfg["input_size"]
        cfg = dict(cfg); cfg["input_size"] = (c, img_size, img_size)
    tfm = create_transform(**cfg)
    print(f"[info] data_config: size={cfg['input_size']} mean={cfg['mean']} std={cfg['std']}")
    return tfm

def clamp_box(x1, y1, x2, y2, W, H):
    x1 = max(0, min(int(x1), W-1)); y1 = max(0, min(int(y1), H-1))
    x2 = max(0, min(int(x2), W-1)); y2 = max(0, min(int(y2), H-1))
    if x2 <= x1: x2 = min(W-1, x1+1)
    if y2 <= y1: y2 = min(H-1, y1+1)
    return x1, y1, x2, y2

def fix_filename(raw: str) -> str:
    """
    Normalize odd names like '806.0jpg' -> '806.jpg',
    but leave valid names (like '00806.jpg') unchanged.
    """
    s = str(raw).strip()
    s = s.replace("\\", "/").split("/")[-1]
    s = s.replace(" ", "")

    # If the string already matches a valid name with dot and extension, just return it
    if re.match(r".+\.(jpe?g|png|bmp|webp)$", s, re.IGNORECASE):
        return s

    # Handle bad cases like '1234.0jpg'
    s = re.sub(r"\.0(jpe?g|png|bmp|webp)$", r".\1", s, flags=re.IGNORECASE)

    # If ends with 'jpg' but missing dot before extension
    if re.search(r"(?i)(jpe?g|png|bmp|webp)$", s) and "." not in s.split(".")[-1]:
        s = re.sub(r"(?i)(jpe?g|png|bmp|webp)$", r".\1", s)

    # Default: if no extension at all, add .jpg
    if "." not in s:
        s = s + ".jpg"

    return s


def locate_image(test_dir: str, name: str) -> Optional[str]:
    """Try several guesses to find a file on disk."""
    cand = os.path.join(test_dir, name)
    if os.path.exists(cand): return cand
    # try stripping leading zeros
    base, ext = os.path.splitext(os.path.basename(name))
    if base.startswith("0"):
        nozero = base.lstrip("0") or "0"
        cand2 = os.path.join(test_dir, nozero + ext)
        if os.path.exists(cand2): return cand2
    # try any file that starts with the base (fallback)
    stem = base
    for fname in os.listdir(test_dir):
        if os.path.splitext(fname)[0] == stem:
            return os.path.join(test_dir, fname)
    return None

def load_image(path: str, bbox: Optional[Tuple[float,float,float,float]], use_bbox: bool) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if use_bbox and bbox is not None:
        W, H = img.size
        x1, y1, x2, y2 = clamp_box(*bbox, W, H)
        img = img.crop((x1, y1, x2, y2))
    return img

def batch_iter(items: List, bs: int):
    for i in range(0, len(items), bs):
        yield items[i:i+bs]

def main():
    ap = argparse.ArgumentParser("Cars test-set inference: fill Class in CSV with predicted indices")
    ap.add_argument("--data-root", required=True, help="root folder containing the test/ subfolder")
    ap.add_argument("--test-csv", required=True, help="CSV path with columns x1 y1 x2 y2 Class image")
    ap.add_argument("--test-subdir", default="test", help="subfolder under data-root where images live")
    ap.add_argument("--checkpoint", required=True, help="trained checkpoint (e.g., model_best.pth.tar)")
    ap.add_argument("--model-name", required=True, help="timm model name, e.g., resnet101 or tnt_s_patch16_224")
    ap.add_argument("--num-classes", type=int, required=True, help="e.g., 196")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--use-bbox", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-csv", default=None, help="output CSV; default: alongside input as test_pred.csv")
    args = ap.parse_args()

    model = load_model(args.model_name, args.num_classes, args.checkpoint, args.device)
    transform = make_transform(model, args.img_size)

    # auto-detect delimiter (handles tabs)
    df = pd.read_csv(args.test_csv, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]

    # required column check
    if "image" not in df.columns:
        raise RuntimeError(f"CSV is missing 'image' column. Found columns: {df.columns.tolist()}")

    test_dir = os.path.join(args.data_root, args.test_subdir)
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test folder not found: {test_dir}")

    # Build per-row info
    rows = []
    for i, r in df.iterrows():
        raw_name = r["image"]
        norm_name = fix_filename(raw_name)
        img_path = locate_image(test_dir, norm_name)
        if img_path is None:
            raise FileNotFoundError(f"Could not find image for CSV entry '{raw_name}' (normalized '{norm_name}') in {test_dir}")

        bbox = None
        if all(col in df.columns for col in ["x1","y1","x2","y2"]):
            x1, y1, x2, y2 = r.get("x1"), r.get("y1"), r.get("x2"), r.get("y2")
            if pd.notna(x1) and pd.notna(y1) and pd.notna(x2) and pd.notna(y2):
                try:
                    bbox = (float(x1), float(y1), float(x2), float(y2))
                except Exception:
                    bbox = None
        rows.append((i, img_path, bbox))

    # Inference
    if args.device.startswith("cuda"):
        autocast = torch.cuda.amp.autocast
    else:
        class _NoOp:
            def __enter__(self): return None
            def __exit__(self, *a): return False
        autocast = _NoOp

    preds = [None] * len(df)
    with torch.no_grad():
        for chunk in batch_iter(rows, args.batch_size):
            idxs, ims = [], []
            for (row_i, path, bbox) in chunk:
                img = load_image(path, bbox, args.use_bbox)
                ims.append(transform(img))
                idxs.append(row_i)
            batch = torch.stack(ims, 0).to(args.device, non_blocking=True)
            with autocast():
                logits = model(batch)
                top1 = logits.argmax(dim=-1)
            for row_i, cls_idx in zip(idxs, top1.detach().cpu().tolist()):
                preds[row_i] = int(cls_idx) + 1 # shift 0..195 -> 1..196

    # Fill Class column (create if missing)
    if "class" in df.columns:
        df["class"] = preds
    else:
        df.insert(len(df.columns), "class", preds)

    # Save
    out_csv = args.out_csv or os.path.join(os.path.dirname(args.test_csv), "test_pred.csv")
    df.to_csv(out_csv, index=False)
    print(f"[done] wrote predictions to: {out_csv}")
    try:
        print(df.head(6).to_string(index=False))
    except Exception:
        pass

if __name__ == "__main__":
    main()
# python inference.py   --data-root data/cars   --test-csv data/cars/test.csv   --checkpoint models/train/20250906-205057-tnt_s_patch16_224-224/model_best.pth.tar   --model-name resnet101   --num-classes 196   --batch-size 32   --use-bbox
