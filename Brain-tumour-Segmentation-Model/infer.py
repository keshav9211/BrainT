
# What this does:
# 1) Loads 4 MRI modalities (t1n, t1c, t2f, t2w)
# 2) Applies deterministic preprocessing (RAS, 1mm spacing, normalize, crop)
# 3) Runs SegResNet with sliding-window inference
# 4) Postprocess: ET speckle cleanup (true 3D components, no scipy)
# 5) Inverts prediction back to native/original geometry using Invertd
# 6) Saves NIfTI with original affine (submission-safe)
# 7) Converts label 3 -> 4 for BraTS expected ET label
# 8) Writes ITK-SNAP workspace so overlay is automatic (no UI hunting)
#
# Output default:
#   Brats_gli/data/results/<case_id>.nii.gz
#   Brats_gli/data/results/<case_id>.itksnap   (overlay workspace)

# Usage:
#   python data/infer.py


import os
import sys
import glob
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import nibabel as nib
from tqdm import tqdm

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    CropForegroundd,
    CopyItemsd,
    Invertd,
)
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet
from monai.data.meta_tensor import MetaTensor



# DEFAULTS (safe on Mac MPS/CPU)

DEFAULT_ROI = (96, 96, 64)
DEFAULT_OVERLAP = 0.6
DEFAULT_SW_BATCH = 1
DEFAULT_MIN_ET_SIZE = 30

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # .../Brats_gli
DEFAULT_OUTDIR = PROJECT_ROOT / "data" / "results"
DEFAULT_CKPT = PROJECT_ROOT / "outputs" / "best_mean_guarded.pt"



# DEVICE PICKER

def pick_device() -> torch.device:
    # Prefer MPS if available, else CPU
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# PATH HELPERS (very important on macOS)

def norm_path(p: str) -> str:
    """
    Normalize path pasted by user:
    - remove surrounding quotes
    - expand ~
    - DO NOT silently change case (mac is case-sensitive often)
    """
    p = (p or "").strip().strip('"').strip("'")
    if not p:
        return ""
    return str(Path(p).expanduser())

def resolve_existing_path(p: str) -> str:
    """
    Resolve a path only if it exists. Keeps errors clean.
    """
    p2 = Path(norm_path(p))
    if p2.exists():
        return str(p2.resolve())
    return str(p2)



# CHECKPOINT RESOLVER (fixes your "Brats_" mistake)

def resolve_checkpoint(user_input: str) -> str:
    """
    Accepts:
      - empty => DEFAULT_CKPT
      - exact file path to .pt
      - folder path => try common checkpoint names inside
      - partial path / partial name => try to find a matching .pt in outputs/
    """
    ui = norm_path(user_input)

    # 1) Enter -> default
    if not ui:
        ck = Path(DEFAULT_CKPT)
        if ck.is_file():
            return str(ck.resolve())
        raise FileNotFoundError(f"Default checkpoint not found: {ck}")

    p = Path(ui)

    # 2) Exact file path
    if p.is_file():
        return str(p.resolve())

    # 3) If they pasted a folder, search inside
    if p.is_dir():
        candidates = [
            p / "best_mean_guarded.pt",
            p / "best_mean_any.pt",
            p / "best_et.pt",
            p / "last.pt",
        ]
        for c in candidates:
            if c.is_file():
                return str(c.resolve())
        # fallback: any .pt
        pts = sorted(p.glob("*.pt"))
        if pts:
            return str(pts[0].resolve())
        raise FileNotFoundError(f"No .pt checkpoint found inside folder: {p}")

    # 4) If path doesn't exist, try to recover:
    #    - if they typed partial name, search outputs/*.pt
    out_dir = PROJECT_ROOT / "outputs"
    if out_dir.is_dir():
        # if they typed something like "/Users/.../Brats_" take the basename part
        base = p.name
        # search by substring in checkpoint filenames
        pts = sorted(out_dir.glob("*.pt"))
        hit = None
        for c in pts:
            if base and base.lower() in c.name.lower():
                hit = c
                break
        if hit is not None:
            return str(hit.resolve())

        # if they typed something broken, still give default if exists
        if Path(DEFAULT_CKPT).is_file():
            print("⚠️ Could not match your checkpoint input; using default instead.")
            return str(Path(DEFAULT_CKPT).resolve())

    # if nothing worked, raise clean error
    raise FileNotFoundError(
        f"Checkpoint not found: {p}\n"
        f"Tip: press Enter for default OR paste the full path to a .pt file inside Brats_gli/outputs/"
    )



# CHECKPOINT LOADER (robust)

def load_model_ckpt(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> str:
    ckpt_path = str(Path(ckpt_path).expanduser().resolve())
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    obj = torch.load(ckpt_path, map_location="cpu")

    # your training saved {"model": state_dict, ...}
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        sd = obj["model"]
    elif isinstance(obj, dict):
        sd = obj  # raw state dict
    else:
        raise RuntimeError("Unrecognized checkpoint format.")

    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()
    return ckpt_path



# TRUE 3D ET speckle cleanup (6-connectivity BFS)
# removes small disconnected ET clusters (label=3 in model space)
# no scipy needed

def cleanup_et_speckles_cc3d(pred_lbl: np.ndarray, min_size: int = 30) -> np.ndarray:
    if min_size <= 0:
        return pred_lbl
    et = (pred_lbl == 3)
    if et.sum() == 0:
        return pred_lbl

    Z, Y, X = et.shape
    visited = np.zeros(et.shape, dtype=np.uint8)
    keep = np.zeros(et.shape, dtype=bool)

    nbrs = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

    def inb(z,y,x):
        return 0 <= z < Z and 0 <= y < Y and 0 <= x < X

    coords = np.argwhere(et)
    for z0, y0, x0 in coords:
        z0 = int(z0); y0 = int(y0); x0 = int(x0)
        if visited[z0,y0,x0]:
            continue

        stack = [(z0,y0,x0)]
        visited[z0,y0,x0] = 1
        comp = []

        while stack:
            z,y,x = stack.pop()
            comp.append((z,y,x))
            for dz,dy,dx in nbrs:
                zz,yy,xx = z+dz, y+dy, x+dx
                if inb(zz,yy,xx) and (not visited[zz,yy,xx]) and et[zz,yy,xx]:
                    visited[zz,yy,xx] = 1
                    stack.append((zz,yy,xx))

        if len(comp) >= min_size:
            for z,y,x in comp:
                keep[z,y,x] = True

    out = pred_lbl.copy()
    out[(pred_lbl == 3) & (~keep)] = 0
    return out



# BUILD TRANSFORMS
# - copy t2w -> t2w_native BEFORE geom ops
# - track_meta=True so Invertd can invert crop/spacing/orientation

def build_infer_transforms():
    infer_tf = Compose([
        LoadImaged(keys=["t1n", "t1c", "t2f", "t2w"]),
        EnsureChannelFirstd(keys=["t1n", "t1c", "t2f", "t2w"]),

        CopyItemsd(keys=["t2w"], names=["t2w_native"], times=1),
        EnsureTyped(keys=["t1n","t1c","t2f","t2w","t2w_native"], track_meta=True),

        Orientationd(keys=["t1n","t1c","t2f","t2w"], axcodes="RAS", labels=None),
        Spacingd(
            keys=["t1n","t1c","t2f","t2w"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear","bilinear","bilinear","bilinear"),
        ),
        NormalizeIntensityd(keys=["t1n","t1c","t2f","t2w"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=["t1n","t1c","t2f","t2w"], source_key="t2w", allow_smaller=True),
    ])

    geom_tf = Compose([
        Orientationd(keys=["t1n","t1c","t2f","t2w"], axcodes="RAS", labels=None),
        Spacingd(
            keys=["t1n","t1c","t2f","t2w"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear","bilinear","bilinear","bilinear"),
        ),
        CropForegroundd(keys=["t1n","t1c","t2f","t2w"], source_key="t2w", allow_smaller=True),
    ])

    inverter_a = Compose([
        Invertd(
            keys="pred",
            transform=geom_tf,
            orig_keys="t2w_native",
            nearest_interp=True,
            to_tensor=True,
        )
    ])

    inverter_b = Compose([
        Invertd(
            keys="pred",
            transform=geom_tf,
            orig_keys="t2w_native",
            meta_keys="pred_meta_dict",
            orig_meta_keys="t2w_native_meta_dict",
            nearest_interp=True,
            to_tensor=True,
        )
    ])

    return infer_tf, inverter_a, inverter_b



# INPUT DISCOVERY
# Supports:
#  A) input_root/
#        BraTS-GLI-xxxx/
#           *-t1n.nii.gz, *-t1c..., *-t2f..., *-t2w...
#  B) a single case folder directly

def find_case_files(case_dir: str) -> Dict[str, str]:
    c = Path(case_dir)
    if not c.is_dir():
        raise FileNotFoundError(f"Case folder not found: {case_dir}")

    def pick(patterns: List[str]) -> Optional[str]:
        hits = []
        for ptn in patterns:
            hits += glob.glob(str(c / ptn))
        return hits[0] if hits else None

    t1n = pick(["*-t1n.nii.gz", "*-t1n.nii"])
    t1c = pick(["*-t1c.nii.gz", "*-t1c.nii"])
    t2f = pick(["*-t2f.nii.gz", "*-t2f.nii"])
    t2w = pick(["*-t2w.nii.gz", "*-t2w.nii"])

    if not all([t1n, t1c, t2f, t2w]):
        raise RuntimeError(f"Missing modality in {case_dir} (need t1n,t1c,t2f,t2w)")

    return {"t1n": t1n, "t1c": t1c, "t2f": t2f, "t2w": t2w}


def list_case_dirs(input_root: str) -> List[str]:
    r = Path(input_root).expanduser().resolve()
    if not r.is_dir():
        raise FileNotFoundError(f"Input path not found: {r}")

    # If user gave a single folder that directly contains the NIfTIs, treat as one case
    nii_hits = list(r.glob("*-t2w.nii.gz")) + list(r.glob("*-t2w.nii"))
    if len(nii_hits) > 0:
        return [str(r)]

    # Else treat as root containing many case folders
    return [str(p) for p in sorted(r.iterdir()) if p.is_dir()]



# SAVE NIFTI (native affine)

def save_pred_nifti(pred_native: np.ndarray, t2w_native_meta: dict, out_path: str):
    affine = None
    if isinstance(t2w_native_meta, dict):
        aff = t2w_native_meta.get("affine", None)
        if aff is not None:
            aff = np.asarray(aff)
            if aff.shape == (4, 4):
                affine = aff
    if affine is None:
        affine = np.eye(4)

    img = nib.Nifti1Image(pred_native.astype(np.int16), affine)
    try:
        img.set_qform(affine, code=1)
        img.set_sform(affine, code=1)
    except Exception:
        pass
    nib.save(img, out_path)



# ITK-SNAP WORKSPACE (auto overlay)

def write_itksnap_workspace(t2w_path: str, seg_path: str, out_ws: str):
    """
    Creates an ITK-SNAP workspace file that opens:
      - background: T2w image
      - segmentation: predicted mask
    """
    t2w_path = str(Path(t2w_path).expanduser().resolve())
    seg_path = str(Path(seg_path).expanduser().resolve())
    out_ws = str(Path(out_ws).expanduser().resolve())

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<itksnapworkspace version="1">
  <layer type="main">
    <entry key="Filename" value="{t2w_path}"/>
  </layer>
  <layer type="segmentation">
    <entry key="Filename" value="{seg_path}"/>
  </layer>
</itksnapworkspace>
"""
    Path(out_ws).write_text(xml, encoding="utf-8")



# RUN ONE CASE

@torch.inference_mode()
def run_one_case(
    model: torch.nn.Module,
    device: torch.device,
    case_id: str,
    files: Dict[str, str],
    out_dir: str,
    roi: Tuple[int, int, int],
    overlap: float,
    sw_batch_size: int,
    min_et_size: int,
):
    infer_tf, inverter_a, inverter_b = build_infer_transforms()

    data = {"t1n": files["t1n"], "t1c": files["t1c"], "t2f": files["t2f"], "t2w": files["t2w"]}
    data = infer_tf(data)

    # model input: (1,4,Z,Y,X)
    x = torch.cat([data["t1n"], data["t1c"], data["t2f"], data["t2w"]], dim=0).unsqueeze(0).to(device)

    def do_infer(curr_roi, curr_sw_batch):
        return sliding_window_inference(
            x, curr_roi, curr_sw_batch, model, overlap=overlap, mode="gaussian"
        )

    try:
        logits = do_infer(roi, sw_batch_size)
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg or "memory" in msg:
            print(f"⚠️ OOM on {case_id} with ROI={roi}, sw_batch={sw_batch_size}. Retrying smaller...")
            if device.type == "cuda":
                torch.cuda.empty_cache()

            small_roi = (max(64, roi[0] // 2), max(64, roi[1] // 2), max(48, roi[2] // 2))
            try:
                logits = do_infer(small_roi, sw_batch_size)
            except RuntimeError:
                logits = do_infer(small_roi, 1)
        else:
            raise

    pred_tr = torch.argmax(logits, dim=1).detach().cpu().numpy()[0].astype(np.int16)

    # ET cleanup in model-space (ET=3)
    pred_tr = cleanup_et_speckles_cc3d(pred_tr, min_size=min_et_size)

    # invert to native using Invertd
    t2w_mt = data["t2w"]  # MetaTensor after geom transforms
    pred_mt = MetaTensor(torch.from_numpy(pred_tr[None, None].astype(np.int64)), meta=t2w_mt.meta)

    inverter_mode = "A"
    try:
        inv = inverter_a({
            "pred": pred_mt,
            "t2w_native": data["t2w_native"],
            "t2w_native_meta_dict": data.get("t2w_native_meta_dict", getattr(data["t2w_native"], "meta", {})),
        })
    except Exception:
        inverter_mode = "B"
        inv = inverter_b({
            "pred": pred_mt,
            "pred_meta_dict": pred_mt.meta,
            "t2w_native": data["t2w_native"],
            "t2w_native_meta_dict": data.get("t2w_native_meta_dict", getattr(data["t2w_native"], "meta", {})),
        })

    pred_native = inv["pred"][0, 0].detach().cpu().numpy().astype(np.int16)

    # BraTS expects ET label 4 (your model uses 3)
    pred_native[pred_native == 3] = 4

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{case_id}.nii.gz")

    ref_meta = data.get("t2w_native_meta_dict", getattr(data["t2w_native"], "meta", {}))
    save_pred_nifti(pred_native, ref_meta, out_path)

    # workspace for easy overlay in ITK-SNAP
    ws_path = os.path.join(out_dir, f"{case_id}.itksnap")
    try:
        write_itksnap_workspace(files["t2w"], out_path, ws_path)
    except Exception:
        pass

    return out_path, ws_path, inverter_mode



# PROMPT HELPERS

def prompt_path(msg: str) -> str:
    p = input(msg)
    return resolve_existing_path(p)

def prompt_int(msg: str, default: int) -> int:
    s = input(msg).strip()
    if not s:
        return default
    return int(s)

def prompt_float(msg: str, default: float) -> float:
    s = input(msg).strip()
    if not s:
        return default
    return float(s)

def prompt_roi(msg: str, default: Tuple[int,int,int]) -> Tuple[int,int,int]:
    s = input(msg).strip()
    if not s:
        return default
    parts = [int(x) for x in s.replace(" ", "").split(",")]
    if len(parts) != 3:
        raise ValueError("ROI must be like 96,96,64")
    return (parts[0], parts[1], parts[2])



# MAIN

def main():
    print("\n🧠 BraTS Inference (Mac) — FULL pipeline, simple prompts")
    print("Tip: you can paste paths with spaces. Just paste normally.\n")

    input_root = prompt_path("📂 Paste INPUT ROOT (root with many cases OR a single case folder):\n> ")

    ckpt_raw = input(f"\n🎯 Paste checkpoint path (Enter = default)\n[{DEFAULT_CKPT}]\n> ")
    ckpt_path = resolve_checkpoint(ckpt_raw)

    out_dir_in = input(f"\n💾 Output folder (Enter = default)\n[{DEFAULT_OUTDIR}]\n> ")
    out_dir = str(DEFAULT_OUTDIR) if not out_dir_in.strip() else str(Path(norm_path(out_dir_in)).expanduser().resolve())

    print("\n⚙️ Optional inference settings (press Enter to keep defaults)")
    roi = prompt_roi(f"ROI (Z,Y,X) like 96,96,64  [default {DEFAULT_ROI}]:\n> ", DEFAULT_ROI)
    overlap = prompt_float(f"Overlap (0.5–0.7 recommended)  [default {DEFAULT_OVERLAP}]:\n> ", DEFAULT_OVERLAP)
    sw_batch = prompt_int(f"sw_batch_size (keep 1 on Mac)  [default {DEFAULT_SW_BATCH}]:\n> ", DEFAULT_SW_BATCH)
    min_et = prompt_int(f"Min ET component size (voxels)  [default {DEFAULT_MIN_ET_SIZE}]:\n> ", DEFAULT_MIN_ET_SIZE)

    device = pick_device()

    print("\n========== RESOLVED SETTINGS ==========")
    print(f"✅ Device: {device}")
    print(f"✅ INPUT : {input_root}")
    print(f"✅ CKPT  : {ckpt_path}")
    print(f"✅ OUT   : {out_dir}")
    print(f"✅ ROI   : {roi} | overlap={overlap} | sw_batch={sw_batch} | min_et={min_et}")
    print("======================================\n")

    # Build model EXACTLY like training
    model = SegResNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        init_filters=16,
        dropout_prob=0.1
    )

    ckpt_loaded = load_model_ckpt(model, ckpt_path, device)
    print(f"✅ Loaded checkpoint: {ckpt_loaded}\n")

    case_dirs = list_case_dirs(input_root)
    if len(case_dirs) == 0:
        raise RuntimeError("No case folders found inside the input path.")

    print(f"🔎 Found {len(case_dirs)} case(s). Starting inference...\n")

    inv_modes = {"A": 0, "B": 0}
    ok, bad = 0, 0
    bad_cases: List[Tuple[str, str]] = []
    t0 = time.time()

    for case_path in tqdm(case_dirs, desc="Infer", unit="case"):
        case_id = Path(case_path).name
        try:
            files = find_case_files(case_path)
            out_path, ws_path, inv_mode = run_one_case(
                model=model,
                device=device,
                case_id=case_id,
                files=files,
                out_dir=out_dir,
                roi=roi,
                overlap=overlap,
                sw_batch_size=sw_batch,
                min_et_size=min_et,
            )
            inv_modes[inv_mode] += 1
            ok += 1
        except Exception as e:
            bad += 1
            bad_cases.append((case_id, f"{type(e).__name__}: {e}"))
            continue

    dt = time.time() - t0
    print("\n✅ DONE")
    print(f"Cases OK: {ok} | Failed: {bad} | Time: {dt/60:.1f} min")
    print("Invertd modes used:", inv_modes)
    print("Saved predictions in:", out_dir)
    print("Tip: open any .itksnap file in results to see overlay instantly.\n")

    if bad_cases:
        print("❌ Failed cases (first 10):")
        for cid, err in bad_cases[:10]:
            print(f"  - {cid}: {err}")
        print("\nTip: Failed usually means missing modality files or broken NIfTI headers.")


if __name__ == "__main__":
    main()