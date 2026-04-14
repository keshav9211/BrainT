
# train_brats_segnet_windows_cuda_v7_3_RTX4050_SAFE.py

import os, glob, json, random, time
from pathlib import Path
from typing import Optional, Tuple, List, Sequence

#HARD DEPENDENCY CHECKS 
def _need(pkg: str, pip: str):
    try:
        __import__(pkg)
    except Exception:
        raise RuntimeError(
            f"\n❌ Missing package: {pkg}\n"
            f"✅ Fix (inside your venv):\n"
            f"   python -m pip install {pip}\n"
        )

_need("numpy", "numpy")
_need("torch", "torch")
_need("monai", "monai")
_need("nibabel", "nibabel")

import numpy as np
import torch
from tqdm import tqdm
import nibabel as nib

from monai.data import DataLoader, Dataset, PersistentDataset, list_data_collate
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    Orientationd, Spacingd, NormalizeIntensityd, CropForegroundd,
    RandCropByPosNegLabeld, RandFlipd, RandRotate90d,
    RandScaleIntensityd, RandShiftIntensityd, RandGaussianNoised,
    RandAffined, MapTransform, Invertd, CopyItemsd
)
from monai.networks.nets import SegResNet
from monai.losses import DiceFocalLoss
from monai.inferers import sliding_window_inference
from monai.data.meta_tensor import MetaTensor

# OPTIONAL REALISM AUGS 
try:
    from monai.transforms import RandBiasFieldd, RandAdjustContrastd, RandGaussianSmoothd, RandGibbsNoised
    HAVE_REALISM_AUGS = True
except Exception:
    HAVE_REALISM_AUGS = False



# USER SETTINGS (safe to edit)

BRATS_GLI_ROOT = r"C:\Users\Ayushman\Desktop\Brats_gli"  # <-- set to your project root

TRAIN_FOLDER_NAME  = "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
EXTVAL_FOLDER_NAME = "ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData"  # usually unlabeled

# RTX 4050 8GB safe defaults:
SEED = 42

PATCH_SIZE   = (96, 96, 64)
VAL_ROI_SIZE = (96, 96, 64)     # 8GB-safe; if stable you can try (112,112,80)
SW_BATCH_SIZE = 1

BATCH_SIZE = 1
NUM_WORKERS = 1                # start 2; increase only if RAM ok
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2

# Caching:
# "stream" (default) -> lowest RAM
# "persistent" -> caches deterministic stage to disk (faster epochs, low RAM)
DATA_CACHE_MODE = "stream"       # "stream" or "persistent"
PERSIST_CACHE_SUBDIR = "persistent_cache"

MAX_EPOCHS = 150
MIN_EPOCHS = 15

PATIENCE = 5   # <-- YOU SAID YOU WANT 5 (change here)

LR = 2e-4
WEIGHT_DECAY = 1e-4
TRAIN_SPLIT = 0.9

USE_AMP = True
ACCUM_STEPS = 2                  # effective batch = 4 (stable, no extra VRAM)
GRAD_CLIP = 1.0

INIT_FILTERS = 16
DROPOUT = 0.1

BACKGROUND_WEIGHT = 0.1
ET_WEIGHT = 2.0

NUM_SAMPLES = 4
POS_SAMPLES = 3
NEG_SAMPLES = 1

USE_RANDAFFINE = True
AFFINE_PROB = 0.10

BIASFIELD_PROB = 0.25
CONTRAST_PROB  = 0.20
SMOOTH_PROB    = 0.15
GIBBS_PROB     = 0.10

ET_MIN_COMPONENT_SIZE = 30
ET_COLLAPSE_FLOOR = 0.05

# Native metrics (HD95/vol) are expensive; compute occasionally
COMPUTE_NATIVE_METRICS = False
NATIVE_METRICS_EVERY_N_EPOCHS = 5

TINY_GT_VOXELS = 10
HD_EMPTY_BOTH = 0.0
HD_EMPTY_ONE  = 999.0

# Export extval predictions (Synapse-ready)
DO_EXPORT_EXTVAL = True
EXPORT_EXTVAL_EVERY_N_EPOCHS = 5
EXPORT_FOLDER_NAME = "extval_predictions"

# Debug mode
DEBUG = False
DEBUG_TRAIN_BATCHES = 2
DEBUG_VAL_BATCHES = 2
DEBUG_EXTVAL_CASES = 2



# PATH RESOLUTION

def resolve_root() -> str:
    if BRATS_GLI_ROOT and os.path.isdir(BRATS_GLI_ROOT):
        return str(Path(BRATS_GLI_ROOT).resolve())
    here = Path(__file__).resolve()
    for p in [here.parent, here.parent.parent, here.parent.parent.parent]:
        if (p / "data").is_dir():
            return str(p)
    return str(here.parent.parent)

ROOT_DIR = resolve_root()
DATA_DIR = os.path.join(ROOT_DIR, "data")
DATA_TRAIN_DIR  = os.path.join(DATA_DIR, TRAIN_FOLDER_NAME)
DATA_EXTVAL_DIR = os.path.join(DATA_DIR, EXTVAL_FOLDER_NAME)

OUT_DIR = os.path.join(ROOT_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

SPLITS_PATH   = os.path.join(OUT_DIR, "splits.json")
METRICS_JSONL = os.path.join(OUT_DIR, "metrics.jsonl")

BEST_MEAN_ANY_PATH     = os.path.join(OUT_DIR, "best_mean_any.pt")
BEST_MEAN_GUARDED_PATH = os.path.join(OUT_DIR, "best_mean_guarded.pt")
BEST_ET_PATH           = os.path.join(OUT_DIR, "best_et.pt")
LAST_PATH              = os.path.join(OUT_DIR, "last.pt")


# UTILITIES

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def pick_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def enable_tf32_if_cuda():
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

def autocast_ctx(device):
    if device.type == "cuda" and USE_AMP:
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
    return torch.autocast(device_type="cpu", enabled=False)

def log_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")

def is_cuda_oom(e: Exception) -> bool:
    msg = str(e).lower()
    return ("out of memory" in msg) or ("cuda out of memory" in msg)

def dice_binary(pred: np.ndarray, gt: np.ndarray, eps=1e-6) -> float:
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)
    inter = (pred * gt).sum()
    return float((2 * inter + eps) / (pred.sum() + gt.sum() + eps))

def compute_brats_dice(pred_lbl: np.ndarray, gt_lbl: np.ndarray) -> Tuple[float, float, float]:
    wt = dice_binary(pred_lbl > 0, gt_lbl > 0)
    tc = dice_binary(np.isin(pred_lbl, [1, 3]), np.isin(gt_lbl, [1, 3]))
    et = dice_binary(pred_lbl == 3, gt_lbl == 3)
    return wt, tc, et

def region_masks(lbl: np.ndarray):
    wt = (lbl > 0)
    tc = np.isin(lbl, [1, 3])
    et = (lbl == 3)
    return wt, tc, et

def _to_numpy_any(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)):
        return np.array(x)
    return None

def native_spacing_from_meta(batch) -> Tuple[float, float, float]:
    # prefer pixdim then affine
    md = batch.get("t2w_native_meta_dict", None)
    if md is None and hasattr(batch.get("t2w_native", None), "meta"):
        md = batch["t2w_native"].meta
    if md is None:
        return (1.0, 1.0, 1.0)

    pixdim = _to_numpy_any(md.get("pixdim", None))
    if pixdim is not None:
        if pixdim.ndim == 2:
            pixdim = pixdim[0]
        if pixdim.shape[0] >= 4:
            return float(pixdim[1]), float(pixdim[2]), float(pixdim[3])

    aff = _to_numpy_any(md.get("affine", None))
    if aff is not None:
        if aff.ndim == 3:
            aff = aff[0]
        sp = np.sqrt((aff[:3, :3] ** 2).sum(axis=0))
        return float(sp[0]), float(sp[1]), float(sp[2])

    return (1.0, 1.0, 1.0)

def hd95_safe(pred_mask: np.ndarray, gt_mask: np.ndarray, spacing: Sequence[float]) -> float:
    pred_any = bool(pred_mask.any())
    gt_any   = bool(gt_mask.any())
    if (not pred_any) and (not gt_any):
        return float(HD_EMPTY_BOTH)
    if pred_any != gt_any:
        return float(HD_EMPTY_ONE)
    try:
        from monai.metrics import compute_hausdorff_distance
        p = torch.from_numpy(pred_mask[None, None].astype(np.bool_))
        g = torch.from_numpy(gt_mask[None, None].astype(np.bool_))
        hd = compute_hausdorff_distance(p, g, percentile=95.0, spacing=spacing)
        v = float(hd.detach().cpu().numpy().reshape(-1)[0])
        if not np.isfinite(v):
            return float(HD_EMPTY_ONE)
        return v
    except Exception:
        return float(HD_EMPTY_ONE)

def volume_error(pred_mask: np.ndarray, gt_mask: np.ndarray, tiny_gt: int = 10, eps=1e-6):
    vp = float(pred_mask.sum())
    vg = float(gt_mask.sum())
    abs_err = float(abs(vp - vg))
    if vg < tiny_gt:
        return abs_err, None
    return abs_err, float(abs_err / (vg + eps) * 100.0)

def mean_ignore_none(xs: List[Optional[float]]) -> Optional[float]:
    ys = [v for v in xs if v is not None]
    return float(np.mean(ys)) if len(ys) else None

def pct_coverage(xs: List[Optional[float]]) -> float:
    if not xs:
        return 0.0
    return float(sum(v is not None for v in xs) / len(xs))


# DATA DISCOVERY

def build_case_dicts(root_dir: str, require_seg: bool) -> List[dict]:
    root = Path(root_dir)
    if not root.exists():
        return []
    items = []
    for c in sorted([p for p in root.iterdir() if p.is_dir()]):
        t1n = glob.glob(str(c / "*-t1n.nii.gz")) + glob.glob(str(c / "*-t1n.nii"))
        t1c = glob.glob(str(c / "*-t1c.nii.gz")) + glob.glob(str(c / "*-t1c.nii"))
        t2f = glob.glob(str(c / "*-t2f.nii.gz")) + glob.glob(str(c / "*-t2f.nii"))
        t2w = glob.glob(str(c / "*-t2w.nii.gz")) + glob.glob(str(c / "*-t2w.nii"))

        seg_gz  = glob.glob(str(c / "*-seg.nii.gz"))
        seg_nii = glob.glob(str(c / "*-seg.nii"))
        seg = seg_gz[0] if seg_gz else (seg_nii[0] if seg_nii else None)

        if not (t1n and t1c and t2f and t2w):
            continue
        if require_seg and (seg is None):
            continue

        d = {"t1n": t1n[0], "t1c": t1c[0], "t2f": t2f[0], "t2w": t2w[0], "case_id": c.name}
        if seg is not None:
            d["seg"] = seg
        items.append(d)
    return items



# STABLE SPLITS

def make_or_load_splits(cases, splits_path, train_split=0.9, seed=42):
    id_to_case = {c["case_id"]: c for c in cases}

    if os.path.exists(splits_path):
        with open(splits_path, "r", encoding="utf-8") as f:
            sp = json.load(f)
        train_ids = sp.get("train", [])
        val_ids   = sp.get("val", [])
        missing = [cid for cid in (train_ids + val_ids) if cid not in id_to_case]
        if missing:
            raise RuntimeError(
                f"❌ splits.json has unknown case_ids (dataset changed). Delete outputs/splits.json and rerun.\n"
                f"Missing sample: {missing[:10]}"
            )
        return [id_to_case[cid] for cid in train_ids], [id_to_case[cid] for cid in val_ids], False

    rng = random.Random(seed)
    cases_copy = list(cases)
    rng.shuffle(cases_copy)
    split_idx = int(len(cases_copy) * train_split)
    train_cases, val_cases = cases_copy[:split_idx], cases_copy[split_idx:]

    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump({"train": [c["case_id"] for c in train_cases],
                   "val":   [c["case_id"] for c in val_cases]}, f, indent=2)

    return train_cases, val_cases, True



# LABEL REMAP (BraTS sometimes has label 4)

class RemapSeg4To3(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            if k not in d:
                continue
            seg = d[k]
            if torch.is_tensor(seg):
                out = seg.clone()
                out[out == 4] = 3
            else:
                out = np.array(seg, copy=True)
                out[out == 4] = 3
            d[k] = out
        return d



# ET CLEANUP (no scipy)

def cleanup_et_speckles(pred_lbl: np.ndarray, min_size: int = 30) -> np.ndarray:
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



# INFERENCE WITH CUDA OOM FALLBACK

def infer_with_roi_fallback(x, model, roi_size, sw_batch_size, device, overlap=0.25):
    try:
        with autocast_ctx(device):
            return sliding_window_inference(
                x, roi_size, sw_batch_size, model,
                overlap=overlap, mode="gaussian"
            )
    except RuntimeError as e:
        if device.type == "cuda" and is_cuda_oom(e):
            print("⚠️ CUDA OOM → fallback to PATCH_SIZE for inference")
            torch.cuda.empty_cache()
            with autocast_ctx(device):
                return sliding_window_inference(
                    x, PATCH_SIZE, sw_batch_size, model,
                    overlap=overlap, mode="gaussian"
                )
        raise



# INVERTERS (robust across MONAI versions)

def build_inverters(geom_transform):
    inv_a = Compose([
        Invertd(
            keys="pred",
            transform=geom_transform,
            orig_keys="t2w_native",
            nearest_interp=True,
            to_tensor=True,
        )
    ])

    inv_b = Compose([
        Invertd(
            keys="pred",
            transform=geom_transform,
            orig_keys="t2w_native",
            meta_keys="pred_meta_dict",
            orig_meta_keys="t2w_native_meta_dict",
            nearest_interp=True,
            to_tensor=True,
        )
    ])
    return inv_a, inv_b



# CHECKPOINT RESUME

def try_resume(model, optimizer, scheduler, scaler):
    if not os.path.exists(LAST_PATH):
        return 1, -1.0, -1.0, -1.0, 0

    try:
        ckpt = torch.load(LAST_PATH, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler", None) is not None and scaler is not None:
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception:
                pass

        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_mean_any = float(ckpt.get("best_mean_any", -1.0))
        best_mean_guarded = float(ckpt.get("best_mean_guarded", -1.0))
        best_et = float(ckpt.get("best_et", -1.0))
        patience_counter = int(ckpt.get("patience_counter", 0))

        print(
            f"🔁 Resumed epoch {start_epoch-1} | "
            f"best_any={best_mean_any:.4f} best_guarded={best_mean_guarded:.4f} best_et={best_et:.4f} "
            f"patience={patience_counter}"
        )
        return start_epoch, best_mean_any, best_mean_guarded, best_et, patience_counter

    except Exception as e:
        print(f"⚠️ Resume failed ({type(e).__name__}): starting fresh.")
        return 1, -1.0, -1.0, -1.0, 0



# EXPORT (Synapse-safe): invert -> native grid -> save affine

def save_pred_nifti(pred_native: np.ndarray, ref_meta: dict, out_path: str):
    affine = None
    if isinstance(ref_meta, dict):
        aff = ref_meta.get("affine", None)
        aff = _to_numpy_any(aff)
        if aff is not None and aff.shape == (4, 4):
            affine = aff
    if affine is None:
        affine = np.eye(4)  # fallback only

    img = nib.Nifti1Image(pred_native.astype(np.int16), affine)
    try:
        img.set_qform(affine, code=1)
        img.set_sform(affine, code=1)
    except Exception:
        pass
    nib.save(img, out_path)


@torch.no_grad()
def export_extval_predictions(model, device, extval_loader, inverter_a, inverter_b, epoch: int):
    model.eval()

    out_dir = os.path.join(OUT_DIR, EXPORT_FOLDER_NAME, f"epoch_{epoch:03d}")
    os.makedirs(out_dir, exist_ok=True)

    inverter_mode = "A"
    it = extval_loader
    if DEBUG:
        # take a few items
        small = []
        for i, b in enumerate(extval_loader):
            small.append(b)
            if i + 1 >= DEBUG_EXTVAL_CASES:
                break
        it = small

    for batch in tqdm(it, desc=f"Export EXTVAL (epoch {epoch})", leave=False):
        x = torch.cat([batch["t1n"], batch["t1c"], batch["t2f"], batch["t2w"]], dim=1).to(device, non_blocking=True)

        logits = infer_with_roi_fallback(x, model, VAL_ROI_SIZE, SW_BATCH_SIZE, device, overlap=0.25)
        pred_tr = torch.argmax(logits, dim=1).cpu().numpy()[0].astype(np.int16)
        pred_tr = cleanup_et_speckles(pred_tr, min_size=ET_MIN_COMPONENT_SIZE)

        # pred MetaTensor: use transformed t2w meta (has forward geom info)
        t2w_mt = batch["t2w"]
        pred_mt = MetaTensor(torch.from_numpy(pred_tr[None, None].astype(np.int64)), meta=t2w_mt.meta)

        try:
            inv = inverter_a({
                "pred": pred_mt,
                "t2w_native": batch["t2w_native"],
                "t2w_native_meta_dict": batch.get("t2w_native_meta_dict", getattr(batch["t2w_native"], "meta", {})),
            })
            inverter_mode = "A"
        except Exception:
            inv = inverter_b({
                "pred": pred_mt,
                "pred_meta_dict": pred_mt.meta,
                "t2w_native": batch["t2w_native"],
                "t2w_native_meta_dict": batch.get("t2w_native_meta_dict", getattr(batch["t2w_native"], "meta", {})),
            })
            inverter_mode = "B"

        pred_native = inv["pred"][0, 0].cpu().numpy().astype(np.int16)
        case_id = batch.get("case_id", ["unknown"])[0]
        out_path = os.path.join(out_dir, f"{case_id}.nii.gz")

        ref_meta = batch.get("t2w_native_meta_dict", getattr(batch["t2w_native"], "meta", {}))
        save_pred_nifti(pred_native, ref_meta, out_path)

    print(f"✅ EXTVAL export done | InvertdMode={inverter_mode} | folder: {out_dir}")


#
# MAIN

def main():
    set_seed(SEED)
    device = pick_device()
    enable_tf32_if_cuda()

    print(f"✅ Device: {device} | CUDA AMP: {USE_AMP} | TF32: {torch.backends.cuda.matmul.allow_tf32 if device.type=='cuda' else 'N/A'}")
    print(f"✅ ROOT_DIR: {ROOT_DIR}")
    print(f"✅ TRAIN_DIR: {DATA_TRAIN_DIR}")
    print(f"✅ EXTVAL_DIR: {DATA_EXTVAL_DIR}")
    print(f"✅ CacheMode: {DATA_CACHE_MODE} | workers={NUM_WORKERS} pin={PIN_MEMORY} | patience={PATIENCE}")

    # ---------------- DISCOVER ----------------
    train_all = build_case_dicts(DATA_TRAIN_DIR, require_seg=True)
    if len(train_all) < 10:
        raise RuntimeError(
            f"❌ Too few labeled training cases ({len(train_all)}).\n"
            f"Check folder:\n{DATA_TRAIN_DIR}"
        )
    print(f"✅ Labeled training cases found: {len(train_all)}")

    extval_all = build_case_dicts(DATA_EXTVAL_DIR, require_seg=False)
    extval_labeled = (len(build_case_dicts(DATA_EXTVAL_DIR, require_seg=True)) > 0)
    print(f"✅ External val cases found: {len(extval_all)} | labeled? {extval_labeled}")

    train_cases, val_cases, created = make_or_load_splits(
        cases=train_all, splits_path=SPLITS_PATH, train_split=TRAIN_SPLIT, seed=SEED
    )
    print(f"✅ Internal split: train={len(train_cases)} val={len(val_cases)} | splits: {'CREATED' if created else 'LOADED'}")
    print(f"✅ splits.json: {SPLITS_PATH}")

    # TRANSFORMS 
    # TRAIN deterministic (meta not needed -> faster)
    det_tf_train = Compose([
        LoadImaged(keys=["t1n","t1c","t2f","t2w","seg"]),
        EnsureChannelFirstd(keys=["t1n","t1c","t2f","t2w","seg"]),
        RemapSeg4To3(keys=["seg"]),
        Orientationd(keys=["t1n","t1c","t2f","t2w","seg"], axcodes="RAS", labels=None),
        Spacingd(keys=["t1n","t1c","t2f","t2w","seg"],
                 pixdim=(1.0,1.0,1.0),
                 mode=("bilinear","bilinear","bilinear","bilinear","nearest")),
        NormalizeIntensityd(keys=["t1n","t1c","t2f","t2w"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=["t1n","t1c","t2f","t2w","seg"], source_key="t2w", allow_smaller=True),
        EnsureTyped(keys=["t1n","t1c","t2f","t2w"], dtype=torch.float32, track_meta=False),
        EnsureTyped(keys=["seg"], dtype=torch.int64, track_meta=False),
    ])

    realism = []
    if HAVE_REALISM_AUGS:
        realism = [
            RandBiasFieldd(keys=["t1n","t1c","t2f","t2w"], prob=BIASFIELD_PROB, coeff_range=(0.0, 0.30)),
            RandAdjustContrastd(keys=["t1n","t1c","t2f","t2w"], prob=CONTRAST_PROB, gamma=(0.7, 1.5)),
            RandGaussianSmoothd(keys=["t1n","t1c","t2f","t2w"], prob=SMOOTH_PROB,
                                sigma_x=(0.5,1.5), sigma_y=(0.5,1.5), sigma_z=(0.5,1.5)),
            RandGibbsNoised(keys=["t1n","t1c","t2f","t2w"], prob=GIBBS_PROB, alpha=(0.3, 0.9)),
        ]

    affine = []
    if USE_RANDAFFINE:
        affine = [RandAffined(
            keys=["t1n","t1c","t2f","t2w","seg"],
            prob=AFFINE_PROB,
            rotate_range=(0.05,0.05,0.05),
            translate_range=(3,3,3),
            scale_range=(0.06,0.06,0.06),
            mode=("bilinear","bilinear","bilinear","bilinear","nearest"),
            padding_mode="border",
        )]

    rand_tf_train = Compose([
        *realism,
        RandCropByPosNegLabeld(
            keys=["t1n","t1c","t2f","t2w","seg"],
            label_key="seg",
            image_key="t2w",
            spatial_size=PATCH_SIZE,
            pos=POS_SAMPLES, neg=NEG_SAMPLES,
            num_samples=NUM_SAMPLES,
        ),
        RandFlipd(keys=["t1n","t1c","t2f","t2w","seg"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["t1n","t1c","t2f","t2w","seg"], prob=0.5, spatial_axis=1),
        RandRotate90d(keys=["t1n","t1c","t2f","t2w","seg"], prob=0.3, max_k=3),
        RandScaleIntensityd(keys=["t1n","t1c","t2f","t2w"], factors=0.1, prob=0.2),
        RandShiftIntensityd(keys=["t1n","t1c","t2f","t2w"], offsets=0.1, prob=0.2),
        RandGaussianNoised(keys=["t1n","t1c","t2f","t2w"], prob=0.12, mean=0.0, std=0.01),
        *affine,
    ])

    # VALIDATION: IMPORTANT -> keep meta BEFORE Spacingd/Orientationd to preserve affines
    val_tf = Compose([
        LoadImaged(keys=["t1n","t1c","t2f","t2w","seg"]),
        EnsureChannelFirstd(keys=["t1n","t1c","t2f","t2w","seg"]),
        RemapSeg4To3(keys=["seg"]),
        CopyItemsd(keys=["seg"], names=["seg_native"], times=1),
        CopyItemsd(keys=["t2w"], names=["t2w_native"], times=1),

        # ✅ KEEP META HERE (fixes your warning)
        EnsureTyped(keys=["t1n","t1c","t2f","t2w","seg","seg_native","t2w_native"], track_meta=True),

        Orientationd(keys=["t1n","t1c","t2f","t2w","seg"], axcodes="RAS", labels=None),
        Spacingd(keys=["t1n","t1c","t2f","t2w","seg"],
                 pixdim=(1.0,1.0,1.0),
                 mode=("bilinear","bilinear","bilinear","bilinear","nearest")),
        NormalizeIntensityd(keys=["t1n","t1c","t2f","t2w"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=["t1n","t1c","t2f","t2w","seg"], source_key="t2w", allow_smaller=True),
    ])

    # Geometry-only (must match the above geom ops)
    val_geom = Compose([
        Orientationd(keys=["t1n","t1c","t2f","t2w","seg"], axcodes="RAS", labels=None),
        Spacingd(keys=["t1n","t1c","t2f","t2w","seg"],
                 pixdim=(1.0,1.0,1.0),
                 mode=("bilinear","bilinear","bilinear","bilinear","nearest")),
        CropForegroundd(keys=["t1n","t1c","t2f","t2w","seg"], source_key="t2w", allow_smaller=True),
    ])
    inverter_a, inverter_b = build_inverters(val_geom)

    # EXTVAL (unlabeled): keep meta too (for correct native export)
    extval_tf = Compose([
        LoadImaged(keys=["t1n","t1c","t2f","t2w"]),
        EnsureChannelFirstd(keys=["t1n","t1c","t2f","t2w"]),
        CopyItemsd(keys=["t2w"], names=["t2w_native"], times=1),

        # ✅ KEEP META HERE TOO
        EnsureTyped(keys=["t1n","t1c","t2f","t2w","t2w_native"], track_meta=True),

        Orientationd(keys=["t1n","t1c","t2f","t2w"], axcodes="RAS", labels=None),
        Spacingd(keys=["t1n","t1c","t2f","t2w"],
                 pixdim=(1.0,1.0,1.0),
                 mode=("bilinear","bilinear","bilinear","bilinear")),
        NormalizeIntensityd(keys=["t1n","t1c","t2f","t2w"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=["t1n","t1c","t2f","t2w"], source_key="t2w", allow_smaller=True),
    ])

    extval_geom = Compose([
        Orientationd(keys=["t1n","t1c","t2f","t2w"], axcodes="RAS", labels=None),
        Spacingd(keys=["t1n","t1c","t2f","t2w"],
                 pixdim=(1.0,1.0,1.0),
                 mode=("bilinear","bilinear","bilinear","bilinear")),
        CropForegroundd(keys=["t1n","t1c","t2f","t2w"], source_key="t2w", allow_smaller=True),
    ])
    ext_inverter_a, ext_inverter_b = build_inverters(extval_geom)

    # ---------------- DATASETS ----------------
    if DATA_CACHE_MODE == "persistent":
        cache_dir = os.path.join(OUT_DIR, PERSIST_CACHE_SUBDIR)
        os.makedirs(cache_dir, exist_ok=True)
        print(f"🧊 PersistentDataset cache: {cache_dir}")

        det_ds = PersistentDataset(train_cases, transform=det_tf_train, cache_dir=cache_dir)
        train_ds = Dataset(det_ds, transform=rand_tf_train)  # random stage on top
    else:
        # streaming: deterministic+random composed
        train_ds = Dataset(train_cases, transform=Compose([det_tf_train, rand_tf_train]))

    val_ds = Dataset(val_cases, transform=val_tf)

    extval_loader = None
    if len(extval_all) > 0:
        extval_ds = Dataset(extval_all, transform=extval_tf)
        extval_loader = DataLoader(extval_ds, batch_size=1, shuffle=False, num_workers=0)

    # ---------------- LOADERS ----------------
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=list_data_collate,
        pin_memory=(PIN_MEMORY if device.type == "cuda" else False),
        persistent_workers=(PERSISTENT_WORKERS and NUM_WORKERS > 0),
        prefetch_factor=(PREFETCH_FACTOR if NUM_WORKERS > 0 else None),
    )

    # safest meta handling for val: 0 workers
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    #  MODEL
    model = SegResNet(
        spatial_dims=3, in_channels=4, out_channels=4,
        init_filters=INIT_FILTERS, dropout_prob=DROPOUT
    ).to(device)

    class_w = torch.tensor([float(BACKGROUND_WEIGHT), 1.0, 1.0, float(ET_WEIGHT)], device=device)
    loss_fn = DiceFocalLoss(to_onehot_y=True, softmax=True, gamma=2.0, weight=class_w)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=6)

    # ✅ NEW AMP API (no deprecation)
    scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and device.type == "cuda"))

    start_epoch, best_mean_any, best_mean_guarded, best_et, patience_counter = try_resume(
        model, optimizer, scheduler, scaler
    )

    # TRAIN LOOP 
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        t0 = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        used_steps = 0
        skipped_steps = 0
        accum = 0

        train_it = train_loader
        if DEBUG:
            small = []
            for i, b in enumerate(train_loader):
                small.append(b)
                if i + 1 >= DEBUG_TRAIN_BATCHES:
                    break
            train_it = small

        for batch in tqdm(train_it, desc=f"Epoch {epoch}/{MAX_EPOCHS}", leave=False):
            x = torch.cat([batch["t1n"], batch["t1c"], batch["t2f"], batch["t2w"]], dim=1).to(device, non_blocking=True)
            y = batch["seg"].to(device, non_blocking=True)

            with autocast_ctx(device):
                logits = model(x)

            loss = loss_fn(logits.float(), y)  # fp32 loss math

            if not torch.isfinite(loss):
                skipped_steps += 1
                optimizer.zero_grad(set_to_none=True)
                accum = 0
                continue

            scaler.scale(loss / ACCUM_STEPS).backward()
            running_loss += float(loss.item())
            used_steps += 1
            accum += 1

            if accum == ACCUM_STEPS:
                if GRAD_CLIP and GRAD_CLIP > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accum = 0

        if accum > 0:
            if GRAD_CLIP and GRAD_CLIP > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_loss = running_loss / max(1, used_steps)

        # VALIDATION 
        model.eval()

        dice_raw_wt, dice_raw_tc, dice_raw_et = [], [], []
        dice_pp_wt,  dice_pp_tc,  dice_pp_et  = [], [], []

        compute_native_now = (COMPUTE_NATIVE_METRICS and (epoch % NATIVE_METRICS_EVERY_N_EPOCHS == 0))
        hd_raw_wt, hd_raw_tc, hd_raw_et = [], [], []
        hd_pp_wt,  hd_pp_tc,  hd_pp_et  = [], [], []
        ve_raw_et_pct, ve_pp_et_pct = [], []

        val_it = val_loader
        if DEBUG:
            small = []
            for i, b in enumerate(val_loader):
                small.append(b)
                if i + 1 >= DEBUG_VAL_BATCHES:
                    break
            val_it = small

        inverter_mode = "A"

        with torch.no_grad():
            for batch in tqdm(val_it, desc="Validating", leave=False):
                x = torch.cat([batch["t1n"], batch["t1c"], batch["t2f"], batch["t2w"]], dim=1).to(device, non_blocking=True)
                gt_tr = batch["seg"].cpu().numpy()[0, 0].astype(np.int16)

                logits = infer_with_roi_fallback(x, model, VAL_ROI_SIZE, SW_BATCH_SIZE, device, overlap=0.25)
                pred_tr_raw = torch.argmax(logits, dim=1).cpu().numpy()[0].astype(np.int16)
                pred_tr_pp  = cleanup_et_speckles(pred_tr_raw, min_size=ET_MIN_COMPONENT_SIZE)

                wt_r, tc_r, et_r = compute_brats_dice(pred_tr_raw, gt_tr)
                wt_p, tc_p, et_p = compute_brats_dice(pred_tr_pp,  gt_tr)

                dice_raw_wt.append(wt_r); dice_raw_tc.append(tc_r); dice_raw_et.append(et_r)
                dice_pp_wt.append(wt_p);  dice_pp_tc.append(tc_p);  dice_pp_et.append(et_p)

                if not compute_native_now:
                    continue

                # invert pred -> native for true physical-space metrics
                t2w_mt = batch["t2w"]
                pred_mt_raw = MetaTensor(torch.from_numpy(pred_tr_raw[None, None].astype(np.int64)), meta=t2w_mt.meta)
                pred_mt_pp  = MetaTensor(torch.from_numpy(pred_tr_pp[None, None].astype(np.int64)),  meta=t2w_mt.meta)

                try:
                    inv_raw = inverter_a({
                        "pred": pred_mt_raw,
                        "t2w_native": batch["t2w_native"],
                        "t2w_native_meta_dict": batch.get("t2w_native_meta_dict", getattr(batch["t2w_native"], "meta", {})),
                    })
                    inv_pp = inverter_a({
                        "pred": pred_mt_pp,
                        "t2w_native": batch["t2w_native"],
                        "t2w_native_meta_dict": batch.get("t2w_native_meta_dict", getattr(batch["t2w_native"], "meta", {})),
                    })
                    inverter_mode = "A"
                except Exception:
                    inv_raw = inverter_b({
                        "pred": pred_mt_raw,
                        "pred_meta_dict": pred_mt_raw.meta,
                        "t2w_native": batch["t2w_native"],
                        "t2w_native_meta_dict": batch.get("t2w_native_meta_dict", getattr(batch["t2w_native"], "meta", {})),
                    })
                    inv_pp = inverter_b({
                        "pred": pred_mt_pp,
                        "pred_meta_dict": pred_mt_pp.meta,
                        "t2w_native": batch["t2w_native"],
                        "t2w_native_meta_dict": batch.get("t2w_native_meta_dict", getattr(batch["t2w_native"], "meta", {})),
                    })
                    inverter_mode = "B"

                pred_native_raw = inv_raw["pred"][0, 0].cpu().numpy().astype(np.int16)
                pred_native_pp  = inv_pp["pred"][0, 0].cpu().numpy().astype(np.int16)

                gt_native = batch["seg_native"].cpu().numpy()[0, 0].astype(np.int16)
                spacing_native = native_spacing_from_meta(batch)

                pr_wt, pr_tc, pr_et = region_masks(pred_native_raw)
                pp_wt, pp_tc, pp_et = region_masks(pred_native_pp)
                gt_wt, gt_tc, gt_et = region_masks(gt_native)

                hd_raw_wt.append(hd95_safe(pr_wt, gt_wt, spacing_native))
                hd_raw_tc.append(hd95_safe(pr_tc, gt_tc, spacing_native))
                hd_raw_et.append(hd95_safe(pr_et, gt_et, spacing_native))

                hd_pp_wt.append(hd95_safe(pp_wt, gt_wt, spacing_native))
                hd_pp_tc.append(hd95_safe(pp_tc, gt_tc, spacing_native))
                hd_pp_et.append(hd95_safe(pp_et, gt_et, spacing_native))

                # track ET volume % error only (most useful)
                _, pct_raw = volume_error(pr_et, gt_et, tiny_gt=TINY_GT_VOXELS)
                _, pct_pp  = volume_error(pp_et, gt_et, tiny_gt=TINY_GT_VOXELS)
                ve_raw_et_pct.append(pct_raw)
                ve_pp_et_pct.append(pct_pp)

        raw_wt_m = float(np.mean(dice_raw_wt)); raw_tc_m = float(np.mean(dice_raw_tc)); raw_et_m = float(np.mean(dice_raw_et))
        pp_wt_m  = float(np.mean(dice_pp_wt));  pp_tc_m  = float(np.mean(dice_pp_tc));  pp_et_m  = float(np.mean(dice_pp_et))
        mean3_raw = (raw_wt_m + raw_tc_m + raw_et_m) / 3.0
        mean3_pp  = (pp_wt_m  + pp_tc_m  + pp_et_m)  / 3.0

        scheduler.step(mean3_raw)
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:03d} | {elapsed:6.1f}s | LR {lr_now:.2e} | "
            f"Loss {train_loss:.4f} (used {used_steps}, skipped {skipped_steps}) | "
            f"DiceRAW WT {raw_wt_m:.4f} TC {raw_tc_m:.4f} ET {raw_et_m:.4f} Mean {mean3_raw:.4f} | "
            f"DicePP  WT {pp_wt_m:.4f} TC {pp_tc_m:.4f} ET {pp_et_m:.4f} Mean {mean3_pp:.4f}"
        )

        if compute_native_now and len(hd_raw_wt) > 0:
            hd_raw_mean = float(np.mean([np.mean(hd_raw_wt), np.mean(hd_raw_tc), np.mean(hd_raw_et)]))
            hd_pp_mean  = float(np.mean([np.mean(hd_pp_wt),  np.mean(hd_pp_tc),  np.mean(hd_pp_et)]))
            print(f"   NativeMetrics | InvertdMode={inverter_mode} | HD95 RAW mean {hd_raw_mean:.2f} | PP mean {hd_pp_mean:.2f}")

        # CHECKPOINTING
        improved_mean_raw = mean3_raw > (best_mean_any + 1e-4)
        improved_et_raw   = raw_et_m   > (best_et + 1e-4)

        if improved_et_raw:
            best_et = raw_et_m
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "best_et": best_et,
                "best_mean_any": best_mean_any,
                "best_mean_guarded": best_mean_guarded,
                "patience_counter": patience_counter,
            }, BEST_ET_PATH)
            print(f"⭐ Saved BEST_ET -> {BEST_ET_PATH} | best_et={best_et:.4f}")

        if improved_mean_raw:
            best_mean_any = mean3_raw
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "best_mean_any": best_mean_any,
                "best_mean_guarded": best_mean_guarded,
                "best_et": best_et,
                "patience_counter": patience_counter,
            }, BEST_MEAN_ANY_PATH)
            print(f"✅ Saved BEST_MEAN_ANY -> {BEST_MEAN_ANY_PATH} | best_mean_any={best_mean_any:.4f}")

        allow_guarded = (epoch < MIN_EPOCHS) or (raw_et_m >= ET_COLLAPSE_FLOOR)
        improved_guarded = mean3_raw > (best_mean_guarded + 1e-4)
        if improved_guarded and allow_guarded:
            best_mean_guarded = mean3_raw
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "best_mean_any": best_mean_any,
                "best_mean_guarded": best_mean_guarded,
                "best_et": best_et,
                "patience_counter": patience_counter,
            }, BEST_MEAN_GUARDED_PATH)
            print(f"🛡️ Saved BEST_MEAN_GUARDED -> {BEST_MEAN_GUARDED_PATH} | best_mean_guarded={best_mean_guarded:.4f}")

        if (epoch >= MIN_EPOCHS) and (not improved_mean_raw):
            patience_counter += 1
            print(f"⏳ No RAW mean improvement. Patience {patience_counter}/{PATIENCE}")
        else:
            patience_counter = 0

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_mean_any": best_mean_any,
            "best_mean_guarded": best_mean_guarded,
            "best_et": best_et,
            "patience_counter": patience_counter,
        }, LAST_PATH)

        log_jsonl(METRICS_JSONL, {
            "epoch": epoch,
            "train_loss": train_loss,
            "dice_raw_mean3": mean3_raw,
            "dice_pp_mean3": mean3_pp,
            "dice_raw_wt": raw_wt_m, "dice_raw_tc": raw_tc_m, "dice_raw_et": raw_et_m,
            "dice_pp_wt": pp_wt_m, "dice_pp_tc": pp_tc_m, "dice_pp_et": pp_et_m,
            "native_metrics_computed": compute_native_now,
            "inverter_mode": inverter_mode if compute_native_now else None,
            "lr": lr_now,
            "time_sec": elapsed,
            "best_mean_any": best_mean_any,
            "best_mean_guarded": best_mean_guarded,
            "best_et": best_et,
            "patience_counter": patience_counter,
        })

        # EXTVAL EXPORT 
        if DO_EXPORT_EXTVAL and extval_loader is not None:
            if (epoch % EXPORT_EXTVAL_EVERY_N_EPOCHS) == 0:
                export_extval_predictions(
                    model=model, device=device, extval_loader=extval_loader,
                    inverter_a=ext_inverter_a, inverter_b=ext_inverter_b, epoch=epoch
                )

        if patience_counter >= PATIENCE:
            print("⛔ Early stopping triggered.")
            break

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n🎯 Training finished.")
    print("Best RAW mean (any):", best_mean_any)
    print("Best RAW mean (guarded):", best_mean_guarded)
    print("Best RAW ET:", best_et)
    print("Outputs:", OUT_DIR)


if __name__ == "__main__":
    main()