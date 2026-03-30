"""
FASP-UNet Brain Tumor Segmentation — Flask Web App  (v3)
Inputs : 4 × .nii.gz  (T1, T1ce, T2, FLAIR)
Outputs:
  - Viewer  : FLAIR input | Overlay | Mask-only  (3 panels)
  - Download: segmentation_*.nii.gz  (label map)
  - Download: mask_*.nii.gz          (binary mask per class)
  - Download: overlay PNG zip        (all overlay slices)
  - Download: mask-only PNG zip      (all mask-only slices)
"""

import os, sys, uuid, traceback, time, io
import numpy as np
import nibabel as nib
import cv2
import torch
from flask import (Flask, request, render_template,
                   jsonify, send_from_directory, abort, send_file)
from pathlib import Path
from scipy import ndimage

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0, os.path.dirname(__file__))
from model.fasp_unet import FASPUNet

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 512 * 1024 * 1024

BASE_DIR    = Path(__file__).parent
UPLOAD_DIR  = BASE_DIR / 'uploads'
RESULTS_DIR = BASE_DIR / 'static' / 'results'
MODEL_PATH  = BASE_DIR / 'model.pth'

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Settings ──────────────────────────────────────────────────────────────────
INFER_THRESH     = {'WT': 0.50, 'TC': 0.45, 'ET': 0.40}
MIN_SIZE         = {'WT': 80,   'TC': 50,   'ET': 30}
IMG_SIZE         = 240
CLASS_NAMES      = ['WT', 'TC', 'ET']
BATCH_SIZE_INFER = 8

# Overlay colours BGR
CLASS_COLORS_BGR = {
    'WT': (0,   220, 255),   # yellow
    'TC': (0,   140, 255),   # orange
    'ET': (0,   50,  220),   # red
}
# Mask-only colours BGR  (on black background)
CLASS_COLORS_MASK = {
    'WT': (80,  220, 255),
    'TC': (50,  140, 255),
    'ET': (50,  50,  220),
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = None

# ── Model loading ─────────────────────────────────────────────────────────────
def load_model():
    global model
    print(f"\n{'='*60}")
    print(f"Loading model : {MODEL_PATH}")
    print(f"Device        : {device}")

    if not MODEL_PATH.exists():
        print("✗  model.pth NOT FOUND")
        print(f"{'='*60}\n")
        return

    try:
        checkpoint = torch.load(str(MODEL_PATH), map_location=device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state = checkpoint['model_state_dict']
            print(f"  Epoch     : {checkpoint.get('epoch','?')}")
            print(f"  Tumor Dice: {checkpoint.get('best_tumor_dice','?')}")
        else:
            state = checkpoint

        m = FASPUNet(in_channels=4, num_classes=3, deep_supervision=True).to(device)
        missing, unexpected = m.load_state_dict(state, strict=False)
        if missing:
            print(f"  ⚠ Missing keys   : {len(missing)}")
        if unexpected:
            print(f"  ⚠ Unexpected keys: {len(unexpected)}")
        if not missing and not unexpected:
            print("  All keys matched ✓")

        m.eval()
        model = m
        print("✓ Model ready")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        traceback.print_exc()

    print(f"{'='*60}\n")


# ── Preprocessing ─────────────────────────────────────────────────────────────
def normalize_volume(vol: np.ndarray) -> np.ndarray:
    brain = vol[vol > 0]
    if brain.size == 0:
        return vol
    out = (vol - brain.mean()) / (brain.std() + 1e-8)
    out[vol == 0] = 0.0
    return out


def load_nii(path: Path) -> np.ndarray:
    print(f"    Loading {path.name} ...", end=' ', flush=True)
    t0  = time.time()
    vol = nib.load(str(path)).get_fdata().astype(np.float32)
    vol = normalize_volume(vol)
    print(f"shape={vol.shape}  {time.time()-t0:.1f}s")
    return vol


# ── Post-processing ───────────────────────────────────────────────────────────
def cc_filter(binary: np.ndarray, min_size: int) -> np.ndarray:
    labeled, n = ndimage.label(binary)
    if n == 0:
        return binary
    sizes   = np.bincount(labeled.ravel())
    keep    = sizes >= min_size
    keep[0] = False
    return keep[labeled].astype(np.float32)


def postprocess_slice(pred: np.ndarray) -> np.ndarray:
    """pred [3,H,W] → binary [3,H,W] with anatomical hierarchy."""
    result = np.zeros_like(pred)
    thresholds = [INFER_THRESH['WT'], INFER_THRESH['TC'], INFER_THRESH['ET']]
    min_sizes  = [MIN_SIZE['WT'],     MIN_SIZE['TC'],     MIN_SIZE['ET']]
    for c in range(3):
        result[c] = cc_filter(
            (pred[c] > thresholds[c]).astype(np.float32), min_sizes[c])
    result[1] = result[1] * result[0]   # TC ⊆ WT
    result[2] = result[2] * result[1]   # ET ⊆ TC
    return result


# ── Image builders ────────────────────────────────────────────────────────────
def norm_flair(flair_sl: np.ndarray) -> np.ndarray:
    """Resize + percentile-normalise FLAIR slice → [H,W] 0-255."""
    sl  = cv2.resize(flair_sl, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    p2, p98 = np.percentile(sl, [2, 98])
    return np.clip((sl - p2) / (p98 - p2 + 1e-8), 0, 1) * 255


def make_overlay_rgb(flair_norm: np.ndarray, mask_bin: np.ndarray) -> np.ndarray:
    """Coloured segmentation overlaid on FLAIR → [H,W,3] RGB."""
    base    = cv2.cvtColor(flair_norm.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    overlay = base.copy()
    for c, name in enumerate(CLASS_NAMES):
        overlay[mask_bin[c] > 0] = CLASS_COLORS_BGR[name]
    blended = cv2.addWeighted(base, 0.55, overlay, 0.45, 0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


def make_maskonly_rgb(mask_bin: np.ndarray) -> np.ndarray:
    """Coloured mask on pure black background → [H,W,3] RGB."""
    canvas = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    # Draw WT first (largest), then TC, then ET on top
    for c, name in enumerate(CLASS_NAMES):
        where = mask_bin[c] > 0
        canvas[where] = CLASS_COLORS_MASK[name]
    return canvas   # already RGB


def arr_to_b64(arr: np.ndarray) -> str:
    import base64
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def arr_to_png_bytes(arr: np.ndarray) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format='PNG')
    return buf.getvalue()


# ── Core inference ────────────────────────────────────────────────────────────
def run_inference(t1_p, t1ce_p, t2_p, flair_p, session_id: str) -> dict:
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"[{session_id}] INFERENCE START  device={device}")
    print(f"{'='*60}")

    # 1. Load
    print("\n[1/5] Loading volumes...")
    t1    = load_nii(t1_p)
    t1ce  = load_nii(t1ce_p)
    t2    = load_nii(t2_p)
    flair = load_nii(flair_p)
    total_slices = t1.shape[2]
    print(f"    Total slices: {total_slices}")

    # 2. Stack
    print(f"\n[2/5] Stacking slice tensors...")
    slices_np = np.zeros((total_slices, 4, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for si in range(total_slices):
        for ci, vol in enumerate([t1, t1ce, t2, flair]):
            slices_np[si, ci] = cv2.resize(
                vol[:, :, si], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    print(f"    Done  shape={slices_np.shape}")

    # 3. Batched inference (no TTA)
    n_batches = (total_slices + BATCH_SIZE_INFER - 1) // BATCH_SIZE_INFER
    print(f"\n[3/5] Inference  ({n_batches} batches × {BATCH_SIZE_INFER} slices)...")
    full_pred = np.zeros((total_slices, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)

    with torch.no_grad():
        for b in range(n_batches):
            s = b * BATCH_SIZE_INFER
            e = min(s + BATCH_SIZE_INFER, total_slices)
            tensor     = torch.tensor(slices_np[s:e]).to(device)
            logits, _  = model(tensor)
            full_pred[s:e] = torch.sigmoid(logits).cpu().numpy()

            if (b + 1) % 5 == 0 or b == n_batches - 1:
                pct = (b + 1) / n_batches * 100
                print(f"    Batch {b+1:3d}/{n_batches} ({pct:5.1f}%)  "
                      f"elapsed={time.time()-t0:.1f}s", flush=True)

    print(f"    Inference done  {time.time()-t0:.1f}s")

    # 4. Post-process + save .nii.gz
    print(f"\n[4/5] Post-processing + saving NIfTI...")
    final_mask = np.zeros((total_slices, 3, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    label_map  = np.zeros((IMG_SIZE, IMG_SIZE, total_slices), dtype=np.int16)

    for si in range(total_slices):
        pp = postprocess_slice(full_pred[si])
        final_mask[si] = (pp * 255).astype(np.uint8)
        lm = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.int16)
        lm[pp[0] > 0] = 1
        lm[pp[1] > 0] = 2
        lm[pp[2] > 0] = 3
        label_map[:, :, si] = lm

    nii_filename = f"segmentation_{session_id}.nii.gz"
    nib.save(nib.Nifti1Image(label_map, np.eye(4)),
             str(RESULTS_DIR / nii_filename))
    print(f"    Saved NIfTI → {nii_filename}")

    tumor_slices = [si for si in range(total_slices)
                    if np.any(final_mask[si] > 127)]
    print(f"    Tumour slices: {len(tumor_slices)} / {total_slices}")

    # 4b. Save binary mask .nii.gz  (3 channels: WT=ch0, TC=ch1, ET=ch2 as uint8 0/1)
    # Shape: [H, W, total_slices, 3]  — one volume per class
    print(f"    Saving binary mask NIfTI...")
    wt_vol = np.zeros((IMG_SIZE, IMG_SIZE, total_slices), dtype=np.uint8)
    tc_vol = np.zeros((IMG_SIZE, IMG_SIZE, total_slices), dtype=np.uint8)
    et_vol = np.zeros((IMG_SIZE, IMG_SIZE, total_slices), dtype=np.uint8)
    for si in range(total_slices):
        wt_vol[:, :, si] = (final_mask[si, 0] > 127).astype(np.uint8)
        tc_vol[:, :, si] = (final_mask[si, 1] > 127).astype(np.uint8)
        et_vol[:, :, si] = (final_mask[si, 2] > 127).astype(np.uint8)

    # Save as 4D NIfTI: shape [H, W, slices, 3]  channels = WT, TC, ET
    mask_4d = np.stack([wt_vol, tc_vol, et_vol], axis=-1).astype(np.uint8)
    mask_nii_filename = f"binary_mask_{session_id}.nii.gz"
    nib.save(nib.Nifti1Image(mask_4d, np.eye(4)),
             str(RESULTS_DIR / mask_nii_filename))
    print(f"    Saved binary mask → {mask_nii_filename}")

    # 5. Build display slice data (viewer only — no ZIPs)
    context = set()
    for si in tumor_slices:
        for d in range(-2, 3):
            if 0 <= si + d < total_slices:
                context.add(si + d)
    display_slices = sorted(context)

    print(f"\n[5/5] Rendering {len(display_slices)} slices for viewer...")

    slice_data = []
    for idx, si in enumerate(display_slices):
        fl_norm  = norm_flair(flair[:, :, si])
        mask_bin = (final_mask[si] > 127).astype(np.float32)

        overlay_rgb  = make_overlay_rgb(fl_norm, mask_bin)
        maskonly_rgb = make_maskonly_rgb(mask_bin)
        mri_rgb      = np.stack([fl_norm.astype(np.uint8)] * 3, axis=-1)

        slice_data.append({
            'slice_idx':   si,
            'mri_b64':     arr_to_b64(mri_rgb),
            'overlay_b64': arr_to_b64(overlay_rgb),
            'mask_b64':    arr_to_b64(maskonly_rgb),
            'class_presence': {
                'WT': bool(np.any(mask_bin[0] > 0)),
                'TC': bool(np.any(mask_bin[1] > 0)),
                'ET': bool(np.any(mask_bin[2] > 0)),
            },
        })

        if (idx + 1) % 20 == 0 or idx == len(display_slices) - 1:
            print(f"    Rendered {idx+1}/{len(display_slices)}", flush=True)

    total_time = time.time() - t0
    print(f"\n✓ [{session_id}] COMPLETE — {total_time:.1f}s\n")

    return {
        'nii_filename':      nii_filename,        # label map  (WT=1,TC=2,ET=3)
        'mask_nii_filename': mask_nii_filename,    # binary mask (4D WT/TC/ET channels)
        'total_slices':      total_slices,
        'tumor_slices':      tumor_slices,
        'display_count':     len(display_slices),
        'slice_data':        slice_data,
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded — check terminal.'}), 503

    for key in ['t1', 't1ce', 't2', 'flair']:
        if key not in request.files or request.files[key].filename == '':
            return jsonify({'error': f'Missing file: {key}'}), 400

    session_id = str(uuid.uuid4())[:8]
    sess_dir   = UPLOAD_DIR / session_id
    sess_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for key in ['t1', 't1ce', 't2', 'flair']:
        f    = request.files[key]
        dest = sess_dir / f'{key}.nii.gz'
        f.save(str(dest))
        print(f"  Upload saved: {dest.name}  ({dest.stat().st_size//1024} KB)")
        paths[key] = dest

    try:
        result = run_inference(
            paths['t1'], paths['t1ce'], paths['t2'], paths['flair'], session_id)
        return jsonify({'status': 'ok', **result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/download/<filename>')
def download(filename):
    allowed = (
        (filename.startswith('segmentation_') and filename.endswith('.nii.gz')) or
        (filename.startswith('binary_mask_')  and filename.endswith('.nii.gz'))
    )
    if not allowed:
        abort(403)
    return send_from_directory(str(RESULTS_DIR), filename, as_attachment=True)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    load_model()
    print(f"{'='*60}")
    print("  FASP-UNet  —  v3  (3-panel + dual download)")
    print(f"  Device : {device}")
    print(f"  Model  : {'LOADED ✓' if model else 'NOT FOUND ✗'}")
    print(f"  Batch  : {BATCH_SIZE_INFER} slices / forward pass")
    print("  URL    : http://127.0.0.1:5000")
    print(f"{'='*60}\n")
    app.run(debug=False, host='127.0.0.1', port=5000, threaded=True)