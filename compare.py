"""
Evaluate FASP-UNet predictions using binary_mask_*.nii.gz
Correct spatial alignment using nibabel resample_from_to
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nibabel.processing import resample_from_to


# ================================
# PATHS
# ================================
pred_mask_path = r"D:\project\static\results\binary_mask_3999c931.nii.gz"
gt_path        = r"E:\Project\brats_2023_organised_test\seg\BraTS-GLI-01660-000-seg.nii.gz"
t1c_path       = r"E:\Project\brats_2023_organised_test\t1c\BraTS-GLI-01660-000-t1c.nii.gz"


# ================================
# LOAD NIFTI
# ================================
pred_nii = nib.load(pred_mask_path)
gt_nii   = nib.load(gt_path)
img_nii  = nib.load(t1c_path)

pred_data = pred_nii.get_fdata().astype(np.uint8)   # [H,W,S,3]
gt_raw    = gt_nii.get_fdata().astype(np.int32)
img       = img_nii.get_fdata()

print("Pred shape:", pred_data.shape)
print("GT shape  :", gt_raw.shape)
print("IMG shape :", img.shape)


# ================================
# EXTRACT CHANNELS
# ================================
pred_WT_raw = pred_data[..., 0]
pred_TC_raw = pred_data[..., 1]
pred_ET_raw = pred_data[..., 2]


# ================================
# RESAMPLE TO GT SPACE (CORRECT)
# ================================
def resample_to_gt(vol):

    nii_tmp = nib.Nifti1Image(vol.astype(np.float32), pred_nii.affine)

    resampled = resample_from_to(
        nii_tmp,
        gt_nii,
        order=0  # nearest neighbour
    )

    return resampled.get_fdata().astype(bool)


pred_WT = resample_to_gt(pred_WT_raw)
pred_TC = resample_to_gt(pred_TC_raw)
pred_ET = resample_to_gt(pred_ET_raw)


# ================================
# GT REGIONS (BraTS)
# ================================
gt_WT = np.isin(gt_raw, [1, 2, 4])
gt_TC = np.isin(gt_raw, [1, 4])
gt_ET = (gt_raw == 4)


# ================================
# METRICS
# ================================
def compute_metrics(pred, gt):

    pred = pred.flatten()
    gt   = gt.flatten()

    TP = np.sum(pred & gt)
    TN = np.sum(~pred & ~gt)
    FP = np.sum(pred & ~gt)
    FN = np.sum(~pred & gt)

    dice = (2*TP)/(2*TP+FP+FN+1e-8)
    iou  = TP/(TP+FP+FN+1e-8)
    acc  = (TP+TN)/(TP+TN+FP+FN+1e-8)
    sens = TP/(TP+FN+1e-8)
    spec = TN/(TN+FP+1e-8)
    prec = TP/(TP+FP+1e-8)

    return dict(
        Dice=dice,
        IoU=iou,
        Accuracy=acc,
        Sensitivity=sens,
        Specificity=spec,
        Precision=prec,
    )


results = {}

for name, (p, g) in {
    "WT": (pred_WT, gt_WT),
    "TC": (pred_TC, gt_TC),
    "ET": (pred_ET, gt_ET),
}.items():

    m = compute_metrics(p, g)
    results[name] = m

    print("\n", name)
    for k, v in m.items():
        print(k, ":", round(v, 4))


# ================================
# SUMMARY
# ================================
print("\n==============================")
print("Metric        WT     TC     ET")

for metric in results["WT"].keys():

    w = results["WT"][metric]
    t = results["TC"][metric]
    e = results["ET"][metric]

    print(f"{metric:<12} {w:.3f}  {t:.3f}  {e:.3f}")


# ================================
# VISUALISATION
# ================================
def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


img_norm = normalize(img)


def overlay(image, wt, tc, et):

    rgb = np.stack([image]*3, axis=-1)

    rgb[wt] = [1,1,0]
    rgb[tc] = [1,0.5,0]
    rgb[et] = [1,0,0]

    return rgb


slice_idx = int(np.argmax(np.sum(gt_raw > 0, axis=(0,1))))

print("Slice:", slice_idx)

img_sl = img_norm[:,:,slice_idx]

gt_ol = overlay(
    img_sl,
    gt_WT[:,:,slice_idx],
    gt_TC[:,:,slice_idx],
    gt_ET[:,:,slice_idx],
)

pred_ol = overlay(
    img_sl,
    pred_WT[:,:,slice_idx],
    pred_TC[:,:,slice_idx],
    pred_ET[:,:,slice_idx],
)


fig, ax = plt.subplots(1,3,figsize=(15,5))

fig.suptitle(
    f"Slice {slice_idx} | Dice WT:{results['WT']['Dice']:.3f} "
    f"TC:{results['TC']['Dice']:.3f} "
    f"ET:{results['ET']['Dice']:.3f}"
)

ax[0].imshow(img_sl, cmap="gray")
ax[0].set_title("T1c")

ax[1].imshow(gt_ol)
ax[1].set_title("GT")

ax[2].imshow(pred_ol)
ax[2].set_title("Prediction")

for a in ax:
    a.axis("off")

plt.show()