import glob
import os
import warnings
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from typing import Callable
from PIL import Image
from IPython.display import display
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torchvision import transforms
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import matplotlib.colors as mcolors
from IPython.display import SVG, display
import matplotlib.image as mpimg
import torch.nn.functional as F
import pandas as pd

mpl.rcParams['font.family'] = 'Arial'

warnings.filterwarnings("ignore", message="xFormers is not available")

############################################
# CATEGORY GROUPS & SELECTION
############################################

DATASET_DIR = "MVTec AD"

TEXTURE_CATEGORIES = ["carpet", "grid", "leather", "tile", "wood"]
OBJECT_CATEGORIES  = [
    "bottle", "cable", "capsule", "metal_nut", "metal_nut",
    "pill", "screw", "toothbrush", "transistor", "zipper",
]

KEY_TO_DISPLAY = {
    "carpet": "Carpet",
    "grid": "Grid",
    "leather": "Leather",
    "tile": "Tile",
    "wood": "Wood",
    "bottle": "Bottle",
    "cable": "Cable",
    "capsule": "Capsule",
    "hazelnut": "Hazelnut",
    "metal_nut": "Metal nut",
    "pill": "Pill",
    "screw": "Screw",
    "toothbrush": "Toothbrush",
    "transistor": "Transistor",
    "zipper": "Zipper",
}

ALL_CATEGORIES = TEXTURE_CATEGORIES + OBJECT_CATEGORIES

def get_categories(mode="all", user_category=None):
    """
    mode:
        - "single": use user_category (must be in ALL_CATEGORIES)
        - "objects": all object categories
        - "textures": all texture categories
        - "all": all categories
    """
    if mode == "single":
        if user_category not in ALL_CATEGORIES:
            raise ValueError(f"Unknown category: {user_category}")
        return [user_category]

    elif mode == "objects":
        return OBJECT_CATEGORIES

    elif mode == "textures":
        return TEXTURE_CATEGORIES

    elif mode == "all":
        return ALL_CATEGORIES

    else:
        raise ValueError("mode must be: 'single', 'objects', 'textures', or 'all'")

def get_category_paths(category):
    """
    Return train/test/GT dirs for a given MVTec category.
    """
    train_good_dir = os.path.join(DATASET_DIR, category, "train", "good")
    test_dir       = os.path.join(DATASET_DIR, category, "test")
    gt_dir         = os.path.join(DATASET_DIR, category, "ground_truth")
    return train_good_dir, test_dir, gt_dir

############################################
# DINOv3 model + preprocessing
############################################

REPO_DIR = "dinov3"
WEIGHTS = "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"


def load_dinov3_model(
    repo_dir: str = REPO_DIR,
    weights: str = WEIGHTS,
    arch: str = "dinov3_vits16",
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = torch.hub.load(
        repo_dir,
        arch,
        source="local",
        weights=weights,
    ).to(device).eval()

    return model, device


def build_preprocess(img_size: int):
    return transforms.Compose([
        transforms.Resize(
            (img_size, img_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


############################################
# PATCH EMBEDDING EXTRACTION
############################################
@torch.no_grad()
def extract_patch_embeddings(
    img_pil: Image.Image,
    model,
    preprocess,
    device,
    img_size: int,
):
    """
    img_pil: a PIL Image (RGB)
    model: DINOv3 model
    preprocess: torchvision transform pipeline
    device: 'cuda' or 'cpu'
    img_size: input resolution used by preprocess() (e.g., 240)
    returns:
        patch_embeds: Tensor [num_patches, D]
        H_p, W_p: spatial patch grid size
    """

    # 1. preprocess and add batch dim
    x = preprocess(img_pil).unsqueeze(0).to(device)   # [1, 3, img_size, img_size]

    # 2. get token embeddings
    tokens = model.get_intermediate_layers(x, n=1)[0]  # [1, T, D]
    tokens = tokens.squeeze(0)                         # [T, D]

    # 3. patch size = 16 for ViT-S/16
    patch_size = 16
    H_p = W_p = img_size // patch_size
    num_patches = H_p * W_p

    if tokens.shape[0] < num_patches:
        raise ValueError(
            f"Got only {tokens.shape[0]} tokens, need at least {num_patches}. "
            f"Check image size or model architecture."
        )

    # 4. last patch tokens = actual patch embeddings
    patch_tokens = tokens[-num_patches:, :]  # [num_patches, D]

    # 5. reshape to grid
    patch_tokens = patch_tokens.view(H_p, W_p, -1)     # [H_p, W_p, D]

    # 6. flatten
    patch_tokens_flat = patch_tokens.view(-1, patch_tokens.shape[-1])  # [H_p*W_p, D]

    return patch_tokens_flat, H_p, W_p

############################################
# MEMORY BANKS
############################################

@torch.no_grad()
def build_memory_bank(train_dir, model, preprocess, device, img_size: int):
    """
    Build a memory bank from all normal images in train_dir.
    returns: memory_bank [M, D] on DEVICE
    """
    all_patches = []

    img_paths = (
        glob.glob(os.path.join(train_dir, "*.png")) +
        glob.glob(os.path.join(train_dir, "*.jpg")) +
        glob.glob(os.path.join(train_dir, "*.jpeg"))
    )

    for p in img_paths:
        img = Image.open(p).convert("RGB")
        patch_embeds, _, _ = extract_patch_embeddings(
            img,
            model=model,
            preprocess=preprocess,
            device=device,
            img_size=img_size,
        )
        all_patches.append(patch_embeds)

    if not all_patches:
        raise ValueError(f"No training images found in {train_dir}")

    return torch.cat(all_patches, dim=0)

@torch.no_grad()
def build_memory_banks_for_categories(
    categories,
    model,
    preprocess,
    device,
    img_size: int,
    get_category_paths,
):
    """
    Build {category: memory_bank} for all categories.
    """
    memory_banks = {}

    for cat in categories:
        train_good_dir, _, _ = get_category_paths(cat)
        print(f"Building memory bank for '{cat}' from {train_good_dir}...")

        mb = build_memory_bank(
            train_good_dir,
            model=model,
            preprocess=preprocess,
            device=device,
            img_size=img_size,
        )

        print(f"  -> memory bank shape: {mb.shape}")
        memory_banks[cat] = mb

    return memory_banks

############################################
# SCORING A SINGLE IMAGE
############################################

@torch.no_grad()
def score_test_image(img_pil, memory_bank, model, preprocess, device, img_size, q=100):
    """
    img_pil: PIL image (test image, may be defective or not)
    memory_bank: [M, D] tensor of normal patch embeddings (on DEVICE)

    returns:
        patch_scores: [H_p, W_p] anomaly score per patch
        image_score:  float, overall anomaly score for the image (max of patches)
    """

    # 1. get patch embeddings for this test image
    test_patches, H_p, W_p = extract_patch_embeddings(img_pil, model, preprocess, device, img_size)   # [N_test, D]

    # 2. compute distances to memory bank (cosine distance)
    test_norm = torch.nn.functional.normalize(test_patches, dim=1)        # [N_test, D]
    bank_norm = torch.nn.functional.normalize(memory_bank, dim=1)         # [M, D]

    sim = test_norm @ bank_norm.T                                         # [N_test, M]
    best_sim, _ = sim.max(dim=1)                                          # [N_test]

    patch_anomaly = 1.0 - best_sim                                        # [N_test]

    # 3. reshape back to patch grid
    patch_scores = patch_anomaly.view(H_p, W_p)                           # [H_p, W_p]

    # image-level anomaly score (max patch anomaly)
    image_score = float(torch.quantile(patch_anomaly, q / 100.0).item())

    return patch_scores, image_score

############################################
# IMAGE-LEVEL PERFORMANCE METRICS
############################################

@torch.no_grad()
def evaluate_image_level_performance(
    category_thresholds,
    memory_banks,
    model,
    preprocess,
    device,
    img_size: int,
    get_category_paths,
):
    """
    Evaluate image-level performance for multiple categories using
    category-specific thresholds and memory banks.

    Returns
    -------
    results : dict
        {
          category: {
             "y_true": np.array [N],
             "y_score": np.array [N],
             "y_pred": np.array [N],
             "fpr": np.array,
             "tpr": np.array,
             "roc_thresholds": np.array,
             "auc": float,
             "acc": float,
             "confusion_matrix": np.ndarray shape (2, 2),
             "threshold": float,
          },
          ...
        }
    """
    results = {}

    # iterate over categories for which we have thresholds
    for cat, threshold in category_thresholds.items():
        # get test directory for this category
        _, test_dir, _ = get_category_paths(cat)

        if cat not in memory_banks:
            raise KeyError(f"Memory bank for category '{cat}' not found in memory_banks dict.")

        mb = memory_banks[cat]

        y_true = []   # 0 = normal, 1 = anomaly
        y_score = []  # continuous anomaly scores
        y_pred = []   # predicted label using threshold

        # subfolders in test_dir: usually 'good', 'defect_type1', ...
        subdirs = sorted(
            d for d in os.listdir(test_dir)
            if os.path.isdir(os.path.join(test_dir, d))
        )

        for sub in subdirs:
            subdir_path = os.path.join(test_dir, sub)
            label = 0 if sub == "good" else 1

            img_paths = (
                glob.glob(os.path.join(subdir_path, "*.png")) +
                glob.glob(os.path.join(subdir_path, "*.jpg")) +
                glob.glob(os.path.join(subdir_path, "*.jpeg"))
            )

            for p in img_paths:
                img = Image.open(p).convert("RGB")
                _, img_score = score_test_image(
                    img,
                    memory_bank=mb,
                    model=model,
                    preprocess=preprocess,
                    device=device,
                    img_size=img_size,
                )

                y_true.append(label)
                y_score.append(img_score)
                y_pred.append(1 if img_score >= threshold else 0)

        y_true = np.array(y_true, dtype=int)
        y_score = np.array(y_score, dtype=np.float32)
        y_pred = np.array(y_pred, dtype=int)

        # metrics
        auc = roc_auc_score(y_true, y_score)
        f1 = f1_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        # data for ROC curve plot
        fpr, tpr, roc_thr = roc_curve(y_true, y_score)

        results[cat] = {
            "y_true": y_true,
            "y_score": y_score,
            "y_pred": y_pred,
            "fpr": fpr,
            "tpr": tpr,
            "roc_thresholds": roc_thr,
            "auc": float(auc),
            "f1": float(f1),
            "acc": float(acc),
            "confusion_matrix": cm,
            "threshold": float(threshold),
        }

    return results

def plot_image_level_rocs_with_confusion_matrices(results_dict, max_cols=5, width=2.5, height=3.5, plot=True):
    """
    Plot ROC curves and confusion matrices for each category
    using ROC plots and CM plots below each ROC.
    """

    categories = sorted(results_dict.keys())
    n_cats = len(categories)

    if n_cats == 0:
        print("No categories found.")
        return

    max_cols = max_cols                     # up to max_cols categories per row
    n_cols = min(max_cols, n_cats)
    n_cat_rows = int(np.ceil(n_cats / n_cols))  # number of category rows

    # --- keep per-category cell size constant ---
    width_per_col = width      # inches per column
    height_per_cat_row = height # inches per (ROC+CM) block vertically

    fig_width = width_per_col * n_cols
    fig_height = height_per_cat_row * n_cat_rows
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Each category row uses 2 GridSpec rows: [ROC, CM]
    gs = GridSpec(
        n_cat_rows * 2,            # double rows (ROC + CM)
        n_cols,
        height_ratios=[3, 1] * n_cat_rows,  # ROC 3x taller than CM
    )

    for idx, cat in enumerate(categories):
        row = idx // n_cols       # which category row
        col = idx % n_cols        # which column

        res = results_dict[cat]

        fpr = res["fpr"]
        tpr = res["tpr"]
        auc = res["auc"]
        acc = res["acc"]
        cm = res["confusion_matrix"]
        thr = res["threshold"]
        y_true = res["y_true"]
        y_score = res["y_score"]

        # ---- compute operating point (TPR, FPR) at threshold ----
        y_pred_thr = (y_score >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thr).ravel()

        tpr_thr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr_thr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # ================================ ROC AXIS
        ax_roc = fig.add_subplot(gs[row * 2, col])

        ax_roc.plot(fpr, tpr, linewidth=2)
        ax_roc.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color="gray")

        ax_roc.set_title(f"Category: {KEY_TO_DISPLAY[cat]}", fontsize=11)
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.grid(True, linestyle=":", linewidth=0.9)

        ax_roc.text(
            0.98, 0.02,
            f"AUROC={auc:.3f}",
            transform=ax_roc.transAxes,
            fontsize=9,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round", alpha=0.2),
        )

        # --- red cross marker at threshold ---
        ax_roc.scatter(
            fpr_thr,
            tpr_thr,
            marker='x',
            s=30,
            color='red',
            linewidth=1.5,
            label="Threshold point",
            zorder=5,
        )

        # ================================ CM AXIS
        ax_cm = fig.add_subplot(gs[row * 2 + 1, col])
        ax_cm.imshow(cm, cmap="coolwarm")

        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(["0", "1"], fontsize=8)
        ax_cm.set_yticklabels(["0", "1"], fontsize=8)
        ax_cm.set_xlabel("Predicted", fontsize=9)
        ax_cm.set_ylabel("True", fontsize=9)

        labels = np.array([["TN", "FP"], ["FN", "TP"]])
        for i in range(2):
            for j in range(2):
                ax_cm.text(
                    j, i,
                    f"{labels[i, j]}\n{cm[i, j]}",
                    ha="center", va="center",
                    fontsize=8,
                    fontweight="bold",
                )

        ax_cm.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("figure_1.svg", format="svg")
    if plot:
    	plt.show()

############################################
# PIXEL-LEVEL PERFORMANCE METRICS
############################################

@torch.no_grad()
def evaluate_pixel_level_performance(
    pixel_thresholds,
    memory_banks,
    model,
    preprocess,
    device,
    img_size: int,
    get_category_paths,
):
    """
    Evaluate pixel-level performance using MULTIPLE thresholds per category.

    pixel_thresholds must be:
        {category: [t1, t2, t3]}

    For each category, returns:
        - y_true, y_score  (per pixel, shared by all thresholds)
        - y_pro: dict { "t1": per-region PRO values, ... }
        - fpr, tpr, roc_thresholds, auc  (from full score curve, pixel-level)
        - pro_fpr_curve, pro_curve, pro_auc (full PRO vs FPR curve)
        - per_threshold: dict with acc, y_pred, cm, fpr, tpr, pro per threshold.
    """
    results = {}

    for cat, thr_list in pixel_thresholds.items():
        if cat not in memory_banks:
            raise KeyError(f"Memory bank missing for '{cat}'")

        # thresholds: [t1, t2, t3]
        thr_keys = [f"t{i+1}" for i in range(len(thr_list))]
        mb = memory_banks[cat]

        _, test_dir, gt_dir = get_category_paths(cat)

        y_true_pix = []
        y_score_pix = []

        # for PRO at the three specific thresholds
        pro_values = {k: [] for k in thr_keys}

        # for PRO curve: store anomaly maps + labels per anomaly image
        anomaly_maps = []
        anomaly_labels = []

        subdirs = sorted(
            d for d in os.listdir(test_dir)
            if os.path.isdir(os.path.join(test_dir, d))
        )

        for sub in subdirs:
            subdir_path = os.path.join(test_dir, sub)
            img_paths = (
                glob.glob(os.path.join(subdir_path, "*.png")) +
                glob.glob(os.path.join(subdir_path, "*.jpg")) +
                glob.glob(os.path.join(subdir_path, "*.jpeg"))
            )

            is_good = (sub == "good")

            for p in img_paths:
                # -------- anomaly map (scores) --------
                img = Image.open(p).convert("RGB")
                patch_scores, _ = score_test_image(
                    img,
                    memory_bank=mb,
                    model=model,
                    preprocess=preprocess,
                    device=device,
                    img_size=img_size,
                )  # [H_p, W_p]

                patch_scores_np = patch_scores.detach().cpu().numpy()

                upsampled = cv2.resize(
                    patch_scores_np,
                    (img_size, img_size),
                    interpolation=cv2.INTER_CUBIC,
                )  # [H, W]

                scores_flat = upsampled.reshape(-1)

                # -------- ground-truth mask --------
                if is_good:
                    gt_mask = np.zeros((img_size, img_size), dtype=np.uint8)
                else:
                    defect_name = sub
                    img_name = os.path.splitext(os.path.basename(p))[0]
                    mask_path = os.path.join(
                        gt_dir,
                        defect_name,
                        img_name + "_mask.png",
                    )

                    if not os.path.exists(mask_path):
                        raise FileNotFoundError(
                            f"Mask not found for '{p}', expected '{mask_path}'"
                        )

                    mask = Image.open(mask_path).convert("L")
                    mask_np = np.array(mask)

                    gt_mask = cv2.resize(
                        mask_np,
                        (img_size, img_size),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    gt_mask = (gt_mask > 0).astype(np.uint8)

                gt_flat = gt_mask.reshape(-1).astype(int)

                # store pixels for ROC / AUROC (global)
                y_true_pix.append(gt_flat)
                y_score_pix.append(scores_flat.astype(np.float32))

                # -------- PRO at the three specific thresholds --------
                if not is_good:
                    # connected components on gt_mask
                    num_labels, labels = cv2.connectedComponents(gt_mask)

                    # keep maps for PRO curve
                    anomaly_maps.append(upsampled)
                    anomaly_labels.append(labels)

                    for label_id in range(1, num_labels):
                        region = (labels == label_id)
                        region_size = region.sum()
                        if region_size == 0:
                            continue

                        for key, thr in zip(thr_keys, thr_list):
                            pred_mask = (upsampled >= thr)  # bool 2D
                            inter = np.logical_and(pred_mask, region).sum()
                            pro_region = inter / float(region_size)
                            pro_values[key].append(pro_region)

        # -------- concatenate all pixels across images --------
        y_true = np.concatenate(y_true_pix).astype(int)
        y_score = np.concatenate(y_score_pix).astype(np.float32)

        # -------- ROC & AUROC (pixel-level, threshold-free) --------
        auc = roc_auc_score(y_true, y_score)
        fpr_curve, tpr_curve, roc_thr = roc_curve(y_true, y_score)

        # -------- per-threshold metrics (acc, cm, fpr_op, tpr_op, pro) --------
        metrics_per_threshold = {}
        y_pro = {}

        for key, thr in zip(thr_keys, thr_list):
            y_pred = (y_score >= thr).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            acc = accuracy_score(y_true, y_pred)

            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
                tpr_op = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                fpr_op = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            else:
                tpr_op = 0.0
                fpr_op = 0.0

            pro_arr = (
                np.array(pro_values[key], dtype=np.float32)
                if len(pro_values[key]) > 0
                else np.zeros(0, dtype=np.float32)
            )
            pro_mean = float(pro_arr.mean()) if pro_arr.size > 0 else float("nan")

            y_pro[key] = pro_arr

            metrics_per_threshold[key] = {
                "threshold": float(thr),
                "f1": float(f1),
                "acc": float(acc),
                "y_pred": y_pred,
                "confusion_matrix": cm,
                "fpr": float(fpr_op),
                "tpr": float(tpr_op),
                "pro": pro_mean,
            }

        # -------- PRO curve (threshold sweep) --------
        pro_fpr_curve = []
        pro_curve = []

        if len(anomaly_maps) > 0:
            # choose a grid of thresholds based on score distribution
            num_points = 50
            qs = np.linspace(0.0, 1.0, num_points + 2)[1:-1]  # avoid extremes
            thr_grid = np.quantile(y_score, qs)

            for thr_c in thr_grid:
                # pixel-level FPR for this threshold
                y_pred_c = (y_score >= thr_c).astype(int)
                cm_c = confusion_matrix(y_true, y_pred_c)
                if cm_c.size == 4:
                    tn_c, fp_c, fn_c, tp_c = cm_c.ravel()
                    fpr_c = fp_c / (fp_c + tn_c) if (fp_c + tn_c) > 0 else 0.0
                else:
                    fpr_c = 0.0

                # region-level PRO for this threshold
                pro_regions = []
                for up_map, labels in zip(anomaly_maps, anomaly_labels):
                    # all GT labels >= 1
                    region_ids = np.unique(labels)
                    region_ids = region_ids[region_ids != 0]

                    pred_mask_c = (up_map >= thr_c)

                    for rid in region_ids:
                        region = (labels == rid)
                        region_size = region.sum()
                        if region_size == 0:
                            continue
                        inter = np.logical_and(pred_mask_c, region).sum()
                        pro_val = inter / float(region_size)
                        pro_regions.append(pro_val)

                if len(pro_regions) > 0:
                    pro_mean_c = float(np.mean(pro_regions))
                else:
                    pro_mean_c = float("nan")

                pro_fpr_curve.append(fpr_c)
                pro_curve.append(pro_mean_c)

            pro_fpr_curve = np.array(pro_fpr_curve, dtype=np.float32)
            pro_curve = np.array(pro_curve, dtype=np.float32)

            # sort by FPR for nice curve & AUC
            order = np.argsort(pro_fpr_curve)
            pro_fpr_curve = pro_fpr_curve[order]
            pro_curve = pro_curve[order]

            # simple trapezoidal PRO-AUC over full FPR range
            # (you can restrict to FPR <= 0.3 if you want MVTec-style)
            # ignore NaNs in pro_curve
            valid = np.isfinite(pro_curve)
            if np.any(valid):
                pro_auc = float(np.trapezoid(pro_curve[valid], pro_fpr_curve[valid]))
            else:
                pro_auc = float("nan")
        else:
            pro_fpr_curve = np.zeros(0, dtype=np.float32)
            pro_curve = np.zeros(0, dtype=np.float32)
            pro_auc = float("nan")

        # -------- store results for this category --------
        results[cat] = {
            "y_true": y_true,
            "y_score": y_score,
            "y_pro": y_pro,                     # per-threshold region PRO values
            "fpr": fpr_curve,                   # pixel ROC
            "tpr": tpr_curve,
            "roc_thresholds": roc_thr,
            "auc": float(auc),
            "pro_fpr_curve": pro_fpr_curve,     # PRO vs FPR curve
            "pro_curve": pro_curve,
            "pro_auc": pro_auc,
            "per_threshold": metrics_per_threshold,
        }

    return results

def plot_pixel_level_roc_and_pro(results_dict, max_cols=5, width=2.5, height=4.5, plot=True):
    """
    For each category:
      - Top: pixel-level ROC curve + three threshold operating points (t1, t2, t3)
      - Bottom: full PRO curve (threshold sweep) + the three PRO operating points.

    Expects the results from evaluate_pixel_level_performance().
    """

    categories = sorted(results_dict.keys())
    n_cats = len(categories)

    if n_cats == 0:
        print("No categories to plot.")
        return

    max_cols = max_cols
    n_cols = min(max_cols, n_cats)
    n_cat_rows = int(np.ceil(n_cats / n_cols))

    # fixed per-category block size
    width_per_col = width
    height_per_cat_row = height

    fig_width = width_per_col * n_cols
    fig_height = height_per_cat_row * n_cat_rows
    fig = plt.figure(figsize=(fig_width, fig_height))

    gs = GridSpec(
        n_cat_rows * 2,        # 2 rows per category: ROC + PRO
        n_cols,
        height_ratios=[1, 1] * n_cat_rows,
    )

    thr_keys = ["t1", "t2", "t3"]
    markers = {"t1": "^", "t2": "x", "t3": "s"}  # square, cross, triangle
    thr_colors = {"t1": "orange", "t2": "red", "t3": "green"}

    for idx, cat in enumerate(categories):
        row = idx // n_cols
        col = idx % n_cols
        res = results_dict[cat]

        fpr_curve = res["fpr"]
        tpr_curve = res["tpr"]
        auc = res["auc"]
        pro_auc = res["pro_auc"]

        pro_fpr_curve = res["pro_fpr_curve"]
        pro_curve = res["pro_curve"]
        per_thr = res["per_threshold"]

        # ==================== ROC AXIS ============================
        ax_roc = fig.add_subplot(gs[row * 2, col])

        ax_roc.plot(fpr_curve, tpr_curve, linewidth=2)
        ax_roc.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color="gray")

        ax_roc.set_title(f"Category: {KEY_TO_DISPLAY[cat]}", fontsize=11)
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.grid(True, linestyle=":", linewidth=0.7)

        ax_roc.text(
            0.98, 0.02,
            f"AUROC={auc:.3f}",
            transform=ax_roc.transAxes,
            fontsize=9,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round", alpha=0.2),
        )

        # ---- Add markers for t1,t2,t3 on ROC ----
        for key in thr_keys:
            if key not in per_thr:
                continue
            pt = per_thr[key]
            fpr_op = pt["fpr"]
            tpr_op = pt["tpr"]

            ax_roc.scatter(
                fpr_op,
                tpr_op,
                marker=markers[key],
                s=40,
                linewidth=0.9,              # border thickness
                facecolor=thr_colors[key],  # marker fill color
                edgecolors="black",         # border
                zorder=5,
            )

        # ==================== PRO AXIS ============================
        ax_pro = fig.add_subplot(gs[row * 2 + 1, col])

        # ---- Full PRO curve (threshold sweep) ----
        if pro_fpr_curve.size > 0:
            ax_pro.plot(
                pro_fpr_curve, pro_curve,
                "-", linewidth=1.5, color="black"
            )

        ax_pro.text(
            0.98, 0.02,
            f"AUPRO={pro_auc:.3f}",
            transform=ax_pro.transAxes,
            fontsize=9,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round", alpha=0.2),
        )

        # ---- Add t1,t2,t3 operating points ----
        for key in thr_keys:
            if key not in per_thr:
                continue
            fpr_op = per_thr[key]["fpr"]
            pro_op = per_thr[key]["pro"]

            if np.isfinite(fpr_op) and np.isfinite(pro_op):
                ax_pro.scatter(
                    fpr_op,
                    pro_op,
                    marker=markers[key],
                    s=40,
                    linewidth=0.9,              # border thickness
                    facecolor=thr_colors[key],  # marker fill color
                    edgecolors="black",         # border
                    zorder=5,
                )

        ax_pro.set_xlabel("False Positive Rate")
        ax_pro.set_ylabel("Per-Region Overlap")
        ax_pro.set_ylim(0, 1.05)
        ax_pro.grid(True, linestyle=":", linewidth=0.7)

    plt.tight_layout()
    plt.savefig("figure_2.svg", format="svg")
    if plot:
    	plt.show()

############################################
# PLOTTING ANOMALY MAPS
############################################

def plot_pixel_example_for_image(
    category: str,
    defect_type: str,
    image_index: int,
    memory_banks: dict,
    pixel_level_banks: dict,
    threshold_key: str,
    model,
    preprocess,
    device,
    img_size: int,
    get_category_paths,
    axes=None,
    show_titles=True,
    width=6,
    height=2,
    plot=True
):
    """
    Visualize pixel-level anomaly detection for a single test image.

    If axes is None -> create a new 1x4 figure (standalone).
    If axes is a list/array of 4 axes -> draw into those (for grids).

    Anomaly map panel now shows the patch-level (image-level) anomaly grid
    instead of the fully upsampled pixel-level map.
    """

    # pick threshold from list based on key
    threshold = pixel_level_banks[category]["per_threshold"][threshold_key]["threshold"]

    mb = memory_banks[category]
    _, test_dir, gt_dir = get_category_paths(category)

    # ------------ locate image ------------
    defect_dir = os.path.join(test_dir, defect_type)
    img_paths = (
        glob.glob(os.path.join(defect_dir, "*.png")) +
        glob.glob(os.path.join(defect_dir, "*.jpg")) +
        glob.glob(os.path.join(defect_dir, "*.jpeg"))
    )
    img_paths = sorted(img_paths)
    if len(img_paths) == 0:
        raise FileNotFoundError(f"No images found in {defect_dir}.")
    if not (0 <= image_index < len(img_paths)):
        raise IndexError(f"image_index {image_index} out of range (0..{len(img_paths)-1})")

    img_path = img_paths[image_index]

    # ------------ load image & compute anomaly scores ------------
    img = Image.open(img_path).convert("RGB")

    # patch_scores: [H_p, W_p] patch-level anomalies
    patch_scores, _ = score_test_image(
        img,
        memory_bank=mb,
        model=model,
        preprocess=preprocess,
        device=device,
        img_size=img_size,
    )

    patch_scores_np = patch_scores.detach().cpu().numpy()  # [H_p, W_p]

    # ---------- anomaly map for visualization (patch-level) ----------
    # patch_scores: [H_p, W_p] -> [1,1,H_p,W_p] for interpolate
    patch_map = patch_scores.unsqueeze(0).unsqueeze(0)  # [1, 1, H_p, W_p]

    # upsample to image resolution
    anomaly_map_t = F.interpolate(
        patch_map,
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    )[0, 0]  # [H, W] tensor

    anomaly_map = anomaly_map_t.detach().cpu().numpy()  # numpy [H, W]

    # normalize anomaly map for nicer visualization (0–1)
    amap_min, amap_max = anomaly_map.min(), anomaly_map.max()
    if amap_max > amap_min:
        anomaly_map_vis = (anomaly_map - amap_min) / (amap_max - amap_min + 1e-8)
    else:
        anomaly_map_vis = np.zeros_like(anomaly_map)

    # colorize heatmap
    heat_uint8 = np.uint8(255 * anomaly_map_vis)
    colored_heat = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    colored_heat = cv2.cvtColor(colored_heat, cv2.COLOR_BGR2RGB)

    # thresholded prediction (same as before, but now from torch-upsampled map)
    pred_mask = (anomaly_map >= threshold).astype(np.uint8)

    # ---------- anomaly map for prediction (pixel-level) ----------
    # keep your previous behavior for segmentation: upsample to img_size
    anomaly_map_pred = cv2.resize(
        patch_scores_np,
        (img_size, img_size),
        interpolation=cv2.INTER_LANCZOS4,
    )
    pred_mask = (anomaly_map_pred >= threshold).astype(np.uint8)

    # ------------ load ground-truth mask ------------
    if defect_type == "good":
        gt_mask = np.zeros((img_size, img_size), dtype=np.uint8)
    else:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(gt_dir, defect_type, img_name + "_mask.png")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found for '{img_path}', expected '{mask_path}'.")
        mask_img = Image.open(mask_path).convert("L")
        mask_np = np.array(mask_img)

        gt_mask = cv2.resize(mask_np, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        gt_mask = (gt_mask > 0).astype(np.uint8)

    # resize image to match masks
    img_resized = img.resize((img_size, img_size), Image.BICUBIC)

    # overlays
    overlay_cmap = mcolors.ListedColormap(["#BB0201"])
    gt_masked = np.ma.masked_where(gt_mask == 0, gt_mask)
    pred_masked = np.ma.masked_where(pred_mask == 0, pred_mask)

    # ------------ handle standalone vs embedded ------------
    standalone = axes is None
    if standalone:
        fig, axes = plt.subplots(1, 4, figsize=(width, height))
    ax0, ax1, ax2, ax3 = axes

    # 1) original image
    ax0.imshow(img_resized)
    if show_titles:
        ax0.set_title("Input image", fontsize=10)
    ax0.axis("off")

    # 2) GT overlay
    ax1.imshow(img_resized)
    ax1.imshow(gt_masked, cmap=overlay_cmap, alpha=1)
    if show_titles:
        ax1.set_title("GT overlay", fontsize=10)
    ax1.axis("off")

    # 3) anomaly heatmap (patch-level)
    ax2.imshow(img_resized, alpha=1)  # base image
    ax2.imshow(
        anomaly_map_vis,        # normalized heatmap [0-1]
        cmap="jet",
        alpha=0.5,             # transparency
        vmin=0, vmax=1
    )
    if show_titles:
        ax2.set_title("Anomaly map", fontsize=10)
    ax2.axis("off")

    # 4) prediction overlay (pixel-level)
    ax3.imshow(img_resized)
    ax3.imshow(pred_masked, cmap=overlay_cmap, alpha=1)
    if show_titles:
        ax3.set_title("Pred. segmentation", fontsize=10)
    ax3.axis("off")

    if standalone:
        fig.suptitle(
            f"Category: {category} | Defect: {defect_type} | idx: {image_index}",
            fontsize=11,
        )
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.05)
        plt.savefig("figure_3.svg", format="svg")
        if plot:
        	plt.show()

def plot_multiple_pixel_rows(
    rows,
    memory_banks,
    pixel_level_banks,
    model,
    preprocess,
    device,
    img_size,
    get_category_paths,
    width=6,
    height=2,
    plot=True
):
    """
    Plot several pixel-level examples as multiple rows in one figure.

    rows: list of dicts, each like:
        {
          "category": "metal_nut",
          "defect_type": "scratch",
          "image_index": 1,
          "threshold_key": "t1",
        }
    """

    n_rows = len(rows)
    if n_rows == 0:
        print("No rows to plot.")
        return

    fig, axes = plt.subplots(
        n_rows, 4,
        figsize=(width, height * n_rows),
    )

    # ensure axes is 2D
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    # column titles (only once, on the first row)
    col_titles = ["Input image", "GT overlay", "Anomaly map", "Prediction"]

    for i, row in enumerate(rows):
        row_axes = axes[i]

        # draw panels without titles (we'll control titles ourselves)
        plot_pixel_example_for_image(
            category=row["category"],
            defect_type=row["defect_type"],
            image_index=row["image_index"],
            memory_banks=memory_banks,
            pixel_level_banks=pixel_level_banks,
            threshold_key=row["threshold_key"],
            model=model,
            preprocess=preprocess,
            device=device,
            img_size=img_size,
            get_category_paths=get_category_paths,
            axes=row_axes,
            show_titles=False,
        )

        # add column titles only for the first row
        if i == 0:
            for j, ax in enumerate(row_axes):
                ax.set_title(col_titles[j], fontsize=11)

        # add vertical category label on the left of the first column
        ax_left = row_axes[0]
        ax_left.text(
            -0.08, 0.5,
            KEY_TO_DISPLAY[row["category"]],
            rotation=90,
            transform=ax_left.transAxes,
            fontsize=11,
            va="center",
            ha="center",
        )

    #plt.tight_layout()
    fig.subplots_adjust(
        left=0.03,
        right=0.97,
        top=0.92,
        bottom=0.05,
        wspace=0.03,   # horizontal gap between columns
        hspace=-0.3,   # vertical gap between rows
    )

    plt.savefig("figure_4.svg", format="svg")
    if plot:
    	plt.show()


############################################
# RESULTS TABLE
############################################

TEXTURE_CATS = ["Carpet", "Grid", "Leather", "Tile", "Wood"]
OBJECT_CATS = [
    "Bottle", "Cable", "Capsule", "Hazelnut", "Metal nut",
    "Pill", "Screw", "Toothbrush", "Transistor", "Zipper",
]

DISPLAY_TO_KEY = {
    "Carpet": "carpet",
    "Grid": "grid",
    "Leather": "leather",
    "Tile": "tile",
    "Wood": "wood",
    "Bottle": "bottle",
    "Cable": "cable",
    "Capsule": "capsule",
    "Hazelnut": "hazelnut",
    "Metal nut": "metal_nut",
    "Pill": "pill",
    "Screw": "screw",
    "Toothbrush": "toothbrush",
    "Transistor": "transistor",
    "Zipper": "zipper",
}


def safe(x):
    return x if x is not None else "N/A"


def best_pixel_f1(entry):
    """Return max f1 among t1,t2,t3 or N/A."""
    if entry is None or "per_threshold" not in entry:
        return "N/A"
    f1_vals = []
    for t in ["t1", "t2", "t3"]:
        if t in entry["per_threshold"]:
            f1_vals.append(entry["per_threshold"][t].get("f1"))
    f1_vals = [v for v in f1_vals if v is not None]
    return max(f1_vals) if f1_vals else "N/A"


def build_simple_table(image_results, pixel_results):
    rows = []

    all_categories = TEXTURE_CATS + OBJECT_CATS

    for disp in all_categories:
        key = DISPLAY_TO_KEY[disp]

        img = image_results.get(key)
        pix = pixel_results.get(key)

        rows.append({
            "Category": disp,
            "Image_AUROC": safe(img.get("auc") if img else None),
            "Image_F1": safe(img.get("f1") if img else None),

            "Pixel_AUROC": safe(pix.get("auc") if pix else None),
            "Pixel_AUPRO": safe(pix.get("pro_auc") if pix else None),
            "Pixel_F1": best_pixel_f1(pix),
        })

    # Convert to DataFrame
    df = pd.DataFrame(rows)

    # Add "Total" row (mean of numeric values only)
    numeric_means = {}
    for col in ["Image_AUROC", "Image_F1", "Pixel_AUROC", "Pixel_AUPRO", "Pixel_F1"]:
        vals = pd.to_numeric(df[col], errors='coerce')
        mean_val = vals.mean()
        numeric_means[col] = round(mean_val, 3) if not np.isnan(mean_val) else "N/A"

    total_row = {"Category": "Total"}
    total_row.update(numeric_means)
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

    return df
