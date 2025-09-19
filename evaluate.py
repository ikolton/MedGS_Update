import os
import trimesh
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import nibabel as nib
import matplotlib.pyplot as plt
from trimesh.voxel import creation
from utils import prepare_mesh, convert_normalized_mesh_to_mm, scale_mesh_to_match
from scipy.ndimage import binary_fill_holes, label


def get_combined_bbox(mesh1, mesh2, padding=1.0):
    min_corner = np.minimum(mesh1.vertices.min(axis=0), mesh2.vertices.min(axis=0)) - padding
    max_corner = np.maximum(mesh1.vertices.max(axis=0), mesh2.vertices.max(axis=0)) + padding
    return min_corner, max_corner

def visualize_3d_masks_side_by_side(gt, pred, every=1, max_slices=64):
    assert gt.shape == pred.shape, "GT and prediction must have the same shape"

    slices = list(range(0, gt.shape[0], every))[:max_slices]
    n = len(slices)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=2 * nrows, ncols=ncols, figsize=(ncols * 2, nrows * 4))

    if nrows == 1:
        axes = axes.reshape(2, -1)

    for idx, slice_idx in enumerate(slices):
        row_gt = (idx // ncols) * 2
        row_pred = row_gt + 1
        col = idx % ncols

        ax_gt = axes[row_gt, col]
        ax_pred = axes[row_pred, col]

        ax_gt.imshow(gt[slice_idx], cmap='Greens')
        ax_gt.set_title(f"GT - Slice {slice_idx}")
        ax_gt.axis('off')

        ax_pred.imshow(pred[slice_idx], cmap='Reds')
        ax_pred.set_title(f"Pred - Slice {slice_idx}")
        ax_pred.axis('off')

    for ax in axes.flat[n * 2:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def voxelize_mesh(mesh, grid_shape=(128, 128, 128), bbox=None):

    if bbox is None:
        min_corner = mesh.vertices.min(axis=0)
        max_corner = mesh.vertices.max(axis=0)
    else:
        min_corner, max_corner = bbox

    scale = max_corner - min_corner
    verts_norm = (mesh.vertices - min_corner) / scale

    verts_voxel = verts_norm * (np.array(grid_shape) - 1)
    mask = np.zeros(grid_shape, dtype=np.uint8)

    for z, y, x in np.round(verts_voxel).astype(int):
        mask[z, y, x] = 1

    # for i in range(mask.shape[0]):
    #     mask[i] = binary_fill_holes(mask[i])
    mask = binary_fill_holes(mask).astype(np.uint8)

    return mask


def transform_mesh_to_voxel_space(mesh, affine):
    inv_affine = np.linalg.inv(affine)
    verts_hom = np.c_[mesh.vertices, np.ones(len(mesh.vertices))]
    verts_voxel = (inv_affine @ verts_hom.T).T[:, :3]
    return trimesh.Trimesh(vertices=verts_voxel, faces=mesh.faces, process=False)

def compute_dice_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    volume_sum = pred_mask.sum() + gt_mask.sum()
    dice = 2 * intersection / volume_sum if volume_sum > 0 else 1.0
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection / union if union > 0 else 1.0
    return round(dice, 4), round(iou, 4)

def chamfer_distance(points_src, points_tgt, squared=False):
    tree_src = cKDTree(points_src)
    tree_tgt = cKDTree(points_tgt)
    dist_src_to_tgt, _ = tree_tgt.query(points_src)
    dist_tgt_to_src, _ = tree_src.query(points_tgt)
    if squared:
        dist_src_to_tgt = dist_src_to_tgt ** 2
        dist_tgt_to_src = dist_tgt_to_src ** 2
    return 0.5 * (np.mean(dist_src_to_tgt) + np.mean(dist_tgt_to_src))


def average_surface_distance(P, Q):
    return 0.5 * (np.mean(cKDTree(Q).query(P)[0]) + np.mean(cKDTree(P).query(Q)[0]))

def hausdorff_metrics(P, Q):
    dist_PQ = cKDTree(Q).query(P)[0]
    dist_QP = cKDTree(P).query(Q)[0]
    return max(np.max(dist_PQ), np.max(dist_QP)), np.percentile(np.concatenate([dist_PQ, dist_QP]), 95)

def evaluate_pair(pred_path, gt_path, nifti_path, mirror = False, edit = False):
    pred_mesh = trimesh.load(pred_path)
    gt_mesh = trimesh.load(gt_path)

    if mirror:
        # pred_mesh.vertices = pred_mesh.vertices[:, [0, 2, 1]]
        pred_mesh.vertices[:, 0] *= -1


    nii = nib.load(nifti_path)
    affine = nii.affine

    if edit:
        resolution = 128
        voxel_origin = np.array([-1, -1, -1])
        voxel_size = 2.0 / (resolution - 1)
        verts_voxel = (pred_mesh.vertices - voxel_origin) / voxel_size
        verts_hom = np.c_[verts_voxel, np.ones(len(verts_voxel))]
        verts_mm = (affine @ verts_hom.T).T[:, :3]
        pred_mesh = trimesh.Trimesh(vertices=verts_mm, faces=pred_mesh.faces, process=False)

    pred_mesh = scale_mesh_to_match(pred_mesh, gt_mesh)

    # pred_center = pred_mesh.vertices.mean(axis=0)
    # gt_center = gt_mesh.vertices.mean(axis=0)
    # pred_mesh.vertices += (gt_center - pred_center)


    # pred_mesh_aligned = prepare_mesh(pred_mesh, gt_mesh, 90,180,90, 1) ## 
    pred_mesh_aligned = prepare_mesh(pred_mesh, gt_mesh, 270,0,270, 100) ## 

    # pred_mesh_aligned = prepare_mesh(pred_mesh, gt_mesh, 0, 0, 0, 1000) ## 
    # pred_mesh_aligned = prepare_mesh(pred_mesh, gt_mesh, 0, 0, 180, 50) ## 
    # pred_mesh_aligned = prepare_mesh(pred_mesh, gt_mesh, 0, 0, 180, 50) #poisson

    pred_pts = trimesh.sample.sample_surface(pred_mesh_aligned, 50000)[0]
    gt_pts = trimesh.sample.sample_surface(gt_mesh, 50000)[0]
    cd = chamfer_distance(pred_pts, gt_pts)
    asd = average_surface_distance(pred_pts, gt_pts)
    hd, hd95 = hausdorff_metrics(pred_pts, gt_pts)

    pred_mesh_aligned.visual.face_colors = [255, 0, 0, 70]
    gt_mesh.visual.face_colors = [0, 255, 0, 100]

    scene = trimesh.Scene([pred_mesh_aligned, gt_mesh])
    scene.show(flags={'wireframe': True})

    # bbox = get_combined_bbox(pred_mesh_aligned, gt_mesh, padding=5.0)
    # pred_mask = voxelize_mesh(pred_mesh_aligned, shape, bbox)
    # gt_mask = voxelize_mesh(gt_mesh, shape, bbox)
    # dsc, iou = compute_dice_iou(pred_mask, gt_mask)
    # visualize_3d_masks_side_by_side(gt_mask, pred_mask, every=2)

    return {
        "ASD(mm)": round(asd, 4),
        "CD(mm)": round(cd, 4),
        "HD(mm)": round(hd, 4),
        "HD95(mm)": round(hd95, 4),
        # "DSC": dsc,
        # "IoU": iou
    }


def evaluate_all(pred_dir, gt_dir, nifti_dir, output_csv="results.csv"):
    all_results = []
    for file in sorted(os.listdir(pred_dir)):
        if not file.endswith("70_pred.ply"):
            continue
        name = file.replace("_pred.ply", "")
        pred_path = os.path.join(pred_dir, file)
        gt_path = os.path.join(gt_dir, fr"{name}_gt.ply")
        nifti_path = os.path.join(nifti_dir, fr"{name}.nii.gz")
        if not os.path.exists(gt_path):
            print(f"Missing GT for {name}, skipping.")
            continue
        print(f"Evaluating {name}...")
        metrics = evaluate_pair(pred_path, gt_path, nifti_path, True, False)
        metrics["Case"] = name
        all_results.append(metrics)

    df = pd.DataFrame(all_results)
    df = df[["Case", "ASD(mm)", "CD(mm)", "HD(mm)", "HD95(mm)"]]#, "DSC", "IoU"]]
    avg = df.drop(columns="Case").mean().to_dict()
    avg["Case"] = "Average"
    df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)
    df.to_csv(output_csv, index=False)
    print("Saved results to", output_csv)


if __name__ == '__main__':
    evaluate_all('prostate/prostate_funsr_predictions', 'prostate/gt_slicer_output', 'prostate/prostate_dset/val/us_labels')
