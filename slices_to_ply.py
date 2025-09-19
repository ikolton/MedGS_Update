import os
import numpy as np
import cv2
import re
import glob
import nibabel as nib
import open3d as o3d
from skimage import measure
from scipy.ndimage import gaussian_filter
import argparse


def extract_numbers(filename):
    match = re.match(r'(\d+)_([\d]+)', filename)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return (0, 0)


def main(input_folder, output_folder, thresh=150):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        render_folder = os.path.join(input_folder, filename, "seg/render")
        nifti_files = glob.glob(os.path.join(input_folder, filename, "*.nii*"))

        if not os.path.isdir(render_folder):
            continue

        if nifti_files:
            nifti_path = nifti_files[0]
            print(f"Using NIfTI: {nifti_path}")
            img = nib.load(nifti_path)
            spacing = img.header.get_zooms()
        else:
            nifti_path = None
            spacing = (1.0, 1.0, 1.0)

        files = sorted(
            [f for f in os.listdir(render_folder) if f.endswith(".png") and not f.startswith('_')],
            key=extract_numbers
        )

        volume = []
        for frame in files:
            frame_path = os.path.join(render_folder, frame)
            img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            volume.append(img)

        if not volume:
            print(f"No images found in {render_folder}")
            continue

        volume = np.array(volume, dtype=np.float32)
        print("Volume shape (slices, height, width):", volume.shape)

        volume = gaussian_filter(volume, sigma=(4, 1, 1))

        verts, faces, normals, values = measure.marching_cubes(
            volume, level=thresh,
            spacing=[spacing[2] / 8, spacing[0], spacing[1]]
        )

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()

        out_path = os.path.join(output_folder, f"{filename}.ply")
        o3d.io.write_triangle_mesh(out_path, mesh, print_progress=True)
        print(f"Saved mesh to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build 3D meshes from 2D slices")
    parser.add_argument("-i", "--input", required=True, help="Input folder")
    parser.add_argument("-o", "--output", required=True, help="Output folder")
    parser.add_argument("--thresh", type=float, default=150, help="Marching cubes threshold")
    parser.add_argument("--inter", type=int, default=8, help="Interpolation factor")
    args = parser.parse_args()

    main(args.input, args.output, args.thresh)
