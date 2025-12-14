import torch
import numpy as np
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from models import gaussianModelRender, gaussianModel  # NOTE: import both
from PIL import Image


def modify_func(
    means3D: torch.Tensor,  # num_gauss x 3, means3D[:,1] = 0
    scales: torch.Tensor,   # num_gauss x 3, scales[:,1] = eps
    rotations: torch.Tensor,  # num_gauss x 4, 3D quaternions of 2D rotations
    time: float,
):
    return means3D, scales, rotations


def find_checkpoint(model_path: str, iteration: int):
    """
    Find a checkpoint .pth file inside `model_path`.

    If iteration >= 0, look for 'chkpnt{iteration}.pth'.
    If iteration < 0, pick the checkpoint with the largest iteration number.
    Returns full path or None if not found.
    """
    if iteration >= 0:
        ckpt_name = f"chkpnt{iteration}.pth"
        ckpt_path = os.path.join(model_path, ckpt_name)
        if os.path.exists(ckpt_path):
            return ckpt_path
        return None

    if not os.path.isdir(model_path):
        return None

    candidates = [
        f for f in os.listdir(model_path)
        if f.startswith("chkpnt") and f.endswith(".pth")
    ]
    if not candidates:
        return None

    best_iter = -1
    best_path = None
    for fname in candidates:
        # expect "chkpntXXXX.pth"
        num_str = fname[len("chkpnt"):-len(".pth")]
        try:
            it = int(num_str)
        except ValueError:
            continue
        if it > best_iter:
            best_iter = it
            best_path = os.path.join(model_path, fname)

    return best_path


def render_set(
    model_path,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    interp,
    extension,
    generate_points_path,
    mask_path,
    seg=False,
    head="img",
    pipeline_name="img",
):
    """
    Render a set of views for a given Gaussian model.

    - pipeline_name="img"  -> outputs to model_path/render_img/
    - pipeline_name="seg"  -> outputs to model_path/render_mask/
    """

    if pipeline_name == "seg":
        render_dir = os.path.join(model_path, "render_mask")
    else:
        render_dir = os.path.join(model_path, "render_img")

    os.makedirs(render_dir, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        for i in range(interp):
            mask_means = None
            if mask_path != "-1":
                file_path = os.path.join(mask_path, "means3Dmasked", f"{idx:05d}.pt")
                if os.path.exists(file_path):
                    mask_means = torch.load(file_path)[:, [0, -1]]

            rendering = render(
                view,
                gaussians,
                pipeline,
                background,
                interp=interp,
                interp_idx=i,
                modify_func=modify_func,
                idx=idx,
                generate_points_path=generate_points_path,
                mask_means=mask_means,
                train=False,
                seg=seg,
                head=head,
            )["render"].cpu()

            file_name = f"{idx:05d}_{i}{extension}"
            torchvision.utils.save_image(rendering, os.path.join(render_dir, file_name))


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    interp: int,
    extension: str,
    generate_points_path,
    mask_path,
    seg: bool = False,
):
    with torch.no_grad():
        # Build Scene from PLY to get cameras (as before)
        gaussians_scene = gaussianModelRender["gs"](dataset.sh_degree)
        scene = Scene(dataset, gaussians_scene, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if seg:
            # Segmentation rendering: try to use checkpoint (two-head mode),
            # otherwise fall back to PLY (single-head mode).
            ckpt_path = find_checkpoint(dataset.model_path, iteration)
            if ckpt_path is None:
                print(
                    "No checkpoint found in model_path; "
                    "rendering segmentation from PLY model (single-head mode, using image head)."
                )
                gaussians = scene.gaussians
                head_name = "img"
                pipeline_name = "seg"
            else:
                print(f"Loading checkpoint for segmentation head from {ckpt_path}")
                gaussians = gaussianModel["gs"](
                    dataset.sh_degree,
                    dataset.poly_degree,
                    frames=0,
                    use_dff=False,
                )
                model_state, _ = torch.load(ckpt_path, map_location="cuda")
                gaussians.restore(model_state, training_args=None, load_optimizer=False)

                # Decide if this checkpoint really has a seg head
                if hasattr(gaussians, "_opacity_seg") and gaussians._opacity_seg.numel() == gaussians._opacity.numel():
                    head_name = "seg"   # two-head / joint model
                    print("Using segmentation head (two-head joint model).")
                else:
                    head_name = "img"   # fallback: single-head model
                    print("Checkpoint has no valid seg head; using image head for seg pipeline (single-head mode).")

                pipeline_name = "seg"
        else:
            # Photometric rendering: use Gaussians loaded from PLY (as before)
            gaussians = scene.gaussians
            head_name = "img"
            pipeline_name = "img"


        render_set(
            dataset.model_path,
            scene.loaded_iter,
            scene.getTestCameras(),
            gaussians,
            pipeline,
            background,
            interp,
            extension,
            generate_points_path,
            mask_path,
            seg=seg,
            head=head_name,
            pipeline_name=pipeline_name,
        )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--camera", type=str, default="mirror")
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument("--num_pts", type=int, default=100_000)
    parser.add_argument("--skip_train", action="store_false")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--poly_degree", type=int, default=1)
    parser.add_argument("--interp", type=int, default=1)
    parser.add_argument("--extension", type=str, default=".png")
    parser.add_argument("--mask_path", type=str, default="-1")
    parser.add_argument("--generate_points_path", type=str, default="-1")
    parser.add_argument("--pipeline", type=str, choices=["img", "seg"], default="img")

    args = get_combined_args(parser)
    model.gs_type = "gs"
    model.camera = args.camera
    model.distance = args.distance
    model.num_pts = args.num_pts
    model.poly_degree = args.poly_degree

    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args.interp,
        args.extension,
        generate_points_path=args.generate_points_path,
        mask_path=args.mask_path,
        seg=(args.pipeline == "seg"),
    )
