import json
import numpy as np
import os
from pathlib import Path
from typing import Optional

import cv2
import torch
import torchvision.transforms.functional as F
import tyro
from rich.console import Console
from rich.progress import track
from torchmetrics.functional import mean_squared_error
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from skimage.metrics import structural_similarity as ssim

from dn_splatter.metrics import DepthMetrics
from dn_splatter.utils.utils import depth_path_to_tensor

CONSOLE = Console(width=120)
BATCH_SIZE = 1

def psnr(img1, img2, mask_img):
    mse = torch.sum((img1 - img2) ** 2) / (torch.sum(mask_img) * 3) # 3 for RGB channels
    # mse = (((img1 - img2)) ** 2).mean()
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

@torch.no_grad()
def rgb_eval(base_dir, eval_dir):
    mask_path = base_dir / Path("masks")
    render_path = eval_dir / Path("test/rgb")  # os.path.join(args.data, "/rgb")
    gt_path = base_dir /Path("images")  # os.path.join(args.data, "gt", "rgb")

    mask_ext = os.path.splitext(os.listdir(mask_path)[0])[1]  # Assuming masks have the same extension
    render_ext = os.path.splitext(os.listdir(render_path)[0])[1]
    gt_ext = os.path.splitext(os.listdir(gt_path)[0])[1]
    image_list = [f for f in os.listdir(render_path) if f.endswith((".png", ".jpg"))]
    image_list = [f.split(".")[0] for f in image_list]
    image_list = sorted(image_list, key=lambda x: int(x.split("_")[-1]))
    num_frames = len(image_list)

    # mse = mean_squared_error
    # psnr = PeakSignalNoiseRatio(data_range=1.0)
    # ssim = StructuralSimilarityIndexMeasure(data_range=1.0, kernel_size=11)
    # lpips = LearnedPerceptualImagePatchSimilarity()

    psnr_score_batch = []
    ssim_score_batch = []
    mse_score_batch = []
    lpips_score_batch = []

    CONSOLE.print(
        f"[bold green]Batchifying and evaluating a total of {num_frames} rgb frames"
    )

    for batch_index in track(range(0, num_frames, BATCH_SIZE)):
        CONSOLE.print(
            f"[bold yellow]Evaluating batch {batch_index // BATCH_SIZE} / {num_frames//BATCH_SIZE}"
        )
        batch_frames = image_list[batch_index : batch_index + BATCH_SIZE]
        predicted_rgb = []
        gt_rgb = []

        for i in batch_frames:
            render_img = cv2.imread(os.path.join(render_path, i + render_ext)) / 255
            origin_img = cv2.imread(os.path.join(gt_path, i + gt_ext)) / 255

            # Load mask and apply it
            mask_img = cv2.imread(os.path.join(mask_path, i + mask_ext), cv2.IMREAD_GRAYSCALE) / 255.0
            # mask_img = mask_img.astype(bool)  # Convert to boolean mask
            mask_img = mask_img.reshape(mask_img.shape[0], mask_img.shape[1], 1)
            assert np.sum(mask_img) != 0, f"No foreground found in mask please check your mask in {i}"
            assert set(np.unique(mask_img)) == {0, 1}, f"mask image values are not binary, got {set(np.unique(mask_img))} in {i}"
            
            # render_img = render_img[mask_img].reshape(-1, 1)
            # origin_img = origin_img[mask_img].reshape(-1, 1)
            origin_img = origin_img * mask_img
            render_img = render_img * mask_img
            assert render_img.shape == origin_img.shape, f"render image shape {render_img.shape} does not match origin image shape {origin_img.shape}"

            # Compute SSIM only over the masked (numpy array)
            (ssim_value, ssim_map) = ssim(render_img, origin_img, full=True, channel_axis=-1, data_range=1.0)
            ssim_value_masked = torch.tensor(np.mean(ssim_map[mask_img.squeeze()>0])).to(torch.float32)

            # Compute PSNR only over the masked (torch tensor)
            origin_img = F.to_tensor(origin_img).to(torch.float32)
            render_img = F.to_tensor(render_img).to(torch.float32)
            mask_img = F.to_tensor(mask_img).to(torch.float32)
            psnr_score = psnr(render_img, origin_img, mask_img)

        psnr_score_batch.append(psnr_score)
        ssim_score_batch.append(ssim_value_masked)
        # mse_score = mse(predicted_image, gt_image)
        # mse_score_batch.append(mse_score)
        # psnr_score = psnr(predicted_image, gt_image)
        # psnr_score_batch.append(psnr_score)
        # ssim_score = ssim(predicted_image, gt_image)
        # ssim_score_batch.append(ssim_score)
        # lpips_score = lpips(predicted_image, gt_image)
        # lpips_score_batch.append(lpips_score)

    mean_scores = {
        # "mse": float(torch.stack(mse_score_batch).mean().item()),
        "psnr": float(torch.stack(psnr_score_batch).mean().item()),
        "ssim": float(torch.stack(ssim_score_batch).mean().item()),
        # "lpips": float(torch.stack(lpips_score_batch).mean().item()),
    }
    print(list(mean_scores.keys()))
    print(list(mean_scores.values()))

    with open(os.path.join(eval_dir, "metrics.json"), "w") as outFile:
        print(f"Saving results to {os.path.join(eval_dir, 'metrics.json')}")
        json.dump(mean_scores, outFile, indent=2)


@torch.no_grad()
def depth_eval(data: Path):
    depth_metrics = DepthMetrics()

    render_path = data / Path("depth/raw/")  # os.path.join(args.data, "/rgb")
    gt_path = data / Path("gt/depth/raw")  # os.path.join(args.data, "gt", "rgb")

    depth_list = [f for f in os.listdir(render_path) if f.endswith(".npy")]
    depth_list = sorted(depth_list, key=lambda x: int(x.split(".")[0].split("_")[-1]))

    mse = mean_squared_error

    num_frames = len(depth_list)

    mse_score_batch = []
    abs_rel_score_batch = []
    sq_rel_score_batch = []
    rmse_score_batch = []
    rmse_log_score_batch = []
    a1_score_batch = []
    a2_score_batch = []
    a3_score_batch = []
    CONSOLE.print(
        f"[bold green]Batchifying and evaluating a total of {num_frames} depth frames"
    )

    for batch_index in track(range(0, num_frames, BATCH_SIZE)):
        CONSOLE.print(
            f"[bold yellow]Evaluating batch {batch_index // BATCH_SIZE} / {num_frames//BATCH_SIZE}"
        )
        batch_frames = depth_list[batch_index : batch_index + BATCH_SIZE]
        predicted_depth = []
        gt_depth = []

        for i in batch_frames:
            render_img = depth_path_to_tensor(
                Path(os.path.join(render_path, i))
            ).permute(2, 0, 1)
            origin_img = depth_path_to_tensor(Path(os.path.join(gt_path, i))).permute(
                2, 0, 1
            )

            if origin_img.shape[-2:] != render_img.shape[-2:]:
                render_img = F.resize(
                    render_img, size=origin_img.shape[-2:], antialias=None
                )
            predicted_depth.append(render_img)
            gt_depth.append(origin_img)

        predicted_depth = torch.stack(predicted_depth, 0)
        gt_depth = torch.stack(gt_depth, 0)

        mse_score = mse(predicted_depth, gt_depth)
        mse_score_batch.append(mse_score)
        (abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3) = depth_metrics(
            predicted_depth, gt_depth
        )
        abs_rel_score_batch.append(abs_rel)
        sq_rel_score_batch.append(sq_rel)
        rmse_score_batch.append(rmse)
        rmse_log_score_batch.append(rmse_log)
        a1_score_batch.append(a1)
        a2_score_batch.append(a2)
        a3_score_batch.append(a3)

    mean_scores = {
        "mse": float(torch.stack(mse_score_batch).mean().item()),
        "abs_rel": float(torch.stack(abs_rel_score_batch).mean().item()),
        "sq_rel": float(torch.stack(sq_rel_score_batch).mean().item()),
        "rmse": float(torch.stack(rmse_score_batch).mean().item()),
        "rmse_log": float(torch.stack(rmse_log_score_batch).mean().item()),
        "a1": float(torch.stack(a1_score_batch).mean().item()),
        "a2": float(torch.stack(a2_score_batch).mean().item()),
        "a3": float(torch.stack(a3_score_batch).mean().item()),
    }
    print(list(mean_scores.keys()))
    print(list(mean_scores.values()))
    with open(os.path.join(render_path, "metrics.json"), "w") as outFile:
        print(f"Saving results to {os.path.join(render_path, 'metrics.json')}")
        json.dump(mean_scores, outFile, indent=2)


def depth_eval_faro(data: Path, path_to_faro: Path):
    depth_metrics = DepthMetrics()

    render_path = data / Path("depth/raw/")
    gt_path = path_to_faro

    depth_list = [f for f in os.listdir(render_path) if f.endswith(".png")]
    depth_list = sorted(depth_list, key=lambda x: int(x.split(".")[0].split("_")[-1]))

    mse = mean_squared_error

    num_frames = len(depth_list)

    mse_score_batch = []
    abs_rel_score_batch = []
    sq_rel_score_batch = []
    rmse_score_batch = []
    rmse_log_score_batch = []
    a1_score_batch = []
    a2_score_batch = []
    a3_score_batch = []
    CONSOLE.print(
        f"[bold green]Batchifying and evaluating a total of {num_frames} depth frames"
    )

    for batch_index in track(range(0, num_frames, BATCH_SIZE)):
        CONSOLE.print(
            f"[bold yellow]Evaluating batch {batch_index // BATCH_SIZE} / {num_frames//BATCH_SIZE}"
        )
        batch_frames = depth_list[batch_index : batch_index + BATCH_SIZE]
        predicted_depth = []
        gt_depth = []
        for i in batch_frames:
            render_img = depth_path_to_tensor(
                Path(os.path.join(render_path, i))
            ).permute(2, 0, 1)

            if not Path(os.path.join(gt_path, i)).exists():
                print("could not find frame ", i, " skipping it...")
                continue
            origin_img = depth_path_to_tensor(Path(os.path.join(gt_path, i))).permute(
                2, 0, 1
            )
            if origin_img.shape[-2:] != render_img.shape[-2:]:
                render_img = F.resize(
                    render_img, size=origin_img.shape[-2:], antialias=None
                )
            predicted_depth.append(render_img)
            gt_depth.append(origin_img)

        predicted_depth = torch.stack(predicted_depth, 0)
        gt_depth = torch.stack(gt_depth, 0)

        mse_score = mse(predicted_depth, gt_depth)
        mse_score_batch.append(mse_score)

        (abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3) = depth_metrics(
            predicted_depth, gt_depth
        )
        abs_rel_score_batch.append(abs_rel)
        sq_rel_score_batch.append(sq_rel)
        rmse_score_batch.append(rmse)
        rmse_log_score_batch.append(rmse_log)
        a1_score_batch.append(a1)
        a2_score_batch.append(a2)
        a3_score_batch.append(a3)

    mean_scores = {
        "mse": float(torch.stack(mse_score_batch).mean().item()),
        "abs_rel": float(torch.stack(abs_rel_score_batch).mean().item()),
        "sq_rel": float(torch.stack(sq_rel_score_batch).mean().item()),
        "rmse": float(torch.stack(rmse_score_batch).mean().item()),
        "rmse_log": float(torch.stack(rmse_log_score_batch).mean().item()),
        "a1": float(torch.stack(a1_score_batch).mean().item()),
        "a2": float(torch.stack(a2_score_batch).mean().item()),
        "a3": float(torch.stack(a3_score_batch).mean().item()),
    }
    print("faro scanner metrics")
    print(list(mean_scores.keys()))
    print(list(mean_scores.values()))


def mask_rendering_evaluation(
    base_dir,
    eval_dir,
    eval_rgb: bool = True,
    eval_depth: bool = False,
    eval_faro: bool = False,
    path_to_faro: Optional[Path] = None,
):
    if eval_rgb:
        rgb_eval(base_dir, eval_dir)
    # if eval_depth:
    #     depth_eval(data)
    # if eval_faro:
    #     assert path_to_faro is not None, "need to specify faro path"
    #     depth_eval_faro(data, path_to_faro=path_to_faro)


if __name__ == "__main__":
    tyro.cli(mask_rendering_evaluation)
