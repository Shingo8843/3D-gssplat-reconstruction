import os
import argparse
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips  # For LPIPS

def load_image(path, size=(800, 800)):
    img = Image.open(path).convert('RGB').resize(size)
    return np.array(img)

def evaluate(rendered_dirs, gt_dir):
    loss_fn = lpips.LPIPS(net='alex')
    psnr_scores, ssim_scores, lpips_scores = [], [], []

    image_extensions = ('.png', '.jpg', '.jpeg')

    # Load GT files
    gt_files = sorted([
        os.path.join(gt_dir, f) for f in os.listdir(gt_dir)
        if f.lower().endswith(image_extensions) and not f.startswith('.')
    ])

    # Load and merge rendered files
    rendered_files = []
    for r_dir in rendered_dirs:
        r_files = [
            os.path.join(r_dir, f) for f in os.listdir(r_dir)
            if f.lower().endswith(image_extensions) and not f.startswith('.')
        ]
        rendered_files.extend(r_files)

    rendered_files = sorted(rendered_files)

    assert len(gt_files) == len(rendered_files), f"Mismatch: {len(gt_files)} GT vs {len(rendered_files)} rendered"

    for r_path, gt_path in zip(rendered_files, gt_files):
        img_r = load_image(r_path)
        img_gt = load_image(gt_path)

        img_r = img_r / 255.0
        img_gt = img_gt / 255.0

        psnr_val = psnr(img_gt, img_r, data_range=1.0)
        ssim_val = ssim(img_gt, img_r, channel_axis=-1, data_range=1.0)
        lpips_val = loss_fn(lpips.im2tensor(img_gt), lpips.im2tensor(img_r)).item()

        psnr_scores.append(psnr_val)
        ssim_scores.append(ssim_val)
        lpips_scores.append(lpips_val)

    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    avg_lpips = np.mean(lpips_scores)

    print(f"PSNR: {avg_psnr:.2f}")
    print(f"SSIM: {avg_ssim:.4f}")
    print(f"LPIPS: {avg_lpips:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rendered-dirs", type=str, default="renders/train/rgb,renders/test/rgb",
                        help="Comma-separated list of rendered image directories")
    parser.add_argument("--gt-dir", type=str, default="car_images/",
                        help="Directory with ground truth images")

    args = parser.parse_args()
    rendered_dirs = [r.strip() for r in args.rendered_dirs.split(",")]

    evaluate(rendered_dirs, args.gt_dir)
    print("Evaluation complete.")