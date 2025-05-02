
import subprocess
import time
import os
import csv

param_combinations = [(2, 0.0001, 0.0, 50000, 50, 0.0003), (2, 0.0001, 0.0, 50000, 50, 0.0005), (2, 0.0001, 0.0, 50000, 100, 0.0003), (2, 0.0001, 0.0, 50000, 100, 0.0005), (2, 0.0001, 0.0, 100000, 50, 0.0003), (2, 0.0001, 0.0, 100000, 50, 0.0005), (2, 0.0001, 0.0, 100000, 100, 0.0003), (2, 0.0001, 0.0, 100000, 100, 0.0005), (2, 0.0001, 0.2, 50000, 50, 0.0003), (2, 0.0001, 0.2, 50000, 50, 0.0005), (2, 0.0001, 0.2, 50000, 100, 0.0003), (2, 0.0001, 0.2, 50000, 100, 0.0005), (2, 0.0001, 0.2, 100000, 50, 0.0003), (2, 0.0001, 0.2, 100000, 50, 0.0005), (2, 0.0001, 0.2, 100000, 100, 0.0003), (2, 0.0001, 0.2, 100000, 100, 0.0005), (2, 0.0002, 0.0, 50000, 50, 0.0003), (2, 0.0002, 0.0, 50000, 50, 0.0005), (2, 0.0002, 0.0, 50000, 100, 0.0003), (2, 0.0002, 0.0, 50000, 100, 0.0005), (2, 0.0002, 0.0, 100000, 50, 0.0003), (2, 0.0002, 0.0, 100000, 50, 0.0005), (2, 0.0002, 0.0, 100000, 100, 0.0003), (2, 0.0002, 0.0, 100000, 100, 0.0005), (2, 0.0002, 0.2, 50000, 50, 0.0003), (2, 0.0002, 0.2, 50000, 50, 0.0005), (2, 0.0002, 0.2, 50000, 100, 0.0003), (2, 0.0002, 0.2, 50000, 100, 0.0005), (2, 0.0002, 0.2, 100000, 50, 0.0003), (2, 0.0002, 0.2, 100000, 50, 0.0005), (2, 0.0002, 0.2, 100000, 100, 0.0003), (2, 0.0002, 0.2, 100000, 100, 0.0005), (3, 0.0001, 0.0, 50000, 50, 0.0003), (3, 0.0001, 0.0, 50000, 50, 0.0005), (3, 0.0001, 0.0, 50000, 100, 0.0003), (3, 0.0001, 0.0, 50000, 100, 0.0005), (3, 0.0001, 0.0, 100000, 50, 0.0003), (3, 0.0001, 0.0, 100000, 50, 0.0005), (3, 0.0001, 0.0, 100000, 100, 0.0003), (3, 0.0001, 0.0, 100000, 100, 0.0005), (3, 0.0001, 0.2, 50000, 50, 0.0003), (3, 0.0001, 0.2, 50000, 50, 0.0005), (3, 0.0001, 0.2, 50000, 100, 0.0003), (3, 0.0001, 0.2, 50000, 100, 0.0005), (3, 0.0001, 0.2, 100000, 50, 0.0003), (3, 0.0001, 0.2, 100000, 50, 0.0005), (3, 0.0001, 0.2, 100000, 100, 0.0003), (3, 0.0001, 0.2, 100000, 100, 0.0005), (3, 0.0002, 0.0, 50000, 50, 0.0003), (3, 0.0002, 0.0, 50000, 50, 0.0005), (3, 0.0002, 0.0, 50000, 100, 0.0003), (3, 0.0002, 0.0, 50000, 100, 0.0005), (3, 0.0002, 0.0, 100000, 50, 0.0003), (3, 0.0002, 0.0, 100000, 50, 0.0005), (3, 0.0002, 0.0, 100000, 100, 0.0003), (3, 0.0002, 0.0, 100000, 100, 0.0005), (3, 0.0002, 0.2, 50000, 50, 0.0003), (3, 0.0002, 0.2, 50000, 50, 0.0005), (3, 0.0002, 0.2, 50000, 100, 0.0003), (3, 0.0002, 0.2, 50000, 100, 0.0005), (3, 0.0002, 0.2, 100000, 50, 0.0003), (3, 0.0002, 0.2, 100000, 50, 0.0005), (3, 0.0002, 0.2, 100000, 100, 0.0003), (3, 0.0002, 0.2, 100000, 100, 0.0005)]
data_path = "car_images/"
base_output_dir = "outputs"
results_file = "tuning_results.csv"
evaluate_script = "evaluate_reconstruction.py"
chunk_size = 10000
max_iters = 10000

# Add header if the results file does not exist
if not os.path.exists(results_file):
    with open(results_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Run Name", "SH Degree", "Learning Rate", "SSIM Lambda",
            "Num Random", "Refine Every", "Densify Thresh",
            "PSNR", "SSIM", "LPIPS", "Train Time (s)"
        ])

# Run tuning
for sh, lr, ssim, num_random, refine_every, densify_thresh in param_combinations:
    run_name = f"sh{sh}_lr{lr}_ssim{ssim}_n{num_random}_ref{refine_every}_dens{densify_thresh}".replace(".", "")
    print(f"\n🚀 Starting run: {run_name}")

    total_iters = 0
    early_stop = False
    total_train_time = 0  # ⏱️ Initialize timer

    while total_iters < max_iters and not early_stop:
        print(f"⏱️ Training chunk to {total_iters + chunk_size}")

        train_cmd = [
            "ns-train", "splatfacto",
            "--data", data_path,
            "--max-num-iterations", str(chunk_size),
            "--pipeline.model.sh-degree", str(sh),
            "--pipeline.model.ssim-lambda", str(ssim),
            "--optimizers.means.optimizer.lr", str(lr),
            "--pipeline.model.random-init", "True",
            "--pipeline.model.num-random", str(num_random),
            "--pipeline.model.random-scale", "5.0",
            "--pipeline.model.refine-every", str(refine_every),
            "--pipeline.model.densify-grad-thresh", str(densify_thresh),
            "--pipeline.model.densify-size-thresh", "0.005",
            "--pipeline.model.sh-degree-interval", "2000",
            "--pipeline.model.use-scale-regularization", "True",
            "--pipeline.model.max-gauss-ratio", "10.0",
            "--optimizers.means.scheduler.warmup-steps", "1000",
            "--optimizers.means.scheduler.lr-final", "0.00005",
            "--optimizers.means.scheduler.ramp", "cosine",
            "--experiment-name", run_name,
            "--viewer.quit-on-train-completion", "True",
            "--vis", "viewer"
        ]

        if total_iters > 0:
            train_cmd += ["--load-dir", os.path.join(base_output_dir, run_name)]

        # ⏱️ Time training chunk
        start_time = time.time()
        subprocess.run(train_cmd)
        end_time = time.time()
        train_duration = end_time - start_time
        total_train_time += train_duration
        print(f"⏲️ Training chunk duration: {train_duration:.2f} sec")

        total_iters += chunk_size

        # Rendering
        exp_dir = os.path.join(base_output_dir, run_name, "splatfacto")
        latest = sorted(os.listdir(exp_dir))[-1]
        config_path = os.path.join(exp_dir, latest, "config.yml")

        render_cmd = [
            "ns-render", "dataset",
            "--load-config", config_path,
            "--split=train+test",
            "--rendered_output_names=rgb"
        ]
        subprocess.run(render_cmd)

        # Evaluation
        eval_cmd = [
            "python3", evaluate_script,
            "--rendered-dirs", "renders/train/rgb,renders/test/rgb",
            "--gt-dir", "car_images/"
        ]
        eval_output = subprocess.check_output(eval_cmd).decode()

        try:
            psnr = float(eval_output.split("PSNR:")[1].split("\n")[0].strip())
            ssim_val = float(eval_output.split("SSIM:")[1].split("\n")[0].strip())
            lpips = float(eval_output.split("LPIPS:")[1].split("\n")[0].strip())
        except Exception as e:
            print(f"⚠️ Failed to parse output: {e}")
            psnr, ssim_val, lpips = None, None, None

        # Early stopping condition
        if psnr and psnr > 28 or lpips and lpips < 0.08:
            print(f"🛑 Early stopping: PSNR={psnr}, LPIPS={lpips}")
            early_stop = True

    # Save to CSV
    with open(results_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            run_name, sh, lr, ssim, num_random, refine_every, densify_thresh,
            psnr, ssim_val, lpips, round(total_train_time, 2)
        ])

    print(f"✅ Finished: {run_name} → PSNR={psnr}, SSIM={ssim_val}, LPIPS={lpips}, Time={total_train_time:.2f}s")