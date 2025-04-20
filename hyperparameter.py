import subprocess
import time
import os
import csv

# 🔧 Hyperparameter grid
sh_degrees = [2, 3, 4]
learning_rates = [0.0001, 0.0002]
ssim_lambdas = [0.0, 0.2]
num_randoms = [30000, 50000, 100000] 

data_path = "car_images/"
base_output_dir = "outputs"
results_file = "tuning_results.csv"
evaluate_script = "evaluate_reconstruction.py"

# 🗂️ Prepare CSV if not exists
if not os.path.exists(results_file):
    with open(results_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Run Name", "SH Degree", "Learning Rate", "SSIM Lambda", "PSNR", "SSIM", "LPIPS"])

# 🧪 Loop over parameter combinations
for sh in sh_degrees:
    for lr in learning_rates:
        for ssim in ssim_lambdas:
            for num_random in num_randoms:
                run_name = f"sh{sh}_lr{lr}_ssim{ssim}_n{num_random}".replace(".", "")
                ...
                train_cmd = [
                    "ns-train", "splatfacto",
                    "--data", data_path,
                    "--max-num-iterations", "100000",
                    "--pipeline.model.sh-degree", str(sh),
                    "--optimizers.means.optimizer.lr", str(lr),
                    "--pipeline.model.ssim-lambda", str(ssim),
                    "--pipeline.model.random-init", "True",
                    "--pipeline.model.num-random", str(num_random),
                    "--pipeline.model.random-scale", "5.0",
                    "--experiment-name", run_name,
                    "--viewer.quit-on-train-completion", "True",
                    "--vis", "viewer"
                ]

                subprocess.run(train_cmd)

                # 🧭 Find latest timestamp folder
                exp_dir = os.path.join(base_output_dir, run_name, "splatfacto")
                timestamps = sorted(os.listdir(exp_dir))
                latest = timestamps[-1]
                config_path = os.path.join(exp_dir, latest, "config.yml")

                # 🎞️ Render output
                render_cmd = [
                    "ns-render", "dataset",
                    "--load-config", config_path,
                    "--split=train+test",
                    "--rendered_output_names=rgb"
                ]
                subprocess.run(render_cmd)

                # 🧠 Run evaluation
                eval_cmd = [
                    "python3", evaluate_script,
                    "--rendered-dirs", "renders/train/rgb,renders/test/rgb",
                    "--gt-dir", "car_images/"
                ]
                eval_output = subprocess.check_output(eval_cmd).decode()

                # 🧾 Parse scores from printed output
                try:
                    psnr = float(eval_output.split("PSNR:")[1].split("\n")[0].strip())
                    ssim_val = float(eval_output.split("SSIM:")[1].split("\n")[0].strip())
                    lpips = float(eval_output.split("LPIPS:")[1].split("\n")[0].strip())
                except Exception as e:
                    print(f"⚠️ Failed to parse evaluation output: {e}")
                    psnr, ssim_val, lpips = None, None, None

                # 📝 Save results
                with open(results_file, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([run_name, sh, lr, ssim, psnr, ssim_val, lpips])

                print(f"✅ Finished run: {run_name} - PSNR: {psnr}, SSIM: {ssim_val}, LPIPS: {lpips}")
