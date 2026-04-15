
import subprocess
import sys
import os
import copy
from datetime import datetime

def run_experiment(cmd, exp_name, output_file, cwd=None):
    print(f"\n{'='*60}")
    print(f" {exp_name}")
    print(f" {' '.join(cmd)}")
    if cwd:
        print(f" {cwd}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f" {exp_name}\n")
        f.write(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f" {' '.join(cmd)}\n")
        f.write(f"{'='*60}\n\n")
        f.write("STDOUT:\n")
        f.write(result.stdout)
        f.write("\n\nSTDERR:\n")
        f.write(result.stderr)
    
    if result.returncode == 0:
        print(f" {exp_name}")
        print(f" {output_file}\n")
    else:
        print(f" {exp_name}")
        print(f" {output_file}\n")
    
    return result.returncode == 0


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    results_dir = os.path.join(script_dir, "ablation_results")
    os.makedirs(results_dir, exist_ok=True)
    script_path = os.path.join(script_dir, "link_pred_circdisease.py")
    if not os.path.exists(script_path):
        print(f" {script_path}")
        return
    
    base_args = {
        "script": script_path,
        "feat_dim": 64,
        "epochs": 1000,
        "lr": 3e-4,
        "rank_lambda": 0.1,
        "classifier_type": "HYBRID",
    }

    success_count = 0
    total_count = len(ablation_configs)
    
    for i, config in enumerate(ablation_configs, 1):
        config_copy = copy.deepcopy(config)
        exp_name = config_copy.pop("name")
        cmd_params = {
            "feat_dim": base_args["feat_dim"],
            "epochs": base_args["epochs"],
            "lr": base_args["lr"],
            "rank_lambda": base_args["rank_lambda"],
            "classifier_type": base_args["classifier_type"],
        }
        cmd_params.update(config_copy)
        cmd = [
            sys.executable,
            base_args["script"],
        ]
        for key, value in cmd_params.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
                else:
                    if key == "use_gconv" and not value:
                        cmd.append("--no_gconv")
                    elif key == "use_tlayer" and not value:
                        cmd.append("--no_tlayer")
                    else:
                        pass
            else:
                cmd.extend([f"--{key}", str(value)])

        output_file = os.path.join(results_dir, f"{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

        if run_experiment(cmd, exp_name, output_file, cwd=script_dir):
            success_count += 1



if __name__ == "__main__":
    main()
