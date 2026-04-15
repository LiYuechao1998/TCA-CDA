

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
    
    # 保存输出
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
        print(f"✓ {exp_name}")
        print(f"  {output_file}\n")
    else:
        print(f" {exp_name}")
        print(f" {output_file}\n")
    
    return result.returncode == 0


def main():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    results_dir = os.path.join(script_dir, "ablation_results", "circRNADisease_param_comparison")
    os.makedirs(results_dir, exist_ok=True)

    script_path = os.path.join(script_dir, "link_pred_circdisease.py")
    if not os.path.exists(script_path):
        print(f"{script_path}")
        return
    
    base_args = {
        "script": script_path,
        "data_dir": "..\\3-circRNADisease",
        "feat_dim": 64,
        "epochs": 1500,
        "lr": 3e-4,
        "rank_lambda": 0.1,
        "rank_margin": 0.5,
        "num_layers": 6,
        "tlayer_attn_dropout": 0.1,
        "tlayer_ffn_dropout": 0.1,
        "classifier_type": "linear_bilinear",
        "embedding_dim": None,
    }
    

    ablation_configs = [
        {"name": "num_layers_4", "num_layers": 4},
        {"name": "num_layers_8", "num_layers": 8},
    ]



    success_count = 0
    total_count = len(ablation_configs)
    
    for i, config in enumerate(ablation_configs, 1):

        config_copy = copy.deepcopy(config)
        exp_name = config_copy.pop("name")
        

        cmd_params = {
            "data_dir": base_args["data_dir"],
            "feat_dim": base_args["feat_dim"],
            "epochs": base_args["epochs"],
            "lr": base_args["lr"],
            "rank_lambda": base_args["rank_lambda"],
            "rank_margin": base_args["rank_margin"],
            "num_layers": base_args["num_layers"],
            "tlayer_attn_dropout": base_args["tlayer_attn_dropout"],
            "tlayer_ffn_dropout": base_args["tlayer_ffn_dropout"],
            "classifier_type": base_args["classifier_type"],
            "embedding_dim": base_args["embedding_dim"],
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
            elif value is None:
                pass
            else:
                cmd.extend([f"--{key}", str(value)])

        output_file = os.path.join(results_dir, f"{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

        if run_experiment(cmd, exp_name, output_file, cwd=script_dir):
            success_count += 1



if __name__ == "__main__":
    main()
