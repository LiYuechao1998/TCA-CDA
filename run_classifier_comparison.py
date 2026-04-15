
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
        print(f"{cwd}")
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
        print(f"{exp_name}")
        print(f"{output_file}\n")
    else:
        print(f" {exp_name}")
        print(f"{output_file}\n")
    
    return result.returncode == 0


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    results_dir = os.path.join(script_dir, "ablation_results", "classifier_comparison")
    os.makedirs(results_dir, exist_ok=True)

    script_path = os.path.join(script_dir, "link_pred_circdisease.py")
    if not os.path.exists(script_path):
        return

    base_args = {
        "script": script_path,
        "feat_dim": 64,
        "epochs": 1000,
        "lr": 3e-4,
        "rank_lambda": 0.1,
        "rank_margin": 0.5,
        "num_layers": 6,
        "tlayer_attn_dropout": 0.1,
        "tlayer_ffn_dropout": 0.1,
    }
    

    classifier_configs = [
        {"name": "LINEAR_MAPPING", "classifier_type": "LINEAR_MAPPING", "desc": "Linear Mapping（线性映射）"},
        {"name": "BILINEAR_INTERACTION", "classifier_type": "BILINEAR_INTERACTION", "desc": "Bilinear Interaction（双线性交互）"},
        {"name": "DOT_PRODUCT", "classifier_type": "DOT_PRODUCT", "desc": "Dot-Product Similarity（点积相似度）"},
        {"name": "HYBRID", "classifier_type": "HYBRID", "desc": "Hybrid Formulation（混合 formulation）"},
        {"name": "MLP", "classifier_type": "MLP", "desc": "MLP（多层感知机）"},
    ]
    

    # 运行所有实验
    success_count = 0
    total_count = len(classifier_configs)
    
    for i, config in enumerate(classifier_configs, 1):
        # 复制配置，避免修改原始配置
        config_copy = copy.deepcopy(config)
        exp_name = config_copy.pop("name")
        desc = config_copy.pop("desc", "")
        cmd_params = copy.deepcopy(base_args)
        cmd_params.pop("script")
        cmd_params.update(config_copy)
        cmd = [
            sys.executable,
            base_args["script"],
        ]

        for key, value in cmd_params.items():
            cmd.extend([f"--{key}", str(value)])

        output_file = os.path.join(results_dir, f"classifier_{exp_name.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        

        print(f"\n[{i}/{total_count}]  {exp_name} ({desc})")
        if run_experiment(cmd, exp_name, output_file, cwd=script_dir):
            success_count += 1

if __name__ == "__main__":
    main()
