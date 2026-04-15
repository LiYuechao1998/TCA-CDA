
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def parse_result_file(file_path: str) -> dict:

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    

    exp_name_match = re.search(r':\s*(.+)', content)
    exp_name = exp_name_match.group(1).strip() if exp_name_match else Path(file_path).stem
    

    classifier_match = re.search(r':\s*(.+)', content)
    classifier_name = classifier_match.group(1).strip() if classifier_match else "未知"
    

    if classifier_name == "" or "（" in classifier_name:

        filename = Path(file_path).stem
        if "linear_mapping" in filename.lower():
            classifier_name = "LINEAR_MAPPING"
        elif "bilinear_interaction" in filename.lower():
            classifier_name = "BILINEAR_INTERACTION"
        elif "dot_product" in filename.lower():
            classifier_name = "DOT_PRODUCT"
        elif "hybrid" in filename.lower():
            classifier_name = "HYBRID"
        elif "mlp" in filename.lower():
            classifier_name = "MLP"
    

    metrics = {}
    summary_pattern = r'(Acc\.|Prec\.|Rec\.|Spec\.|F1|MCC|AUC|AUPR)\s+([\d.]+)\s+([\d.]+)'
    matches = re.findall(summary_pattern, content)
    
    for metric, mean, std in matches:
        metric_key = metric.rstrip('.')
        metrics[f'{metric_key}_mean'] = float(mean)
        metrics[f'{metric_key}_std'] = float(std)
    

    if not metrics:
        fold_pattern = r'(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
        fold_matches = re.findall(fold_pattern, content)
        
        if fold_matches:
            metric_names = ['Acc', 'Prec', 'Rec', 'Spec', 'F1', 'MCC', 'AUC', 'AUPR']
            fold_data = []
            for match in fold_matches:
                values = [float(x) for x in match[1:]]
                fold_data.append(values)
            
            fold_array = np.array(fold_data)
            for i, metric_name in enumerate(metric_names):
                metrics[f'{metric_name}_mean'] = float(np.mean(fold_array[:, i]))
                metrics[f'{metric_name}_std'] = float(np.std(fold_array[:, i]))
    
    return {
        'experiment': exp_name,
        'classifier': classifier_name,
        **metrics
    }


def create_comparison_table(results_dir: str) -> pd.DataFrame:

    results_dir = Path(results_dir)
    result_files = list(results_dir.glob("classifier_*.txt"))
    
    if not result_files:
        print(f"")
        return pd.DataFrame()

    all_results = []
    for file_path in result_files:
        try:
            result = parse_result_file(str(file_path))
            all_results.append(result)
        except Exception as e:

            continue
    
    if not all_results:

        return pd.DataFrame()
    

    df = pd.DataFrame(all_results)
    

    df = df.set_index('classifier')
    

    metric_names = ['Acc', 'Prec', 'Rec', 'Spec', 'F1', 'MCC', 'AUC', 'AUPR']
    mean_cols = [f'{m}_mean' for m in metric_names]
    std_cols = [f'{m}_std' for m in metric_names]

    comparison_data = []
    for idx in df.index:
        row = {'': idx}
        for metric in metric_names:
            mean_key = f'{metric}_mean'
            std_key = f'{metric}_std'
            if mean_key in df.columns and std_key in df.columns:
                mean_val = df.loc[idx, mean_key]
                std_val = df.loc[idx, std_key]
                row[metric] = f"{mean_val:.4f} ± {std_val:.4f}"
            elif mean_key in df.columns:
                row[metric] = f"{df.loc[idx, mean_key]:.4f}"
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.set_index('')
    
    return comparison_df


def save_comparison_table(df: pd.DataFrame, output_path: str, format_type: str = "all"):

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format_type in ["csv", "all"]:
        csv_path = str(output_path) + ".csv"
        df.to_csv(csv_path, encoding='utf-8-sig')
        print(f"✓  CSV: {csv_path}")
    
    if format_type in ["excel", "all"]:
        try:
            excel_path = str(output_path) + ".xlsx"
            df.to_excel(excel_path, engine='openpyxl')
            print(f"✓  Excel: {excel_path}")
        except ImportError:

    
    if format_type in ["png", "all"]:
        png_path = str(output_path) + ".png"
        fig, ax = plt.subplots(figsize=(14, max(6, len(df) * 0.5)))
        ax.axis('tight')
        ax.axis('off')
        

        table = ax.table(
            cellText=df.values,
            rowLabels=df.index,
            colLabels=df.columns,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        

        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        

        for i in range(len(df.index)):
            table[(i + 1, -1)].set_facecolor('#E3F2FD')
            table[(i + 1, -1)].set_text_props(weight='bold')
        
        plt.title('', fontsize=14, fontweight='bold', pad=20)
        plt.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
        print(f"✓: {png_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="ablation_results/classifier_comparison",
        help=""
    )
    parser.add_argument(
        "--output",
        type=str,
        default="classifier_comparison_table",
        help=""
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "excel", "png", "all"],
        default="all",
        help=""
    )
    
    args = parser.parse_args()
    

    script_dir = Path(__file__).parent
    results_dir = script_dir / args.results_dir
    
    if not results_dir.exists():
        print(f": {results_dir}")
        return
    

    print(f" {results_dir} ")
    df = create_comparison_table(str(results_dir))
    
    if df.empty:
        print("")
        return
    

    print("\n" + "=" * 80)
    print("")
    print("=" * 80)
    print(df.to_string())
    print("=" * 80 + "\n")
    

    output_path = script_dir / args.output
    save_comparison_table(df, str(output_path), args.format)


if __name__ == "__main__":
    main()
