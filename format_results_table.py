
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def parse_text_results(text: str) -> tuple:

    lines = text.strip().split('\n')
    fold_data = []
    summary_data = {}
    

    for line in lines:
        line = line.strip()
        if not line or line.startswith('=') or line.startswith('-') or 'Fold' in line:
            continue
        

        parts = line.split()
        if len(parts) >= 9 and parts[0].isdigit():
            fold_id = int(parts[0])
            metrics = {
                'Fold': fold_id,
                'Acc.': float(parts[1]),
                'Prec.': float(parts[2]),
                'Rec.': float(parts[3]),
                'Spec.': float(parts[4]),
                'F1': float(parts[5]),
                'MCC': float(parts[6]),
                'AUC': float(parts[7]),
                'AUPR': float(parts[8])
            }
            fold_data.append(metrics)
        

        if len(parts) >= 3 and parts[0] in ['Acc.', 'Prec.', 'Rec.', 'Spec.', 'F1', 'MCC', 'AUC', 'AUPR']:
            metric_name = parts[0].rstrip('.')
            if metric_name not in summary_data:
                summary_data[metric_name] = {}
            if len(parts) == 3:
                summary_data[metric_name]['mean'] = float(parts[1])
                summary_data[metric_name]['std'] = float(parts[2])
    
    return fold_data, summary_data


def create_dataframe(fold_data: list, summary_data: dict, model_name: str = "CircR2Dise") -> pd.DataFrame:

    df_folds = pd.DataFrame(fold_data)
    

    if not summary_data:
        metrics = ['Acc.', 'Prec.', 'Rec.', 'Spec.', 'F1', 'MCC', 'AUC', 'AUPR']
        summary_data = {}
        for metric in metrics:
            metric_key = metric.rstrip('.')
            values = df_folds[metric].values
            summary_data[metric_key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
    

    summary_row = {'Fold': 'Average'}
    for metric in ['Acc.', 'Prec.', 'Rec.', 'Spec.', 'F1', 'MCC', 'AUC', 'AUPR']:
        metric_key = metric.rstrip('.')
        if metric_key in summary_data:
            summary_row[metric] = summary_data[metric_key]['mean']
        else:
            summary_row[metric] = df_folds[metric].mean()
    
    std_row = {'Fold': 'Std'}
    for metric in ['Acc.', 'Prec.', 'Rec.', 'Spec.', 'F1', 'MCC', 'AUC', 'AUPR']:
        metric_key = metric.rstrip('.')
        if metric_key in summary_data:
            std_row[metric] = summary_data[metric_key]['std']
        else:
            std_row[metric] = df_folds[metric].std()
    

    df_summary = pd.DataFrame([summary_row, std_row])
    df = pd.concat([df_folds, df_summary], ignore_index=True)
    

    df = df.rename(columns={'Fold': model_name})

    for col in ['Acc.', 'Prec.', 'Rec.', 'Spec.', 'F1', 'MCC', 'AUC', 'AUPR']:
        df[col] = df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
    
    return df


def save_to_csv(df: pd.DataFrame, output_path: str):

    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✓  CSV: {output_path}")


def save_to_excel(df: pd.DataFrame, output_path: str):

    df.to_excel(output_path, index=False, engine='openpyxl')
    print(f"✓  Excel: {output_path}")


def save_to_image(df: pd.DataFrame, output_path: str, figsize=(12, 4)):

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')
    

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    

    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    

    for i in range(len(df.columns)):
        if len(df) >= 2:

            table[(len(df) - 1, i)].set_facecolor('#E8F5E9')

            table[(len(df), i)].set_facecolor('#F1F8E9')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"✓ : {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "--input",
        type=str,
        help=""
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results_table",
        help=""
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "excel", "png", "all"],
        default="all",
        help=""
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="CircR2Dise",
        help=""
    )
    parser.add_argument(
        "--text",
        type=str,
        help=""
    )
    
    args = parser.parse_args()
    

    if args.text:
        text = args.text
    elif args.input:
        with open(args.input, 'r', encoding='utf-8') as f:
            text = f.read()
    else:

        print("")
        text = ""
        try:
            while True:
                line = input()
                text += line + "\n"
        except EOFError:
            pass
    

    fold_data, summary_data = parse_text_results(text)
    
    if not fold_data:
        print("")
        return
    
    # 创建 DataFrame
    df = create_dataframe(fold_data, summary_data, args.model_name)
    
    # 显示表格
    print("\n" + "=" * 80)
    print("")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80 + "\n")
    
    # 保存文件
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.format in ["csv", "all"]:
        save_to_csv(df, str(output_path) + ".csv")
    
    if args.format in ["excel", "all"]:
        try:
            save_to_excel(df, str(output_path) + ".xlsx")
        except ImportError:
            print("")
    
    if args.format in ["png", "all"]:
        save_to_image(df, str(output_path) + ".png")


if __name__ == "__main__":
    # 示例：直接使用文本数据
    example_text = """

"""
    
    # 如果直接运行，可以使用示例数据测试
    import sys
    if len(sys.argv) == 1:
        print("：")
        print("  python format_results_table.py --text \"" --output results_table")
        print("  python format_results_table.py --input results.txt --output results_table")
        print("\n：")
        print("  python format_results_table.py --text \"$(cat results.txt)\" --output results_table")

        print("\n" + "=" * 80)
        main()
    else:
        main()
