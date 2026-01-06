import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def save_fig(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def run_eda(csv_path: str, out_dir: str):
    df = pd.read_csv(csv_path)

    # 1) Class balance
    ax = df["target"].value_counts().sort_index().plot(kind="bar")
    ax.set_title("Class Balance (target)")
    ax.set_xlabel("target")
    ax.set_ylabel("count")
    save_fig(os.path.join(out_dir, "class_balance.png"))

    # 2) Histograms for numeric columns
    numeric_cols = [c for c in df.columns if df[c].dtype.kind in "if" and c != "target"]
    df[numeric_cols].hist(figsize=(12, 8))
    save_fig(os.path.join(out_dir, "numeric_histograms.png"))

    # 3) Correlation heatmap (manual with matplotlib)
    corr = df.corr(numeric_only=True)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    cax = ax.imshow(corr.values, aspect="auto")
    ax.set_title("Correlation Heatmap (numeric)")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    fig.colorbar(cax)
    save_fig(os.path.join(out_dir, "corr_heatmap.png"))

    # 4) Save simple EDA summary
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "eda_summary.txt"), "w", encoding="utf-8") as f:
        f.write("Rows: %d\nCols: %d\n\n" % (len(df), df.shape[1]))
        f.write("Missing values per column:\n")
        f.write(df.isna().sum().to_string())
        f.write("\n\nDescribe (numeric):\n")
        f.write(df.describe().to_string())

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--csv", required=True)
    r.add_argument("--out", default="artifacts/eda")
    args = p.parse_args()
    if args.cmd == "run":
        run_eda(args.csv, args.out)
        print(f"EDA artifacts saved to: {args.out}")

if __name__ == "__main__":
    main()
