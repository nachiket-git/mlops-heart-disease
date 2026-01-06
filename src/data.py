import argparse
import os
import pandas as pd
from urllib.request import urlretrieve

DEFAULT_URL = (
    # Public mirror of UCI Heart dataset in ready-to-use CSV format.
    # Source repository: sharmaroshan/Heart-UCI-Dataset
    "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"
)

REQUIRED_COLUMNS = [
    "age","sex","cp","trestbps","chol","fbs","restecg","thalach",
    "exang","oldpeak","slope","ca","thal","target"
]

def download_csv(url: str, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    urlretrieve(url, out_path)
    return out_path

def load_and_validate(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}. Found: {list(df.columns)}")
    return df

def clean_basic(df: pd.DataFrame) -> pd.DataFrame:
    # This dataset is usually clean; still demonstrate robust cleaning:
    df = df.copy()
    # Replace common 'unknown' placeholders if any
    df.replace(["?", "NA", "NaN", ""], pd.NA, inplace=True)
    # Median impute numeric columns
    for col in df.columns:
        if df[col].dtype.kind in "if":  # int/float
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0])
    # Ensure target is binary {0,1}
    if set(df["target"].unique()) - {0, 1}:
        df["target"] = (df["target"] > 0).astype(int)
    return df

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("download", help="Download dataset CSV")
    p_dl.add_argument("--url", default=DEFAULT_URL)
    p_dl.add_argument("--out", default="data/raw/heart.csv")

    p_clean = sub.add_parser("clean", help="Clean dataset and write to output path")
    p_clean.add_argument("--csv", default="data/raw/heart.csv")
    p_clean.add_argument("--out", default="data/processed/heart_clean.csv")

    args = parser.parse_args()

    if args.cmd == "download":
        path = download_csv(args.url, args.out)
        print(f"Downloaded to: {path}")

    if args.cmd == "clean":
        df = load_and_validate(args.csv)
        df = clean_basic(df)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"Saved cleaned CSV to: {args.out} (rows={len(df)}, cols={df.shape[1]})")

if __name__ == "__main__":
    main()
