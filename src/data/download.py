"""Download and save the Telco Customer Churn dataset."""

import pathlib
import urllib.request
import pandas as pd

RAW_DIR = pathlib.Path(__file__).parents[2] / "data" / "raw"
URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
    "/master/data/Telco-Customer-Churn.csv"
)


def download() -> pathlib.Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / "telco_churn.csv"
    if dest.exists():
        print(f"Already exists: {dest}")
        return dest
    print(f"Downloading to {dest}...")
    urllib.request.urlretrieve(URL, dest)
    print(f"Downloaded {dest.stat().st_size / 1024:.1f} KB")
    return dest


if __name__ == "__main__":
    download()
    df = pd.read_csv(RAW_DIR / "telco_churn.csv")
    print(f"Shape: {df.shape}")
    print(df["Churn"].value_counts(normalize=True).round(3))
