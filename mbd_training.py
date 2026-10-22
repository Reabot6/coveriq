"""Train a small product-propensity model from the public MBD-mini dataset.

MBD's four product labels are anonymous. This script keeps those labels intact
and evaluates on a held-out client fold, so the demo does not imply that the
dataset contains salary or insurance outcomes.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import tarfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as parquet
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TARGET_COLUMNS = [f"target_{index}" for index in range(1, 5)]
FOLD_PATTERN = re.compile(r"fold=(\d+)")


def _fold_from_path(member_name: str) -> int:
    match = FOLD_PATTERN.search(member_name)
    if match is None:
        raise ValueError(f"Could not find fold number in {member_name!r}")
    return int(match.group(1))


def _parquet_tables(archive_path: Path, required_path: str):
    """Yield parquet tables from one of the dataset's compressed tar files."""
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".parquet"):
                continue
            if required_path not in member.name:
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            yield member.name, parquet.read_table(io.BytesIO(source.read())).to_pandas()


def load_latest_targets(targets_archive: Path, customers_per_fold: int) -> pd.DataFrame:
    """Load one latest reporting-month label row per client, stratified by fold."""
    frames = []
    for member_name, frame in _parquet_tables(targets_archive, "targets/fold="):
        frame["fold"] = _fold_from_path(member_name)
        frames.append(frame)

    if not frames:
        raise FileNotFoundError("No target parquet files were found in the MBD archive.")

    targets = pd.concat(frames, ignore_index=True)
    targets["mon"] = pd.to_datetime(targets["mon"], errors="coerce")
    targets = targets.dropna(subset=["mon"])
    targets = targets.sort_values(["fold", "client_id", "mon"])
    latest = targets.drop_duplicates(["fold", "client_id"], keep="last")

    if customers_per_fold > 0:
        # Hash ordering gives a repeatable sample without using client IDs as features.
        latest["_sample_key"] = pd.util.hash_pandas_object(
            latest["client_id"], index=False
        ).astype("uint64")
        latest = (
            latest.sort_values(["fold", "_sample_key"])
            .groupby("fold", group_keys=False)
            .head(customers_per_fold)
            .drop(columns="_sample_key")
        )

    return latest.reset_index(drop=True)


def _transaction_features(transactions: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Aggregate anonymized transaction amount and activity before each label date."""
    if transactions.empty:
        return pd.DataFrame(columns=["client_id"])

    events = transactions.copy()
    events["event_time"] = pd.to_datetime(events["event_time"], errors="coerce", utc=True)
    events["amount"] = pd.to_numeric(events["amount"], errors="coerce")
    events = events.dropna(subset=["event_time", "amount"])

    cutoffs = targets.set_index("client_id")["mon"].dt.tz_localize("UTC")
    events["report_month"] = events["client_id"].map(cutoffs)
    events = events.dropna(subset=["report_month"])
    # The target refers to a product taken in the month after the reporting
    # date. Only transactions through that reporting date enter the features.
    events = events[
        (events["event_time"] <= events["report_month"])
        & (events["event_time"] > events["report_month"] - pd.Timedelta(days=365))
    ].copy()

    if events.empty:
        return pd.DataFrame(columns=["client_id"])

    events["month"] = events["event_time"].dt.strftime("%Y-%m")
    grouped = events.groupby("client_id", sort=False)
    summary = grouped.agg(
        transaction_count=("amount", "size"),
        amount_total=("amount", "sum"),
        amount_mean=("amount", "mean"),
        amount_std=("amount", "std"),
        amount_min=("amount", "min"),
        amount_max=("amount", "max"),
        active_days=("event_time", lambda values: values.dt.date.nunique()),
        active_months=("month", "nunique"),
    ).reset_index()

    monthly = (
        events.assign(month_net=events["amount"])
        .groupby(["client_id", "month"], sort=False)["month_net"]
        .sum()
        .reset_index()
    )
    month_summary = monthly.groupby("client_id")["month_net"].agg(
        monthly_net_mean="mean", monthly_net_std="std", monthly_net_min="min"
    ).reset_index()
    return summary.merge(month_summary, on="client_id", how="left")


def load_transaction_features(
    detail_archive: Path,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    """Read only transaction parquet members and retain sampled target clients."""
    by_fold = {
        int(fold): set(group["client_id"])
        for fold, group in targets.groupby("fold", sort=False)
    }
    transaction_frames: dict[int, list[pd.DataFrame]] = {fold: [] for fold in by_fold}

    for member_name, frame in _parquet_tables(detail_archive, "detail/trx/fold="):
        fold = _fold_from_path(member_name)
        if fold not in by_fold or "client_id" not in frame:
            continue
        frame = frame[frame["client_id"].isin(by_fold[fold])]
        if not frame.empty:
            transaction_frames[fold].append(frame)

    feature_frames = []
    for fold, target_rows in targets.groupby("fold", sort=False):
        parts = transaction_frames[int(fold)]
        transactions = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        features = _transaction_features(transactions, target_rows)
        features["fold"] = int(fold)
        feature_frames.append(features)

    if not feature_frames:
        raise FileNotFoundError("No transaction parquet files were found in the detail archive.")
    return pd.concat(feature_frames, ignore_index=True)


def _rank_precision(y_true: np.ndarray, probabilities: np.ndarray, fraction: float = 0.05) -> float:
    count = max(1, int(len(y_true) * fraction))
    selected = np.argsort(probabilities)[-count:]
    return float(y_true[selected].mean())


def train_and_evaluate(
    targets_archive: Path,
    detail_archive: Path,
    output_dir: Path,
    customers_per_fold: int = 20_000,
    test_fold: int = 4,
) -> dict:
    """Fit one classifier per anonymous product and report held-out metrics."""
    targets = load_latest_targets(targets_archive, customers_per_fold)
    features = load_transaction_features(detail_archive, targets)
    data = targets.merge(features, on=["client_id", "fold"], how="left")
    columns = [
        column
        for column in features.columns
        if column not in {"client_id", "fold"}
    ]
    data[columns] = data[columns].replace([np.inf, -np.inf], np.nan)

    train = data[data["fold"] != test_fold]
    test = data[data["fold"] == test_fold]
    if train.empty or test.empty:
        raise ValueError(f"Fold {test_fold} must have both training and held-out clients.")

    output_dir.mkdir(parents=True, exist_ok=True)
    models = {}
    report = {
        "dataset": "MBD-mini",
        "training_clients": int(train["client_id"].nunique()),
        "held_out_clients": int(test["client_id"].nunique()),
        "held_out_fold": int(test_fold),
        "features": columns,
        "products": {},
        "interpretation": (
            "Four anonymous MBD banking-product propensity scores. They are not insurance "
            "eligibility, affordability, or Zenith customer predictions."
        ),
    }

    for target in TARGET_COLUMNS:
        if train[target].nunique() < 2:
            raise ValueError(f"Training fold has only one class for {target}.")

        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        )
        model.fit(train[columns], train[target].astype(int))
        probabilities = model.predict_proba(test[columns])[:, 1]
        actual = test[target].astype(int).to_numpy()
        report["products"][target] = {
            "positive_rate": float(actual.mean()),
            "roc_auc": float(roc_auc_score(actual, probabilities))
            if len(np.unique(actual)) == 2
            else None,
            "average_precision": float(average_precision_score(actual, probabilities)),
            "precision_at_top_5_percent": _rank_precision(actual, probabilities),
            "top_5_percent_lift_over_random": (
                float(_rank_precision(actual, probabilities) / actual.mean())
                if actual.mean() > 0
                else None
            ),
        }
        models[target] = model

    joblib.dump({"models": models, "features": columns}, output_dir / "mbd_propensity.joblib")
    with (output_dir / "mbd_evaluation.json").open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/mbd-mini"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--customers-per-fold", type=int, default=20_000)
    parser.add_argument("--test-fold", type=int, default=4)
    args = parser.parse_args()

    report = train_and_evaluate(
        args.data_dir / "targets.tar.gz",
        args.data_dir / "detail.tar.gz",
        args.output_dir,
        customers_per_fold=args.customers_per_fold,
        test_fold=args.test_fold,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
