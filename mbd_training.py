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
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TARGET_COLUMNS = [f"target_{index}" for index in range(1, 5)]
TRANSACTION_CATEGORIES = ("event_type", "event_subtype", "currency")
FOLD_PATTERN = re.compile(r"fold=(\d+)")


def _fold_from_path(member_name: str) -> int:
    match = FOLD_PATTERN.search(member_name)
    if match is None:
        raise ValueError(f"Could not find fold number in {member_name!r}")
    return int(match.group(1))


def _parquet_tables(archive_path: Path, required_path: str):
    """Yield parquet tables from one of the dataset's compressed tar files."""
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for index, member in enumerate(archive, start=1):
            if required_path.startswith("detail/trx/") and index % 1000 == 0:
                print(f"Scanned {index:,} members in the detail archive...", flush=True)
            if not member.isfile() or not member.name.endswith(".parquet"):
                continue
            if required_path not in member.name:
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            yield member.name, parquet.read_table(io.BytesIO(source.read())).to_pandas()


def load_training_targets(targets_archive: Path, customers_per_fold: int) -> pd.DataFrame:
    """Sample clients by fold and retain every reporting-month label row."""
    frames = []
    for member_name, frame in _parquet_tables(targets_archive, "targets/fold="):
        frame["fold"] = _fold_from_path(member_name)
        frames.append(frame)

    if not frames:
        raise FileNotFoundError("No target parquet files were found in the MBD archive.")

    targets = pd.concat(frames, ignore_index=True)
    targets["mon"] = pd.to_datetime(targets["mon"], errors="coerce")
    targets = targets.dropna(subset=["mon"])
    if customers_per_fold > 0:
        # Sample clients once, then retain all of their monthly observations.
        clients = targets[["fold", "client_id"]].drop_duplicates().copy()
        clients["_sample_key"] = pd.util.hash_pandas_object(
            clients["client_id"], index=False
        ).astype("uint64")
        selected_clients = (
            clients.sort_values(["fold", "_sample_key"])
            .groupby("fold", group_keys=False)
            .head(customers_per_fold)
            .drop(columns="_sample_key")
        )
        targets = targets.merge(selected_clients, on=["fold", "client_id"], how="inner")

    targets["report_month"] = targets["mon"].dt.to_period("M")
    return targets.reset_index(drop=True)


def _transaction_features(
    transactions: pd.DataFrame,
    targets: pd.DataFrame,
    top_category_codes: dict[str, list],
) -> pd.DataFrame:
    """Build a trailing 12-month activity summary for each reporting month."""
    if transactions.empty:
        return pd.DataFrame(columns=["fold", "client_id", "report_month"])

    events = transactions.copy()
    events["event_time"] = pd.to_datetime(
        events["event_time"], errors="coerce", utc=True
    ).dt.tz_convert(None)
    events["amount"] = pd.to_numeric(events["amount"], errors="coerce")
    events = events.dropna(subset=["event_time", "amount"])
    if events.empty:
        return pd.DataFrame(columns=["fold", "client_id", "report_month"])

    events["event_month"] = events["event_time"].dt.to_period("M")
    result_frames = []
    for fold, fold_targets in targets.groupby("fold", sort=False):
        fold_events = events[events["fold"] == fold]
        if fold_events.empty:
            continue

        for report_month in fold_targets["report_month"].drop_duplicates().sort_values():
            window = fold_events[
                fold_events["event_month"].between(report_month - 11, report_month)
            ]
            if window.empty:
                continue

            summary = window.groupby("client_id", sort=False).agg(
                transaction_count=("amount", "size"),
                amount_total=("amount", "sum"),
                amount_magnitude=("amount", lambda values: values.abs().sum()),
                amount_mean=("amount", "mean"),
                amount_std=("amount", "std"),
                amount_min=("amount", "min"),
                amount_max=("amount", "max"),
                active_days=("event_time", lambda values: values.dt.date.nunique()),
                active_months=("event_month", "nunique"),
            ).reset_index()
            if "event_type" in window:
                event_types = window.groupby("client_id")["event_type"].nunique()
                summary["distinct_event_types"] = summary["client_id"].map(event_types)

            for category, codes in top_category_codes.items():
                if category not in window or not codes:
                    continue
                typed = window[window[category].isin(codes)].copy()
                if typed.empty:
                    continue
                typed["amount_magnitude"] = typed["amount"].abs()
                by_code = typed.groupby(["client_id", category], sort=False).agg(
                    category_count=("amount", "size"),
                    category_amount_magnitude=("amount_magnitude", "sum"),
                )
                counts = by_code["category_count"].unstack(fill_value=0)
                magnitudes = by_code["category_amount_magnitude"].unstack(fill_value=0)
                counts = counts.reindex(columns=codes, fill_value=0)
                magnitudes = magnitudes.reindex(columns=codes, fill_value=0)
                counts.columns = [f"{category}_code_{code}_count" for code in counts.columns]
                magnitudes.columns = [
                    f"{category}_code_{code}_amount_magnitude"
                    for code in magnitudes.columns
                ]
                summary = summary.merge(counts, left_on="client_id", right_index=True, how="left")
                summary = summary.merge(
                    magnitudes, left_on="client_id", right_index=True, how="left"
                )
                category_columns = [*counts.columns, *magnitudes.columns]
                summary[category_columns] = summary[category_columns].fillna(0)

            monthly = window.groupby(["client_id", "event_month"], sort=False).agg(
                month_amount=("amount", "sum"),
                month_transactions=("amount", "size"),
            ).reset_index()
            monthly_summary = monthly.groupby("client_id").agg(
                monthly_amount_mean=("month_amount", "mean"),
                monthly_amount_std=("month_amount", "std"),
                monthly_transaction_mean=("month_transactions", "mean"),
                monthly_transaction_std=("month_transactions", "std"),
            ).reset_index()
            summary = summary.merge(monthly_summary, on="client_id", how="left")
            summary["fold"] = int(fold)
            summary["report_month"] = report_month
            result_frames.append(summary)

    if not result_frames:
        return pd.DataFrame(columns=["fold", "client_id", "report_month"])
    return pd.concat(result_frames, ignore_index=True)


def load_transaction_features(
    detail_archive: Path,
    targets: pd.DataFrame,
    test_fold: int,
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
            useful_columns = [
                column
                for column in ("client_id", "event_time", "amount", *TRANSACTION_CATEGORIES)
                if column in frame
            ]
            frame = frame[useful_columns].copy()
            frame["fold"] = fold
            transaction_frames[fold].append(frame)

    if not any(transaction_frames.values()):
        raise FileNotFoundError(
            "No MBD transaction parquet rows matched the sampled client folds."
        )

    training_parts = [
        frame
        for fold, frames in transaction_frames.items()
        if fold != test_fold
        for frame in frames
    ]
    top_category_codes = {}
    if training_parts:
        available_categories = [
            category for category in TRANSACTION_CATEGORIES
            if category in training_parts[0]
        ]
        for category in available_categories:
            category_counts = pd.Series(dtype="int64")
            for frame in training_parts:
                category_counts = category_counts.add(
                    frame[category].value_counts(),
                    fill_value=0,
                )
            top_category_codes[category] = category_counts.nlargest(12).index.tolist()

    feature_frames = []
    for fold, target_rows in targets.groupby("fold", sort=False):
        print(f"Building transaction features for fold {fold}...", flush=True)
        parts = transaction_frames[int(fold)]
        transactions = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        features = _transaction_features(
            transactions,
            target_rows,
            top_category_codes,
        )
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
    customers_per_fold: int = 10_000,
    test_fold: int = 4,
) -> dict:
    """Fit one classifier per anonymous product and report held-out metrics."""
    print("Loading monthly product outcomes...", flush=True)
    targets = load_training_targets(targets_archive, customers_per_fold)
    print(
        f"Loaded {len(targets):,} customer-month rows from "
        f"{targets['client_id'].nunique():,} sampled clients.",
        flush=True,
    )
    print("Reading the transaction archive and building rolling 12-month features...", flush=True)
    features = load_transaction_features(detail_archive, targets, test_fold)
    data = targets.merge(
        features,
        on=["client_id", "fold", "report_month"],
        how="left",
    )
    columns = [
        column
        for column in features.columns
        if column not in {"client_id", "fold", "report_month"}
    ]
    data[columns] = data[columns].replace([np.inf, -np.inf], np.nan)

    train = data[data["fold"] != test_fold]
    test = data[data["fold"] == test_fold]
    if train.empty or test.empty:
        raise ValueError(f"Fold {test_fold} must have both training and held-out clients.")

    output_dir.mkdir(parents=True, exist_ok=True)
    models = {}
    training_prevalence = {
        target: float(train[target].mean())
        for target in TARGET_COLUMNS
    }
    report = {
        "dataset": "MBD-mini",
        "training_clients": int(train["client_id"].nunique()),
        "held_out_clients": int(test["client_id"].nunique()),
        "training_rows": int(len(train)),
        "held_out_rows": int(len(test)),
        "held_out_fold": int(test_fold),
        "split_strategy": "unseen clients held out by the dataset's client fold; not a later-time split",
        "demo_profile_sampling": (
            "Illustrative accounts include held-out positive and negative examples; "
            "reported metrics use the complete held-out fold."
        ),
        "training_prevalence": training_prevalence,
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

        print(f"Training model for {target}...", flush=True)
        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),
        )
        model.fit(train[columns], train[target].astype(int))
        probabilities = model.predict_proba(test[columns])[:, 1]
        actual = test[target].astype(int).to_numpy()
        train_prevalence = float(train[target].mean())
        baseline_brier = brier_score_loss(actual, np.full(len(actual), train_prevalence))
        average_precision = float(average_precision_score(actual, probabilities))
        report["products"][target] = {
            "positive_rate": float(actual.mean()),
            "roc_auc": float(roc_auc_score(actual, probabilities))
            if len(np.unique(actual)) == 2
            else None,
            "average_precision": average_precision,
            "average_precision_baseline": float(actual.mean()),
            "average_precision_lift": (
                average_precision / float(actual.mean()) if actual.mean() > 0 else None
            ),
            "brier_score": float(brier_score_loss(actual, probabilities)),
            "prevalence_baseline_brier_score": float(baseline_brier),
            "brier_skill_over_prevalence": (
                float(1.0 - brier_score_loss(actual, probabilities) / baseline_brier)
                if baseline_brier > 0
                else None
            ),
            "precision_at_top_5_percent": _rank_precision(actual, probabilities),
            "top_5_percent_lift_over_random": (
                float(_rank_precision(actual, probabilities) / actual.mean())
                if actual.mean() > 0
                else None
            ),
        }
        models[target] = model

    # Choose illustrative examples from the held-out fold. Include true
    # next-month matches where a model score beats its baseline, plus negative
    # outcomes; the reported evaluation above still covers the complete fold.
    latest_test = (
        test.sort_values(["client_id", "mon"])
        .drop_duplicates("client_id", keep="last")
        .copy()
    )
    for target in TARGET_COLUMNS:
        latest_test[f"score_{target}"] = models[target].predict_proba(
            latest_test[columns]
        )[:, 1]

    predicted_top = latest_test[[f"score_{name}" for name in TARGET_COLUMNS]].idxmax(axis=1)
    selected_indices: list[int] = []
    for target in TARGET_COLUMNS:
        matching = latest_test[
            (predicted_top == f"score_{target}")
            & (latest_test[target].astype(int) == 1)
            & (latest_test[f"score_{target}"] > training_prevalence[target])
        ].sort_values(f"score_{target}", ascending=False)
        if not matching.empty:
            selected_indices.append(int(matching.index[0]))

    latest_test["_sample_key"] = pd.util.hash_pandas_object(
        latest_test["client_id"], index=False
    ).astype("uint64")
    below_baseline = pd.Series(True, index=latest_test.index)
    for target in TARGET_COLUMNS:
        below_baseline &= latest_test[f"score_{target}"] <= training_prevalence[target]
    holdback_examples = latest_test[below_baseline].sort_values("_sample_key")
    if not holdback_examples.empty:
        selected_indices.append(int(holdback_examples.index[0]))

    # Add strong-scoring non-purchase cases so the UI shows misses as well as
    # matches instead of presenting a hand-picked run of successes.
    no_purchase = latest_test[TARGET_COLUMNS].sum(axis=1) == 0
    latest_test["highest_score"] = latest_test[
        [f"score_{target}" for target in TARGET_COLUMNS]
    ].max(axis=1)
    negative_examples = latest_test[no_purchase].sort_values(
        "highest_score", ascending=False
    )
    for index in negative_examples.index:
        if len(selected_indices) >= 8:
            break
        if int(index) not in selected_indices:
            selected_indices.append(int(index))

    for index in latest_test.sort_values("_sample_key").index:
        if len(selected_indices) >= 12:
            break
        if int(index) not in selected_indices:
            selected_indices.append(int(index))

    demo_profiles = latest_test.loc[selected_indices].drop(
        columns=[
            "_sample_key",
            "highest_score",
            *[f"score_{target}" for target in TARGET_COLUMNS],
        ],
        errors="ignore",
    ).reset_index(drop=True)
    demo_profiles = demo_profiles[[*columns, *TARGET_COLUMNS]].copy()
    demo_profiles.insert(0, "profile_number", np.arange(1, len(demo_profiles) + 1))

    joblib.dump(
        {
            "models": models,
            "features": columns,
            "training_prevalence": training_prevalence,
            "demo_profiles": demo_profiles,
        },
        output_dir / "mbd_propensity.joblib",
    )
    with (output_dir / "mbd_evaluation.json").open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)
    print(f"Saved the model and evaluation report under {output_dir}.", flush=True)
    return report


def rank_profile(bundle: dict, profile: pd.Series | dict) -> list[dict]:
    """Rank the four anonymous product propensities for one held-out profile."""
    row = pd.DataFrame([{name: profile.get(name, np.nan) for name in bundle["features"]}])
    row = row.replace([np.inf, -np.inf], np.nan)
    results = []
    for target in TARGET_COLUMNS:
        score = float(bundle["models"][target].predict_proba(row)[0, 1])
        baseline = bundle["training_prevalence"][target]
        results.append(
            {
                "product": target,
                "propensity_score": score,
                "training_prevalence": baseline,
                "above_training_baseline": score > baseline,
                "observed_next_month_purchase": int(profile[target]),
            }
        )
    return sorted(results, key=lambda result: result["propensity_score"], reverse=True)


def explain_profile(bundle: dict, profile: pd.Series | dict, product: str) -> list[dict]:
    """Return the largest positive feature contributions for one product score."""
    model = bundle["models"][product]
    row = pd.DataFrame([{name: profile.get(name, np.nan) for name in bundle["features"]}])
    row = row.replace([np.inf, -np.inf], np.nan)
    imputer = model.named_steps["simpleimputer"]
    scaler = model.named_steps["standardscaler"]
    classifier = model.named_steps["logisticregression"]
    transformed = scaler.transform(imputer.transform(row))[0]
    names = imputer.get_feature_names_out(bundle["features"])
    contributions = transformed * classifier.coef_[0]
    ranked = np.argsort(contributions)[::-1]

    explanations = []
    for index in ranked:
        if contributions[index] <= 0 or len(explanations) == 3:
            continue
        explanations.append(
            {
                "signal": str(names[index]).replace("_", " "),
                "log_odds_contribution": float(contributions[index]),
            }
        )
    return explanations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/mbd-mini"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--customers-per-fold", type=int, default=10_000)
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
