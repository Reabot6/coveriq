# CoverIQ

CoverIQ is an intelligent product-matching layer for banks. It turns permitted transaction patterns into ranked product candidates. This version trains on the open, anonymized MBD-mini dataset. The earlier prototype used assumed outcomes; this build uses the dataset's recorded transaction and product-purchase examples instead.

For a plain-language walkthrough of the demo screen, metrics, and current limits, see [DEMO_GUIDE.md](DEMO_GUIDE.md).
For the pitch sections in the facilitator's photo, see [POWERPOINT_POINTS.md](POWERPOINT_POINTS.md).

## Run the demo

The demo uses the public [MBD-mini dataset](https://huggingface.co/datasets/ai-lab/MBD-mini). Put its `targets.tar.gz` and `detail.tar.gz` files in `data/mbd-mini/`. The larger transaction archive must finish downloading before model training can start.

The dataset card lists MBD-mini under CC BY 4.0 and cites the [MBD paper](https://arxiv.org/abs/2409.17587).

On Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe mbd_training.py --data-dir data/mbd-mini --customers-per-fold 10000 --test-fold 4
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Training reads the transaction and target archives, builds rolling 12-month activity summaries, trains four logistic-regression models, and saves the model and evaluation report under `models/`. That folder and the dataset are ignored by Git.

See [MODEL.md](MODEL.md) for the feature calculations, evaluation setup, and how the files fit together.

The app lets you select held-out account profiles, run the models, and compare the top-ranked product with the recorded next-month outcome. It returns at most one candidate per account, and only when its model score is above that product's training purchase rate; otherwise the action is `hold_back`. The labels appear as `target_1` through `target_4` because the dataset does not identify the products by name.

## What the model learns

MBD-mini contains anonymized banking activity and four labels for whether a customer took a product in the month after a reporting month. CoverIQ makes a repeatable sample of 10,000 clients in each of the five client folds and retains their monthly rows. Four binary classifiers learn from transaction volume, amount patterns, active days and months, transaction-type diversity, and frequency/magnitude for common anonymized transaction, subtype, and currency codes. Client IDs are only used to join records and create client-level train/test folds; they are not model features.

Fold 4 is held out for evaluation. The report compares ROC-AUC, average precision, top-5% precision, and lift against the observed purchase rate. Scores are ranking signals for this dataset. They are not insurance eligibility or affordability decisions, and they are not validated predictions about Zenith customers.

## Current limits

- MBD-mini covers 12 reporting months, not three years.
- Its product names are anonymous, so this demo does not map them to Smart Save or another insurer's catalogue.
- The dataset describes corporate bank customers; it does not contain salary histories or insurance purchases.
- The public research data does not stand in for deployment consent; a bank integration must enforce its own permission checks.
- The demo trains from scratch on the public data. No live bank connection is included, and it does not send offers.
- COIL 200 describes a different insurance-purchase task and cannot be appended to MBD as if its records had the same features and labels.

Those limits keep the demo's claims matched to the data. A bank integration would need its own permitted transaction features, catalogue IDs, and observed offer outcomes before the model could be evaluated on that bank's customers.
