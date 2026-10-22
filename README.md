# CoverIQ v1

CoverIQ is a customer-permitted recommendation layer. It uses financial behaviour to help decide whether a customer should see a product offer now, wait for more history, or have the offer held back. The demo uses one fictional customer, Chinedu, and Prudential Zenith Life products as the example catalogue.

## What the demo does

1. Chinedu opts in and selects the types of offers he wants to review.
2. CoverIQ analyzes 36 months of fictional account transactions.
3. A logistic regression model estimates whether positive monthly cash flow will continue in each of the next three months.
4. CoverIQ combines that estimate with the selected interests and a timing safeguard, then shows an offer for review, waits, or holds it back.

“Suggested for review” does not mean insurance approval or a verified affordability decision. The model is trained and evaluated on simulated histories. Its metrics describe the simulator, not Zenith customers. No bank connection is made and no message is sent.

## Project files

- `app.py` — Streamlit demo flow and result screen.
- `chinedu_data.py` — Chinedu’s fictional, transaction-level history.
- `features.py` — shared calculations used by training data and demo inference.
- `synthetic_data.py` — creates fictional transaction histories and labels from the following three months.
- `train_model.py` — trains and evaluates the cash-flow persistence model.
- `recommend.py` — applies consent, interest, timing, and model-score checks.
- `products.py` — small product catalogue for the demo.

## Run on Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

To see the model evaluation in the terminal:

```powershell
.\.venv\Scripts\python.exe train_model.py
```

## Model target

Each training example contains 36 months of generated transactions. The label is `1` only when net cash flow is positive in all three generated months after that history; otherwise it is `0`. Features are calculated only from the earlier 36 months. The train/test split is by fictional customer, and model metrics are compared with a majority-class baseline.

The simulator contains assumptions about income, spending, raises, and unexpected outflows. That makes this a working proof of concept, not validation for real-world customers. A bank deployment would need customer-permitted data and outcomes collected under the bank’s governance, then evaluation on unseen customers and time periods.
