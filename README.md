# CoverIQ

CoverIQ is a bank-side product-matching proof of concept. This demo uses insurance as its example: it reviews transaction patterns, estimates whether a sample monthly contribution could be sustained, then routes a possible My Smart Save offer for bank review or holds it back.

## Run it

In PowerShell, from the project folder:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Click **Run customer batch** in the app. It generates the histories, trains the model, scores the customer batch and displays the routing results. The data and trained model are created in memory; no external dataset or API is required.

## What the demo does

1. Creates 36 months of transaction histories for 1,000 generated training profiles and a separate 12-account demo batch, including Chinedu.
2. Builds income, outflow, surplus, balance, and income-change features from each history.
3. Trains a logistic-regression classifier on generated profiles. Each training example pairs a customer history with an illustrative monthly contribution.
4. Uses the following three generated months to label whether the balance stayed above a basic buffer after that contribution.
5. Scores five illustrative contribution amounts, routes the largest amount above the demo review threshold, or holds the profile back.
6. Shows an individual account view for the bank, plus aggregate routing counts for an insurance partner.

The insurance catalogue example is Prudential Zenith Life's My Smart Save. The amounts are comparison inputs made for the demo, not product prices, premiums, or quotes. The bank review queue is illustrative; no offer is sent.

## What the model means

The model estimates whether a basic cash buffer is likely to remain for the next three months under a sample contribution scenario. It is trained and evaluated on generated histories and generated future outcomes. This proves the end-to-end model workflow; it does not establish accuracy for Zenith customers or real insurance affordability.

