# CoverIQ — PowerPoint points

Working content for the pitch deck. This answers the sections in the facilitator's photo using the build we have now. It keeps the prototype's results separate from the wider product vision.

## 1. Title and team intro

**Project:** CoverIQ  
**Suggested tagline:** Product recommendations that fit customer behaviour.  
**One-line description:** A prototype that learns from transaction patterns to rank product offers and hold back weaker matches.

**Team intro:** Onimisi & divine 

## 2. Problem

**The problem:** Customers can be presented with several banking products without a clear signal about which one fits their behaviour or when it is relevant. Broad or repeated promotions can create noise and make useful offers easier to miss.

**Why it matters:** Customers spend time sorting through offers that may not suit them. Banks and product teams can spend campaign effort on customers unlikely to act. Track 9 asks for recommendations based on permitted customer information and financial behaviour, with fewer irrelevant or excessive messages.

**Who is affected:** Banking customers receiving product offers, and bank teams responsible for product discovery and customer communications.

**Simple story to tell:** A bank may have several products to introduce. If every customer sees the same set, the customer must work out what fits. CoverIQ's goal is to use permitted transaction patterns to rank likely matches and hold back weaker ones.

## 3. Solution: what we built in three days

We built a working machine-learning proof of concept using MBD-mini, a public anonymized banking dataset. It converts transaction activity into features, trains four classifiers from scratch, and ranks four anonymous product-purchase labels for an account.

The prototype can:

- summarize up to 12 months of transaction activity;
- rank four product labels using trained logistic-regression models;
- route the top label for review when its score clears a simple training-data baseline, or return `hold_back`;
- show which activity features raised the score;
- compare predictions with the recorded next-month outcome on held-out accounts.

**Core insight / “aha” moment:** The system can rank different product labels for different transaction profiles and hold back a weak match, instead of treating every account as an equal candidate for every product.

**What is not built yet:** 

## 4. How it works / tech overview

1. **Input data:** MBD-mini transaction records and next-month product-purchase outcomes.
2. **Feature building:** For each reporting month, calculate transaction counts, amount patterns, active days/months, monthly variation, and frequencies/magnitudes for common anonymized transaction codes. 
3. **Model training:** Train one scikit-learn logistic-regression pipeline for each of the four anonymous product labels.
4. **Evaluation:** Use four client folds for training and a separate client fold for evaluation. Training used 40,000 clients / 480,000 customer-month rows; the holdout used 10,000 different clients / 120,000 rows.
5. **Recommendation output:** Score and rank the four labels. Return the highest one for review if it clears that label's training purchase-rate baseline; otherwise hold it back.
6. **Demo UI:** A Streamlit app lets the presenter select held-out accounts, run the models, inspect the ranking and feature contributions, and compare the result with the dataset's recorded next-month outcome.

**Tools:** Python, pandas, PyArrow, scikit-learn, joblib, and Streamlit. There is no external AI API in this version.

**Feasibility choice:** We use transaction and product-outcome examples that already exist in MBD-mini rather than inventing synthetic customer outcomes. The held-out split is by client, so the evaluation uses people the models did not train on.

## 5. Demo / prototype

The live demo is the evidence. Select several held-out accounts and click **Run product matching**. For each one, show:

- number of transactions and active months in the 12-month summary;
- the model's top-ranked anonymous label and whether it is routed for review or held back;
- the full four-label ranking and the strongest positive model signals;
- the recorded next-month outcome as a post-score benchmark comparison.

The interface's **“20 offer slots”** means five selected accounts multiplied by four labels. It is a hypothetical count of account-product combinations if all four were offered to all five. No offer is actually sent. **“4 for review”** means four of the selected five accounts passed the prototype's score threshold; 
The example accounts are selected from the held-out fold to include both purchases and non-purchases, so the presenter can show matches and misses. The metrics below use all 120,000 customer-month rows from the 10,000 held-out clients, not only the examples shown.

## 6. Impact and market potential

**Who could use this:** Banks or other financial institutions that have permission to analyze customer activity and maintain a catalogue of products and observed customer responses.

**Potential impact:** If validated with a bank's own data, a ranking-and-holdback layer could help product teams prioritize relevant offers and reduce exposure to weak matches. This could improve product discovery for customers and campaign efficiency for institutions.

**What we can claim today:** The current prototype ranks next-month product uptake better than random selection on MBD-mini for all four anonymous labels. We have not measured conversion uplift, customer satisfaction, cost savings, or message reduction in a live bank.

**How it differs from a broad broadcast:** A broadcast gives every customer the same products. CoverIQ scores each account against each product label and can hold an account back when no score clears the prototype threshold. This comparison explains the design; it is not a measured comparison against Zenith's actual campaign system.

**Scale path:** The local demo would become a bank-integrated service only after the bank supplies approved transaction fields, a product catalogue, and observed offer outcomes. The model would then need retraining, evaluation, and controlled deployment on that bank's data.



## 7. Results to show

These results come from the current MBD-mini holdout. Metrics use 120,000 customer-month rows from 10,000 held-out corporate banking clients. Labels stay anonymous because MBD does not identify the products.

| Product label | ROC-AUC | Average precision vs. purchase-rate baseline | Top 5% lift |
| --- | ---: | ---: | ---: |
| target_1 | 0.751 | 1.86% vs. 0.47% | 4.9× |
| target_2 | 0.818 | 1.17% vs. 0.059% | 8.7× |
| target_3 | 0.772 | 1.19% vs. 0.342% | 4.4× |
| target_4 | 0.743 | 1.39% vs. 0.248% | 4.1× |

**How to explain the result:** The model does a better job ranking accounts with later recorded purchases near the top than a random ranking does. Purchases remain rare, and most high-ranked cases still did not purchase. The model's raw scores are not well calibrated, so present them as ranking scores, not percentages or affordability estimates.

## 8. Next steps: what we would do with three more months

1. **Get the right bank data:** Agree on permitted transaction fields, product catalogue IDs, and observed outcomes such as offer shown, opened, accepted, declined, or ignored.
2. **Map product relevance:** Work with product owners to define what a relevant match means for each real product. MBD's anonymous labels cannot supply this mapping.
3. **Build privacy and governance into integration:** Use bank-approved access, identifiers, consent/permission handling, retention limits, and audit logs before processing real customer records.
4. **Improve evaluation:** Test on later time periods and multiple customer groups; check calibration, false positives, fairness, and performance against the bank's existing campaign baseline.
5. **Run in shadow mode:** Generate recommendations without sending them. Have bank staff review whether rankings make sense.
6. **Pilot carefully:** Only after bank approval, compare a controlled pilot with the existing approach. Measure accepted offers, customer complaints/opt-outs, and whether irrelevant messages decrease.
7. **Monitor and retrain:** Track changes in transaction patterns and outcomes, and retrain only with approved data and a documented evaluation.

