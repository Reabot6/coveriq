# MBD product model

## Data and prediction target

CoverIQ trains on the public MBD-mini dataset. Each target row describes one customer and reporting month. The four `target_1` to `target_4` fields record whether the customer took each anonymous product in the following month. The product names are not included, so the demo keeps the IDs anonymous.

This MVP reads the transaction stream only. It leaves the dialogue embeddings and geolocation stream out of the model.

The code samples 10,000 customers from each of the five client folds and retains all 12 monthly rows for each sampled customer. This gives up to 600,000 customer-month examples. A customer's ID is used to join transactions and keep all of that customer's rows in one fold; it is never passed to a model as a feature.

## Transaction features

For a reporting month `t`, the model only sees transactions from `t-11` through the end of `t`. The target describes the next month, so its outcome is outside that feature window.

The feature builder calculates:

- transaction count, amount total, absolute amount magnitude, mean, standard deviation, minimum, and maximum;
- active days and active months;
- the number of distinct transaction types;
- average and standard deviation of monthly amount totals and transaction counts across active months;
- for the 12 most common event-type, subtype, and currency codes found in training folds, the transaction count and total absolute amount.

These event-type codes are anonymized categories. CoverIQ uses them as codes and does not assign meanings such as salary, rent, or shopping.

## Training and evaluation

One logistic-regression classifier is fitted for each product label. Each pipeline fills missing values with the training median, adds missing-value indicators, standardizes features, and fits the classifier. The dataset's fold 4 is kept aside for evaluation; folds 0 to 3 train the models. This tests performance on unseen clients within the dataset period; it is not a later-time test of performance drift.

For product `k`, the model calculates `p_k = 1 / (1 + exp(-(b_k + sum(w_kj * z_j))))`, where `z_j` is a standardized transaction feature. The demo selects the highest-scored product only if `p_k` is above that product's positive rate in the training folds. Otherwise it returns `hold_back`.

The report includes ROC-AUC, average precision, Brier score, precision among the highest-scored 5% of examples, and lift over the held-out purchase rate. Average precision is compared with the purchase-rate baseline. These measures help show whether transaction patterns rank likely next-month purchases above a simple baseline. They do not establish insurance eligibility or prove performance on Zenith customers.

The demo's routing policy selects at most one product per account: the highest-ranked product whose model score exceeds that product's purchase rate in the training folds. If none clears its own baseline, the output is `hold_back`. This is a simple review threshold for the prototype, not a calibrated decision boundary.

The Streamlit demo takes selected held-out feature rows and runs the saved classifiers at runtime. Example accounts are chosen to include both recorded purchases and non-purchases, while the reported evaluation metrics use the full held-out fold. It shows the top product label, the complete four-label ranking, the recorded next-month outcome for comparison, and the largest positive linear-model feature contributions for the top-ranked product. The outcome is available only as an offline benchmark; it is not an input to scoring. Contributions explain the model score; they are not causal explanations. No real customer identity is displayed and no offer is sent.

The model scores are useful for ranking in this dataset, but the reported Brier scores show little or no improvement over a purchase-rate-only estimate. Treat the scores as ranking signals, not well-calibrated probabilities or measures of affordability.

## Source

MBD-mini is published by the AI Lab under CC BY 4.0. Dataset card: <https://huggingface.co/datasets/ai-lab/MBD-mini>. Paper: <https://arxiv.org/abs/2409.17587>.
