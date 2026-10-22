# CoverIQ · PowerPoint points

Working notes for the pitch deck. Keep the distinction clear between what the prototype demonstrates and what a bank pilot would still need to prove.

## Title / team

**Project:** CoverIQ  
**Tagline:** Relevant offers, routed at a better time.
**One-line description:** A bank-side intelligence layer that reviews transaction patterns and routes relevant product offers for review.

**Team:** Onimisi & Divine

## Problem

Customers can be presented with several banking products without a clear signal about which offer fits their current circumstances. Broad promotions can make relevant information harder to notice, while a bank's product teams need a way to prioritize which offers to show and when.

Track 9 asks for product recommendations based on customer-permitted information and financial behaviour, while avoiding irrelevant or excessive marketing.

## Solution

CoverIQ turns bank transaction patterns into a structured customer profile, estimates whether a selected offer's illustrative contribution could fit the account's cash-flow pattern, then routes the offer for bank review or holds it back.

**Insurance example:** Prudential Zenith Life's My Smart Save, described publicly as a long-term savings plan with life cover. The demo uses this product to make the routing flow concrete. It does not reproduce the insurer's pricing or underwriting.

## What we built

- A transaction-history generator for 1,000 training profiles and a separate 12-account demo batch.
- 36 months of generated bank activity per account, including income, recurring outflows, savings transfers, other spending and balances.
- A logistic-regression model trained on generated histories and their later generated outcomes.
- A Streamlit demo that scores five illustrative contribution amounts, shows the bank review queue, and gives an aggregate partner view.

Chinedu's profile shows an income increase in the final three history months. CoverIQ checks the cash-flow pattern against each comparison amount rather than automatically treating a recent increase as either safe or unsafe.

## How it works

1. **Read history:** summarize 36 months of transaction patterns.
2. **Build features:** calculate income regularity, outflows, monthly remainder, balance trend, cash buffer and recent income changes.
3. **Estimate sustainability:** for each illustrative contribution, the model estimates whether the simulated balance would remain above a basic buffer for the next three months.
4. **Route:** use a stated demo threshold to queue the highest comparison amount that passes for bank review, or hold the account back.
5. **Summarize:** show account-level routing to the bank and aggregate counts by contribution scenario for the insurance partner.

The model is trained and checked on generated profiles. This demonstrates the end-to-end workflow; it is not validated performance on real Zenith customers. The sample contribution amounts are not product prices or quotes.

## Demo · the central moment

1. Click **Run customer batch**.
2. Show the number of accounts routed for review and the number held back.
3. Open Chinedu's account and point to the income increase, outflows and balance history.
4. Compare the model estimates across contribution scenarios; show the reason signals and the selected review action.
5. Show the partner summary, which contains group counts rather than the bank's account-level mapping.

## Impact / market potential

**Potential users:** Banks that distribute products through insurance or other product partners.

**Potential value to test:** Better prioritization could reduce low-fit outreach and help customers see offers that are more relevant to their observed financial patterns. These are hypotheses; the prototype has not measured campaign improvement or customer impact.

**Commercial paths to validate:** Bank software or integration fees, and/or a partner referral arrangement. The pilot would need to test who pays, how value is measured and how the bank and product partner divide responsibilities.

## Next steps

With more time, we would:

1. Get the bank's approved transaction schema and the insurer's current catalogue and contribution options.
2. Define the real outcome to predict and collect offer outcomes or future cash-flow outcomes under an approved pilot.
3. Retrain and evaluate on later real customer periods, including calibration and customer-group checks.
4. Run in shadow mode so bank staff can review recommendations before any offer is sent.
5. Pilot against the current approach and measure relevant offer views, acceptances, holdbacks, complaints and opt-outs.

## Claims to keep accurate

- The app trains a real classifier, but its training population and outcomes are generated for the demo.
- The demo predicts a simulated cash-buffer outcome for contribution scenarios; it does not decide insurance eligibility or prove affordability.
- My Smart Save is the catalogue example; the comparison amounts are not Prudential prices.
- No live bank system is connected, and no offers are sent.
- Conversion uplift, reduced unwanted outreach and revenue gains are future pilot measures, not results already achieved.
