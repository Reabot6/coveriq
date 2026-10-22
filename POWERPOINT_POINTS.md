# CoverIQ pitch points

## Positioning

CoverIQ is a reusable intelligence SDK for banks and fintechs. It turns financial activity into structured signals that help decide which product offer may fit a customer and when to surface it.

Insurance is the first example. Zenith and Prudential Zenith Life's My Smart Save make the example concrete; 

Keep the slides light: one bold takeaway and one visual. The explanation comes from the presenter, while each slide still gives the audience a clear idea to follow.

## Deck order

### 1. Cover and team

**On slide**

> **CoverIQ**
> Better product matches from financial activity

TEAM:REABOT6
names: Onimisi Adeiza ADEolu /Divine owolabi oluwafisayomi

**Visual:** Financial activity flowing into a matching decision.

**Presenter says:** “CoverIQ is an intelligence layer banks and fintechs can use to make product recommendations more relevant. We’re showing the idea through one insurance example.”

### 2. Problem

**On slide**

> **The data is there. But these's a gap in product matching**

**Visual:** One customer’s activity beside a catalogue of offers, with the matching decision in between.

**Presenter says:** “Track 9 asks how customer-permitted information and financial behaviour can help identify relevant products at the right time, while avoiding irrelevant or excessive marketing. The opportunity is to turn financial behavior into a decision the bank can use.”

**Live cue:** Open the terminal and start the Streamlit app. Leave the app ready; wait until Slide 4 to run the batch.

### 3. Solution

**On slide**

> **the  intelligence layer. Product-specific matching.**

**Visual:** Bank or fintech activity → CoverIQ → relevant offer for review.

**Presenter says:** “CoverIQ provides a shared way to process activity and route product candidates. The matching logic is tailored to the product: insurance needs different fit signals from other offers. The bank stays in control of what to surface and we handle the logic of making sure it reaches the right users.”

### 4. How it works 

**On slide**

> **History → signals → model score → review**

**Visual:** A simple four-step pipeline with small labels for income, outflows, surplus, balance and candidate contribution.

**Live cue:** Click **Run customer batch**, then return to the PowerPoint while it runs.

**Presenter says:** “That click starts the full prototype pipeline. It generates a set of example account histories, turns each history into features, trains a logistic-regression model on simulated outcomes, and scores the demo accounts against five contribution scenarios. The output is a comparison score and a review-or-hold route. It does not send an offer.”

Explain that the model looks across Chinedu’s history, not only at his recent salary change. Explain the score as a simulated estimate that the account keeps a basic cash buffer under a contribution scenario.

### 5. See the result

**On slide**

> **Now see what the model found.**

**Visual:** A small output-shaped illustration: account → contribution scenarios → review / hold.

**Live cue:** Switch to the app in Chrome. Show the completed batch, open Chinedu, then point to his transaction history and the scenario results. Keep this part short; the result should confirm the process you just explained.

**Presenter says:** “Here is the output in action. Chinedu’s salary increase is one part of the picture; the model also sees what comes in, what goes out, and what remains. Each contribution is tested against that history, so the proposed amount can change the routing.”
## impact / market potential then scale and next three months
### 6. Scale and next three months

**On slide**

> **From one example to a matching layer**

**Visual:** One shared SDK branching into distinct product-specific signal modules, then into a review queue.

Show a three-step timeline with short labels:

1. Month 1: integration and catalogue
2. Month 2: product-specific matching
3. Month 3: measured shadow pilot

**Presenter says:** “This MVP proves one narrow workflow with an insurance example. Over the next three months, the goal would be to define a secure integration with a bank or fintech, let partners configure their product catalogue, and build the right feature set and model for each product category. Insurance can use cash-flow fit signals; another offer may need different behaviour signals. We would test the recommendations in a shadow pilot against real outcomes before using them in customer communications.”

Describe relevance, fewer poor-fit referrals and customer response as outcomes to measure, not results already achieved. Commercial terms and the pilot scope would be agreed with partners.
 the rest after here is for me to read
## Live run order

1. Introduce CoverIQ and explain the Track 9 problem.
2. Start Streamlit and leave the app ready. If startup time is a risk, start the server before the pitch, but do not click **Run customer batch** yet.
3. Explain the solution. When you reach **How it works**, click **Run customer batch**.
4. Return to the deck. Explain what the click started, which signals are calculated, what the model scores, and what the output means while the batch runs.
5. On **See the result**, switch to Chrome and show the completed batch and Chinedu's account. Keep the recap brief.
6. Return to the deck for **Scale and next three months**.

If the batch finishes early, leave its results ready. If it is still processing, use the remaining model explanation; do not start a second run.

## Technical details to keep accurate

- The prototype uses generated histories: 1,000 training profiles and a separate 12-account demo batch.
- It trains and evaluates logistic regression on simulated histories and generated future outcomes, not on Zenith customer data.
- The five monthly contribution values are comparison scenarios, not insurer prices or quotes.
- The score estimates simulated cash-buffer persistence. It is not a measure of insurance need, policy eligibility or real-world affordability.
- There is no live bank or insurer connection, and the app sends no offers.
- The current MVP demonstrates one insurance example and one cash-flow sustainability question. Product-specific matching across a broad catalogue is the product direction, not a capability proven by this MVP.
- “Product-specific matching” means each product category needs relevant signals and its own validation. Do not say the current prototype already has a separate model for every product.

## Customer-protection wording

Call CoverIQ a product-fit and cash-flow review layer

Presenter sources:

- [Zecathon Track 9 challenge](https://beyondlimits.global/zecathonhackathon/)
- [CBN Consumer Protection Framework (2016)](https://www.cbn.gov.ng/out/2016/cfpd/consumer%20protection%20framework%20%28final%29.pdf)
- [NAICOM Revised Market Conduct and Business Practice Guidelines (2024)](https://naicom.gov.ng/wp-content/uploads/2025/08/JANUARY-2024-REVISED-MARKET-CONDUCT-AND-BUSINESS-PRACTICE-GUIDELINES-FOR-INSURANCE-REINSURANCE-COMPANIES-JAN.-2024.pdf)
