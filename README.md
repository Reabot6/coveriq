# CoverIQ

## My notes on the idea

After the research and outreach I’ve done so far, I’m updating the build to focus on the full Track 9 challenge: help a bank work out which products may be relevant to each customer, when to show them, and when to leave the customer alone.

CoverIQ is the intelligence layer that does that matching. It uses customer-permitted information and financial behaviour to build a useful, changing picture of what a customer might need. It then checks the bank’s product catalogue and returns a next step, such as show a relevant product to review, wait for more evidence, or suppress the offer.

The customer ID is how the bank links permitted signals and past interactions to the right profile. It is an identifier, not permission by itself. The customer should still be able to choose whether to look at a product or continue with it.

The point is to make outreach more useful: fewer messages that do not fit, and clearer product suggestions when there is a good reason to show one. The bank can also learn from what happens after an offer is shown, like whether it was viewed, dismissed, or explored. We need to measure that; we have not proved business results yet.

## Why insurance is in the demo

Insurance is the example that makes the idea easy to see. The demo uses a Prudential Zenith Life savings product and a fictional customer. CoverIQ is not only an insurance recommender; the same matching approach could work with other products if we have the right catalogue information and fit checks for them.

## What I learned from the research

References

- [Amazon Personalize: Personalized Ranking v2](https://docs.aws.amazon.com/personalize/latest/dg/native-recipe-personalized-ranking-v2.html)
- [Amazon Personalize: Next Best Action](https://docs.aws.amazon.com/personalize/latest/dg/native-recipe-next-best-action.html)
- [Amazon Personalize: Recording real-time events](https://docs.aws.amazon.com/personalize/latest/dg/recording-events.html)

## How I see CoverIQ working

1. Link permitted customer signals to a bank-controlled customer ID.
2. Keep a useful profile of the customer’s interests, activity, and responses to previous offers.
3. Check the product catalogue and remove options that do not pass the relevant fit and contact rules.
4. Rank the remaining options and decide whether to show something, wait, or suppress outreach.
5. Return the suggestion and a plain explanation to a channel the bank already uses.
6. Record what happened so the bank can measure whether the suggestions were useful.

## What I want the next build to prove

The next version should show the broader intelligence layer more clearly: a customer profile linked by ID, a catalogue with several options, a reasoned ranking, timing and suppression decisions, and a record of what the customer did with the suggestion. Insurance can stay as the first example. The demo should make it obvious why CoverIQ chose one next step for Chinedu and why it held back on the others.
