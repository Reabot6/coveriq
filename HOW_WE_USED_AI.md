How We Used AI

This note explains where AI assisted CoverIQ and where product decisions, core implementation, and responsibility stayed with the team. It is written from the repository and the development conversation through commit .



We defined the Track 9 problem, supplied the research and examples, chose the product direction, challenged approaches that did not fit, reviewed the demo, and made every Git commit. AI was used as a research, coding, and documentation assistant: it helped explore options, draft implementation code, generate synthetic-data logic under our requirements, explain model behaviour, debug issues, and prepare documentation.

All core product logic, feature choices, routing rules, model specification, validation criteria, and final code ownership remained under human control. We treated AI as a capable junior pair-programmer that required clear context, tight review, and frequent correction—not as an autonomous author of the solution.

Where AI helped

Research and product thinking

AI was used to:
Explore the challenge statement and compare recommendation approaches
Reason about potential datasets and candidate feature signals
Turn research notes and facilitator feedback into possible product flows
Stress-test earlier directions (rule-based matching, catalogue ranking, affordability scenarios, MBD product-purchase pipeline) against time, evidence, and demo constraints

We brought the research, examples, and corrections. We chose the final direction and revised or removed ideas that did not fit the available evidence or the demo scope.

CoverIQ moved through several iterations. The earlier MBD product-propensity approach was deliberately removed after we observed that generative suggestions introduced unsupported claims and hallucinated product behaviour. We then focused on a narrower, manually researched cash-flow sustainability example that we could fully understand, defend, and explain. That pivot was a human decision driven by hard research and repeated validation
Synthetic data and mock product scenarios

AI helped draft the synthetic-data generator (synthetic_data.py) and its scenario logic from detailed product requirements we defined.

The generator creates repeatable fictional account histories:





36 historical months + 3 future outcome months



1,000 training profiles



12 demo accounts (including Chinedu)



Variation across income regularity, higher commitments, recent income changes, income disruption, and steadier saving patterns





We owned the requirements. We specified the number of months, the profile count, the variation dimensions, the fixed random seed, the buffer definition (25 % of typical essential outflows), and the decision that every record must remain fully synthetic. We supplied the scenario design and corrected intermediate drafts. AI generated the bulk of the generator code after receiving that context through iterative prompt engineering. The current catalogue example (My Smart Save) and the illustrative contribution amounts (₦5,000 / ₦15,000 / ₦30,000 / ₦60,000 / ₦120,000) were chosen by the team as comparison scenarios.

Code drafting, documentation, and debugging

AI assisted with:


Drafting README, MODEL.md, and supporting documentation

d




Responding to reported runtime and interface problems



Producing explanatory notes for the model and data flow

What stayed under human control





We set and repeatedly refined the problem being solved and the audience for the demo.



Product scope, pivots, and rejected approaches were chosen by us, including the final decision to narrow the build to one insurance cash-flow example and a small proof of concept i wrote all the core code while ai focused on generateing synhtetic data.



AI suggestions were mostly  research assistance only and giving me reference for code i got confused on  or explaining how they worked 

R



I led the product decisions and controlled the scope. I used AI as a research and coding assistant, including for drafting the synthetic-data generator and parts of the model and demo code after I supplied detailed requirements and context. I reviewed and iterated on every substantial change, wrote core logic  and I am responsible for explaining and validating what I present.

Commit history review



These were user-directed commits with AI-assisted implementation. Git cannot reliably identify line-by-line authorship. Core feature logic, product decisions, and final validation were human-controlled; AI primarily accelerated scaffolding, synthetic-data generation under our specifications, documentation, and routine refactors.







Commit



Diff scope and change



Contribution classification





11cbf9d



Initial CoverIQ app, transaction features, products, recommender, synthetic data and model training (10 files)



AI-assisted initial scaffolding based on the team's research brief and product direction





8b31011



Reworked README to reflect changed product direction



Human-led direction change; AI-assisted documentation edit





fbba663







6322cec



Added transaction activity demo module



AI-assisted implementation following the team's iteration request





66f5c98



Added transaction-history profile features (monthly cash flow, balance trend, salary change)



Human-selected signals; AI-assisted feature implementation





cd715cf



ff2ba73



Added consent-gated batch-matching scene and revised demo activity



Human-directed flow; AI-assisted UI and logic changes





8dd3083



Added evidence-based catalogue ranking



AI-assisted ranking implementation for the selected product behaviour





0e7fc14



Added synthetic offer-ranking model and decision layer; revised UI



Human chose to pursue an ML-based approach; AI-assisted model 





7d8e9e4



Large rework after research: removed earlier matching modules, major app/data/model refactor



Human-led research and pivot; AI-assisted implementation and cleanup





a3d23f0



Revised Streamlit flow and supporting features, data and recommendation code



Human-directed scope changes; AI-assisted implementation





4d0adf4



Added affordability scenarios and catalogue; revised cash-flow matching and synthetic data


 AI-assisted feature and model changes





bfe5324



Updated .gitignore



Human-requested repository cleanup; AI-assisted configuration edit





ef52ab5



Added MBD product-propensity training pipeline and dependency


AI-assisted pipeline with finding the model we use [research]





352101c



Shifted to monthly MBD outcomes, added MBD demo app and model docs; removed previous insurance pipeline



Human-directed experiment; AI-assisted scaffold . Later replaced by the me.





95e2554



Clarified model scores and demo outcomes; updated README and PowerPoint notes



Human feedback on clarity; AI-assisted documentation and app edits





13532c4



Replaced MBD demo with current synthetic cash-flow insurance-routing example; added cashflow_model.py




To inspect any original patch: git show <commit-hash>.
