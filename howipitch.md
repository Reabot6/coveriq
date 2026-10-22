# How I pitch CoverIQ with the live demo

I want the slides and the app to tell one story. I keep the slides light: one clear point and a visual. I explain the detail.

## Run of show

1. **Cover and team:** Introduce CoverIQ as an intelligence SDK for banks and fintechs. Insurance is the first example.
2. **Problem:** Explain the Track 9 challenge: use financial behaviour to make product matching more relevant. Start the Streamlit app and leave it ready.
3. **Solution:** Explain that CoverIQ turns activity into customer signals and compares those signals with a product offer. Different products need different fit signals.
4. **How it works:** Click **Run customer batch**. Return to the PowerPoint while it runs. Explain that the code generates example histories, calculates features, trains logistic regression on simulated outcomes, scores five contribution scenarios and routes the result for review or hold.
5. **See the result:** Switch to Chrome. Show the completed batch and Chinedu's account history. Point out that salary, outflows, surplus, balance and the proposed amount all matter. Keep the recap short.
6. **Scale and next three months:** Return to the deck. Explain the path from this one insurance example to a reusable integration, partner-configured product catalogue and product-specific matching modules. Finish with a measured shadow pilot using real outcomes.

## What the demo proves

It shows the full model workflow working on generated account histories. It does not show a live bank integration, validated performance on Zenith data, insurer pricing, or an offer being sent. The model estimates simulated cash-buffer persistence for contribution scenarios; it does not determine insurance need, eligibility or real affordability.

## Before presenting

- Keep the terminal at the project folder with this PowerShell command ready: **.\.venv\Scripts\python.exe -m streamlit run app.py**.
- If app startup may take time, start the server before presenting and leave the app open. Click **Run customer batch** only when you reach How it works.
- Rehearse once on the presentation laptop and time the batch. Do not run it twice during the pitch.
