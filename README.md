# ReviewLens

Prompt workflow for analyzing e-commerce product reviews (SCSE4041 Problem C14, Team PromptPilots).

## What it does

1. **Input and preprocessing:** Accepts pasted reviews or a CSV upload and strips emails, phone numbers, and order IDs.
2. **Theme and sentiment extraction:** Groups reviews into 3 to 5 themes, counts mentions, classifies sentiment from text rather than star ratings alone, and copies verbatim quotes.
3. **Actionable fixes:** Generates 3 specific seller improvements based on negative and mixed themes.
4. **Verification and export:** Checks that every quote appears in the input text and exports the result as JSON.

## Files

- `app.py` - Streamlit prototype (supports Gemini API key or offline demo mode)
- `prompts/` - Saved prompt versions `v1` through `v4` from our iteration log
- `data/sample_reviews.csv` - 25 synthetic reviews covering sizing, packaging, Hinglish, and sarcasm
- `docs/` - Milestone 2 PDF, Technical Report PDF, and 120-second reel script

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
