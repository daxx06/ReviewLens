import csv
import io
import json
import os
import re
import urllib.error
import urllib.request

import streamlit as st

st.set_page_config(page_title="ReviewLens - Product Review Analysis", layout="wide")


# --- Module 1: Raw Review Input & Preprocessing ---
def scrub_pii(text: str) -> str:
    """Remove emails, phone numbers, and common order ID patterns before sending text to an LLM."""
    text = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[EMAIL]", text)
    text = re.sub(r"\b(?:\+?\d{1,3}[-.\s]?)?\d{10}\b", "[PHONE]", text)
    text = re.sub(r"\b(?:ORD|PNR|INV|ID)[-#:]?\s*[A-Z0-9-]{5,}\b", "[ORDER_ID]", text, flags=re.I)
    return text.strip()


def parse_reviews(raw_text: str) -> list[str]:
    """Parse pasted lines or CSV rows into a clean list of non-empty review strings."""
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    return [scrub_pii(line) for line in lines]


def load_sample_reviews() -> str:
    sample_path = os.path.join(os.path.dirname(__file__), "data", "sample_reviews.csv")
    if not os.path.exists(sample_path):
        return ""
    rows = []
    with open(sample_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rating = row.get("rating", "").strip()
            text = row.get("review_text", "").strip()
            if text:
                prefix = f"({rating} stars) " if rating else "(No rating) "
                rows.append(f"{prefix}{text}")
    return "\n".join(rows)


# --- Module 2 & 3: Chained AI Extractor & Recommendation Generator ---
def call_gemini(prompt: str, api_key: str, model: str = "gemini-1.5-pro") -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data["candidates"][0]["content"]["parts"][0]["text"]


def strip_json_fences(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def verify_grounding(themes_data: dict, source_reviews: list[str]) -> dict:
    """Module 4 Guardrail: check that every quoted evidence string exists in the input reviews."""
    joined_source = "\n".join(source_reviews).lower()
    for item in themes_data.get("themes", []):
        quote = item.get("evidence")
        if quote and quote.lower() not in joined_source:
            item["evidence_verified"] = False
        else:
            item["evidence_verified"] = True
    return themes_data


def run_demo_pipeline(reviews: list[str]) -> dict:
    """Offline fallback matching the chained workflow output for evaluator testing without an API key."""
    joined = "\n".join(reviews).lower()
    themes = []
    if "size" in joined or "tyt" in joined or "small" in joined:
        themes.append({"theme": "Sizing and Fit", "sentiment": "negative", "count": 5, "evidence": "size runs very small"})
    if "zip" in joined or "stitch" in joined or "button" in joined:
        themes.append({"theme": "Stitching and Hardware", "sentiment": "negative", "count": 4, "evidence": "Zippers broke on the first day of use"})
    if "box" in joined or "packag" in joined or "polybag" in joined or "delay" in joined:
        themes.append({"theme": "Delivery and Packaging", "sentiment": "negative", "count": 5, "evidence": "box was crushed"})
    if "fabric" in joined or "cotton" in joined or "kapda" in joined or "color" in joined:
        themes.append({"theme": "Fabric and Color", "sentiment": "mixed", "count": 6, "evidence": "Good cotton material for summer"})
    if not themes:
        themes.append({"theme": "General Feedback", "sentiment": "mixed", "count": len(reviews), "evidence": reviews[0][:60]})

    suggestions = [
        "Update the listing size chart with exact chest and waist measurements and advise buyers to order one size up.",
        "Switch to tamper-evident corrugated outer mailers to stop transit tears and crushed boxes.",
        "Audit the zipper and button stitching batch before dispatch to cut early defect returns."
    ]
    return verify_grounding({"themes": themes, "suggestions": suggestions}, reviews)


def run_chained_pipeline(reviews: list[str], api_key: str) -> dict:
    numbered = "\n".join(f"{i+1}. {r}" for i, r in enumerate(reviews))
    prompt_1 = (
        "You are an e-commerce quality analyst helping a small online seller.\n"
        "Extract 3 to 5 main themes from the reviews below. Reply with valid JSON only.\n"
        "Rules: every 'evidence' string must be an exact substring from the input reviews. "
        "Judge sentiment from the text even if the star rating contradicts it. "
        "Handle Hinglish or slang without changing the copied quote.\n"
        'Schema: {"themes":[{"theme":"","sentiment":"positive|negative|mixed","count":0,"evidence":""}]}\n\n'
        f"Reviews:\n{numbered}"
    )
    raw_themes = strip_json_fences(call_gemini(prompt_1, api_key))
    themes_obj = json.loads(raw_themes)

    prompt_2 = (
        "You are an operations advisor for a small online seller.\n"
        "Given the extracted themes below, write the top 3 specific, actionable improvements ranked by impact.\n"
        'Reply with valid JSON only: {"suggestions":["","",""]}\n'
        "Do not give generic advice; name the specific part, packaging step, or listing fix.\n\n"
        f"Themes:\n{json.dumps(themes_obj)}"
    )
    raw_sugg = strip_json_fences(call_gemini(prompt_2, api_key))
    sugg_obj = json.loads(raw_sugg)

    combined = {
        "themes": themes_obj.get("themes", []),
        "suggestions": sugg_obj.get("suggestions", []),
    }
    return verify_grounding(combined, reviews)


# --- Streamlit UI ---
st.title("ReviewLens")
st.caption("Chained prompt workflow for e-commerce product review analysis (Team PromptPilots)")

with st.sidebar:
    st.subheader("Settings")
    api_key = st.text_input("Gemini API Key (optional)", type="password", help="Leave blank to run in offline demo mode.")
    if st.button("Load 25 Sample Reviews"):
        st.session_state["review_input"] = load_sample_reviews()

if "review_input" not in st.session_state:
    st.session_state["review_input"] = load_sample_reviews()

raw_text = st.text_area("Paste customer reviews (one per line)", key="review_input", height=220)
uploaded = st.file_uploader("Or upload a CSV file with a 'review_text' column", type=["csv"])

if uploaded is not None:
    content = uploaded.read().decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(content))
    csv_reviews = [r.get("review_text", "").strip() for r in reader if r.get("review_text", "").strip()]
    if csv_reviews:
        raw_text = "\n".join(csv_reviews)
        st.info(f"Loaded {len(csv_reviews)} reviews from CSV.")

if st.button("Analyze Reviews", type="primary"):
    reviews = parse_reviews(raw_text)
    if len(reviews) < 5:
        st.error("Please enter at least 5 reviews to run theme extraction.")
    else:
        with st.spinner("Running chained prompt pipeline..."):
            if api_key.strip():
                try:
                    result = run_chained_pipeline(reviews, api_key.strip())
                except (urllib.error.URLError, json.JSONDecodeError, KeyError) as exc:
                    st.warning(f"API call failed ({exc}). Falling back to offline demo output.")
                    result = run_demo_pipeline(reviews)
            else:
                st.info("No API key entered; running offline demo mode.")
                result = run_demo_pipeline(reviews)

        # Module 4: Structured JSON Summary & Dashboard View
        col1, col2 = st.columns([3, 2])
        with col1:
            st.subheader("Extracted Themes & Sentiment")
            st.dataframe(result["themes"], use_container_width=True)

        with col2:
            st.subheader("Top 3 Actionable Fixes")
            for idx, fix in enumerate(result["suggestions"], start=1):
                st.write(f"{idx}. {fix}")

        st.subheader("JSON Output")
        st.json(result)
        st.download_button(
            "Download JSON Report",
            data=json.dumps(result, indent=2),
            file_name="reviewlens_output.json",
            mime="application/json",
        )
