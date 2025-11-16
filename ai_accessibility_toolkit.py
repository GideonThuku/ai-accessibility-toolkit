# app.py
# AI Accessibility Toolkit — upgraded UI + features
# Save as app.py and run: streamlit run app.py

import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from transformers import pipeline
import numpy as np
from PIL import Image
import re
import math

# -------------------------
# --- Custom Styling CSS ---
# -------------------------
GLOBAL_CSS = """
<style>
/* Page background */
[data-testid="stAppViewContainer"] > .main {
  background: linear-gradient(180deg, #f8fbff 0%, #ffffff 100%);
}

/* Container card look */
.card {
  background-color: white;
  padding: 20px;
  border-radius: 14px;
  box-shadow: 0 8px 24px rgba(17,24,39,0.06);
  margin-bottom: 18px;
}

/* Fancy header */
h1, h2, h3 {
  font-family: 'Inter', 'Segoe UI', sans-serif;
  color: #0b3d91;
}

/* Buttons */
div.stButton > button {
  border-radius: 12px;
  padding: 0.6rem 1.1rem;
  font-size: 0.95rem;
  background: linear-gradient(90deg,#4a90e2,#6b9afc);
  color: white;
  border: none;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg,#ffffff,#fafafa);
  border-right: 1px solid #eee;
  padding: 18px;
}

/* Marked highlights */
mark {
  background: #fff0b3;
  border-radius: 4px;
  padding: 0 4px;
}

/* Small badge */
.badge {
  display:inline-block;
  padding:4px 8px;
  border-radius:999px;
  background:#eef2ff;
  color:#0b3d91;
  font-size:0.85rem;
  margin-right:6px;
}

/* Footer */
.app-footer {
  text-align:center;
  color:#666;
  padding:10px;
  margin-top:24px;
  font-size:0.9rem;
}
</style>
"""

st.set_page_config(page_title="AI Accessibility Toolkit", page_icon="♿", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ----------------------------------------
# --- Helper: Card wrapper for pages  ---
# ----------------------------------------
def card_wrap(inner_fn):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    inner_fn()
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# --- AI Model loaders  ---
# -------------------------
@st.cache_resource
def load_image_model():
    st.info("Loading Image Model (first time only)...")
    model = mobilenet_v2.MobileNetV2(weights='imagenet')
    return model

def process_image(img_file):
    img = Image.open(img_file).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def get_image_predictions(img_file):
    model = load_image_model()
    processed_img = process_image(img_file)
    predictions = model.predict(processed_img)
    decoded = decode_predictions(predictions, top=6)[0]
    return decoded

@st.cache_resource
def load_summarizer_model():
    st.info("Loading Summarizer Model (first time only)...")
    summarizer = pipeline("summarization", model="t5-small")
    return summarizer

def summarize_text(text_input, max_in=1024, max_out=150, min_out=30):
    summarizer = load_summarizer_model()
    text_chunk = text_input[:max_in]
    summary = summarizer(text_chunk, max_length=max_out, min_length=min_out, do_sample=False)
    return summary[0]['summary_text']

# --------------------------------------------
# --- Inclusive language detection utilities ---
# --------------------------------------------
# Improved, grouped dictionary of problematic terms
PROBLEMATIC_TERMS = {
    "ableist": {
        "crazy": "Use 'surprising', 'unexpected', or 'remarkable' instead (avoid stigmatizing mental health).",
        "insane": "Use 'unbelievable' or 'chaotic' instead (avoid stigmatizing mental health).",
        "lame": "Use 'disappointing' or 'underwhelming' instead (avoid ableist language).",
        "dumb": "Use 'unhelpful' or 'not effective' instead.",
        "handicapped": "Use 'person with a disability' or 'people with disabilities'.",
        "the disabled": "Use 'people with disabilities' or 'persons with disabilities'.",
        "crippled": "Avoid — use 'person with a mobility impairment' if necessary."
    },
    "gendered": {
        "guys": "Use 'everyone', 'team', or 'folks' for mixed-gender groups.",
        "chairman": "Use 'chair', 'chairperson', or 'chair of the board'.",
        "housewife": "Use 'homemaker' or 'caregiver' depending on context.",
        "manned": "Use 'staffed' or 'operated'."
    },
    "race_ethnicity": {
        "illegal immigrant": "Use 'undocumented person' or 'migrant without legal status'.",
        "oriental" : "Use 'Asian' (note: 'oriental' is outdated/offensive when describing people)."
    },
    "demeaning": {
        "slave": "Avoid metaphorical use — use 'very hard work' or 'oppressive conditions' instead.",
        "retarded": "Do not use — use 'person with an intellectual disability' or 'person with a developmental disability'."
    }
}

# Flatten mapping for quick search
FLAT_TERM_MAP = {}
for group, terms in PROBLEMATIC_TERMS.items():
    for k, v in terms.items():
        FLAT_TERM_MAP[k] = v

# regex-based scanner
def scan_inclusive_language(text):
    found = []
    # look for whole words, case-insensitive
    for term, suggestion in FLAT_TERM_MAP.items():
        if re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE):
            found.append((term, suggestion))
    return found

# Highlight terms in text (returns HTML)
def highlight_text(text):
    def repl(match):
        word = match.group(0)
        return f"<mark>{word}</mark>"
    # Build pattern from all keys
    if not FLAT_TERM_MAP:
        return text
    pattern = r'\b(' + '|'.join(re.escape(k) for k in sorted(FLAT_TERM_MAP.keys(), key=lambda x: -len(x))) + r')\b'
    highlighted = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return highlighted

# ---------------------------------------
# --- Readability: ARI (Automated Readability Index)
# ---------------------------------------
def automated_readability_index(text):
    # ARI = 4.71*(characters/words) + 0.5*(words/sentences) - 21.43
    if not text or text.strip() == "":
        return None
    characters = len(re.sub(r'\s', '', text))
    words = len(re.findall(r'\w+', text))
    sentences = max(1, len(re.findall(r'[.!?]+', text)))
    try:
        ari = 4.71 * (characters / max(1, words)) + 0.5 * (words / max(1, sentences)) - 21.43
        ari = round(ari, 2)
    except Exception:
        ari = None
    return ari

# --------------------------------
# --- Page: About / Home Page ---
# --------------------------------
def run_about_page():
    def inner():
        st.markdown("## Welcome to the AI Accessibility Toolkit 🚀")
        st.write("Final student project — collection of tools to support inclusive design and accessibility.")
        st.markdown("""
        **Tools included**
        - 🧠 Cognitive Simplifier — summarizer & readability assist (NLP)
        - 👁️ Image Inspector — image labels, ALT text helper & accessibility risk (CV)
        - 🤝 Inclusive Language Scanner — highlights and suggestions (Ethics)
        """)
        st.markdown("---")
        st.info("Tip: use the sidebar to switch between tools. The first run may download ML models.")
    card_wrap(inner)

# --------------------------------
# --- Page: Cognitive Simplifier ---
# --------------------------------
def run_simplifier():
    def inner():
        st.markdown("## 🧠 Cognitive Simplifier")
        st.caption("UN Goal 4: Quality Education — reduces cognitive load by producing concise summaries.")
        st.info("Uses a pretrained summarization model. Try pasting an article or long email.")

        text_input = st.text_area("Paste text to simplify", height=300, placeholder="Paste a long article, email, or document here...")
        col1, col2 = st.columns([1, 1])
        with col1:
            bullets = st.checkbox("Output as bullet points", value=False, key="bullets_checkbox")
        with col2:
            show_readability = st.checkbox("Show readability (ARI)", value=True, key="ari_checkbox")

        if st.button("Simplify Text", key="simplify_btn"):
            if not text_input or text_input.strip() == "":
                st.warning("Please paste some text to simplify.")
                return
            with st.spinner("Generating summary..."):
                try:
                    summary_text = summarize_text(text_input)
                except Exception as e:
                    st.error(f"Summarizer error: {e}")
                    return

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### ✨ AI-Generated Summary")
            if bullets:
                # simple split into sentences and list
                sentences = re.split(r'(?<=[.!?])\s+', summary_text.strip())
                for s in sentences:
                    s_clean = s.strip()
                    if s_clean:
                        st.markdown(f"- {s_clean}")
            else:
                st.write(summary_text)

            if show_readability:
                ari = automated_readability_index(text_input)
                if ari is not None:
                    st.markdown(f"**Readability (ARI):** {ari} (approx. U.S. grade level)")
                else:
                    st.markdown("**Readability (ARI):** N/A")

            # quick keyword highlight
            keywords = sorted(set(re.findall(r'\b\w{5,}\b', text_input.lower())), key=lambda x:-len(x))[:10]
            if keywords:
                st.markdown("**Highlighted keywords:**")
                st.write(", ".join(keywords[:10]))
            st.markdown('</div>', unsafe_allow_html=True)
    card_wrap(inner)

# -------------------------------
# --- Page: Image Inspector  ---
# -------------------------------
def run_image_inspector():
    def inner():
        st.markdown("## 👁️ Image Inspector")
        st.caption("UN Goal 10: Reduced Inequalities — helps low-vision users by generating descriptions and accessibility cues.")
        st.info("Upload a photo. The MobileNetV2 model will propose labels and an ALT-text suggestion. Note: ImageNet training has biases.")

        uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"], key="image_uploader")
        if uploaded_file is None:
            st.info("Try uploading a photo of a scene or object (e.g., a staircase, a crowded street).")
            return

        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Inspecting image with MobileNetV2..."):
            try:
                predictions = get_image_predictions(uploaded_file)
            except Exception as e:
                st.error(f"Image model error: {e}")
                return

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### What the AI sees")
        # Display predictions as badges with confidence
        for (obj_id, label, prob) in predictions:
            label_clean = label.replace('_', ' ')
            st.markdown(f"<span class='badge'>{label_clean} ({prob*100:.1f}%)</span>", unsafe_allow_html=True)

        # Create a short alt-text suggestion from top labels
        top_labels = [label.replace('_', ' ') for (_, label, _) in predictions[:3]]
        alt_text = f"A photo depicting {', '.join(top_labels)}."
        st.markdown("**Suggested ALT text:**")
        st.info(alt_text)

        # Accessibility risk heuristics (simple rules)
        risk_issues = []
        low_vision_issues = []
        for (_, label, prob) in predictions:
            label_l = label.lower()
            if any(x in label_l for x in ['stair', 'stairs', 'step', 'escalator']):
                risk_issues.append("Detected stairs/steps — barrier for wheelchair users.")
            if any(x in label_l for x in ['wheelbarrow','wheelchair','cane','white cane']):
                low_vision_issues.append("Detected mobility/assistive device — consider inclusive access features.")
            if 'crowd' in label_l or 'person' in label_l and prob > 0.6:
                risk_issues.append("Crowded scene detected — may be a challenge for visually impaired people in navigation.")

        if risk_issues or low_vision_issues:
            st.markdown("### Accessibility Concerns")
            for r in risk_issues:
                st.warning(r)
            for r in low_vision_issues:
                st.info(r)
        else:
            st.success("No immediate accessibility hazards detected by quick heuristics — please review manually for real deployments.")

        # Accessibility score (0-100) simple heuristic
        score = 100
        if risk_issues:
            score -= 30
        if low_vision_issues:
            score -= 20
        # scale to 0-100
        score = max(0, min(100, score))
        st.markdown(f"**Accessibility risk score:** {score}/100")

        st.markdown('</div>', unsafe_allow_html=True)
    card_wrap(inner)

# -------------------------------------
# --- Page: Inclusive Language Scan ---
# -------------------------------------
def run_language_scanner():
    def inner():
        st.markdown("## 🤝 Inclusive Language Scanner")
        st.caption("UN Goal 10 & Goal 5 — detects non-inclusive, ableist, or biased language and suggests alternatives.")
        st.info("Paste text (job ad, email, blog, policy). The scanner highlights problematic terms and provides suggestions.")

        text_input = st.text_area("Paste text to scan", height=300, placeholder="Paste a job description, email, or blog post here...", key="scanner_area")
        col1, col2 = st.columns([1,1])
        with col1:
            show_highlight = st.checkbox("Highlight found terms inline", value=True, key="highlight_checkbox")
        with col2:
            auto_fix = st.checkbox("Show suggested replacements inline", value=False, key="autofix_checkbox")

        if st.button("Scan Text", key="scan_btn"):
            if not text_input or text_input.strip() == "":
                st.warning("Please paste some text to scan.")
                return

            with st.spinner("Scanning text..."):
                results = scan_inclusive_language(text_input)

                if not results:
                    st.success("Looks good — no flagged terms found by the scanner.")
                else:
                    st.error(f"The scanner found {len(results)} potential issue(s):")
                    # display detailed findings
                    for (term, suggestion) in results:
                        st.markdown(f"**Term:** \"{term}\" — **Suggestion:** {suggestion}")

                # Inline highlighting or suggested-replacements view
                if show_highlight:
                    highlighted = highlight_text(text_input)
                    st.markdown("**Highlighted Text:**")
                    st.markdown(f"<div class='card'>{highlighted}</div>", unsafe_allow_html=True)

                if auto_fix and results:
                    fixed_text = text_input
                    # Replace terms (case-insensitive) with suggested neutral phrase (take first sentence)
                    for term, suggestion in results:
                        # pick a short replacement from the suggestion text (take text before ';' or '.')
                        repl = suggestion.split(';')[0].split('.')[0].strip()
                        if repl == '':
                            repl = '[suggested alternative]'
                        fixed_text = re.sub(r'\b' + re.escape(term) + r'\b', repl, fixed_text, flags=re.IGNORECASE)
                    st.markdown("**Auto-suggested replacement (preview):**")
                    st.markdown(f"<div class='card'>{fixed_text}</div>", unsafe_allow_html=True)

    card_wrap(inner)

# ---------------------
# --- Main app loop ---
# ---------------------
def main():
    st.sidebar.title("AI Toolkit Navigation")
    st.sidebar.write("Choose a tool below:")
    pages = {
        "🏠 About": run_about_page,
        "🧠 Cognitive Simplifier": run_simplifier,
        "👁️ Image Inspector": run_image_inspector,
        "🤝 Inclusive Language Scanner": run_language_scanner
    }
    choice = st.sidebar.radio("Select a page", list(pages.keys()))
    st.sidebar.markdown("---")
    st.sidebar.markdown("Built for the **AI for Software Engineering** course — Africa Ability Trust student project")

    # Run chosen page
    pages[choice]()

    # Footer
    st.markdown("<div class='app-footer'>Built with ❤️ • AI Accessibility Toolkit • Africa Ability Trust</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
