# app.py
# AI Accessibility Toolkit — Premium UI Edition

import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from transformers import pipeline
import numpy as np
from PIL import Image
import re

# ----------------------------
# BEAUTIFUL PREMIUM CSS DESIGN
# ----------------------------
st.set_page_config(page_title="AI Accessibility Toolkit", page_icon="♿", layout="wide")

st.markdown("""
<style>

:root {
    --primary: #1E88E5;
    --accent: #00ACC1;
    --success: #43A047;
    --warning: #FB8C00;
    --danger: #E53935;
    --purple: #8E24AA;
    --soft-bg: #F4F7FB;
    --card-bg: rgba(255, 255, 255, 0.85);
}

/* Global background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #eef3ff 0%, #ffffff 100%) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff, #f6f9ff);
    border-right: 1px solid #e0e6f1;
    padding: 1.5rem 1rem;
}

/* Typography */
h1, h2, h3 {
    font-weight: 700 !important;
    color: #1a1f36 !important;
    letter-spacing: -0.5px;
}

/* Card design */
.card {
    background: var(--card-bg);
    backdrop-filter: blur(10px);
    padding: 24px;
    border-radius: 18px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    margin-bottom: 25px;
    transition: 0.25s ease;
}
.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 14px 35px rgba(0,0,0,0.12);
}

/* Buttons */
div.stButton > button {
    background: var(--primary);
    color: white;
    padding: 0.65rem 1.6rem;
    border-radius: 12px;
    font-size: 1.05rem;
    border: none;
    transition: 0.25s ease;
}
div.stButton > button:hover {
    background: var(--accent);
    transform: translateY(-2px);
}

/* Text areas */
textarea {
    border-radius: 14px !important;
    border: 1px solid #d0d5e0 !important;
}

/* Highlight marks */
mark {
    background: #ffe59e;
    padding: 3px 5px;
    border-radius: 4px;
}

/* Badges */
.badge {
    padding: 5px 10px;
    background: var(--accent);
    color: white;
    border-radius: 30px;
    font-size: 0.85rem;
    margin-right: 8px;
}

/* Tool color accents */
.tool-nlp h2 { color: var(--primary) !important; }
.tool-cv h2 { color: var(--success) !important; }
.tool-ethics h2 { color: var(--purple) !important; }

/* Fade-in Animation */
.main, .sidebar {
    animation: fadeIn 0.6s ease-out;
}
@keyframes fadeIn {
    0% { opacity: 0; transform: translateY(4px); }
    100% { opacity: 1; transform: translateY(0); }
}

/* Footer */
.app-footer {
    text-align: center;
    padding: 12px;
    color: #777;
    font-size: 0.9rem;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# ----------------------
# COMPONENT: Card Wrapper
# ----------------------
def card(body_fn):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    body_fn()
    st.markdown('</div>', unsafe_allow_html=True)


# ----------------------
# MODEL LOADERS (CACHE)
# ----------------------
@st.cache_resource
def load_image_model():
    st.info("Loading Image Model… (only on first run)")
    return mobilenet_v2.MobileNetV2(weights='imagenet')

@st.cache_resource
def load_summarizer_model():
    st.info("Loading Summarizer Model… (only on first run)")
    return pipeline("summarization", model="t5-small")


# --------------------------
# TEXT SUMMARIZER FUNCTION
# --------------------------
def summarize_text(text):
    summarizer = load_summarizer_model()
    summary = summarizer(text[:1024], max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']


# --------------------------
# IMAGE PROCESSING FUNCTION
# --------------------------
def get_image_predictions(img_file):
    model = load_image_model()
    img = Image.open(img_file).convert('RGB')
    img = img.resize((224, 224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    preds = model.predict(arr)
    return decode_predictions(preds, top=5)[0]


# ---------------------------------------
# INCLUSIVE LANGUAGE SCANNER DICTIONARY
# ---------------------------------------
PROBLEMATIC = {
    "crazy": "Use 'unexpected' or 'surprising'.",
    "insane": "Use 'unbelievable' or 'intense'.",
    "lame": "Use 'uninspiring' or 'weak'.",
    "dumb": "Use 'unhelpful' or 'unwise'.",
    "handicapped": "Use 'person with a disability'.",
    "the disabled": "Use 'persons with disabilities'.",
    "guys": "Use 'everyone', 'team', or 'folks'.",
    "chairman": "Use 'chairperson' or 'chair'.",
    "housewife": "Use 'homemaker' or 'caregiver'."
}

def scan_language(text):
    found = []
    for word, advice in PROBLEMATIC.items():
        if re.search(rf"\\b{word}\\b", text, re.IGNORECASE):
            found.append((word, advice))
    return found

def highlight_terms(text):
    for word in PROBLEMATIC.keys():
        text = re.sub(rf"\\b{word}\\b", f"<mark>{word}</mark>", text, flags=re.IGNORECASE)
    return text


# --------------------
# PAGE: ABOUT / HOME
# --------------------
def about_page():
    card(lambda: st.markdown("""
        <h2>Welcome to the AI Accessibility Toolkit</h2>
        <p>
        This toolkit provides inclusive, accessible AI tools designed to support cognitive accessibility,
        visual accessibility, and ethical language use.
        </p>
    """, unsafe_allow_html=True))


# ---------------------------
# PAGE: COGNITIVE SIMPLIFIER
# ---------------------------
def simplifier_page():
    st.markdown("<div class='tool-nlp'><h2>🧠 Cognitive Simplifier</h2></div>", unsafe_allow_html=True)

    card(lambda:
        st.info(
            "This tool helps make long or complex text easier to understand. It uses an AI summarizer to generate a "
            "shorter, clearer version ideal for accessibility, busy readers, and cognitive load reduction."
        )
    )

    text = st.text_area("Paste text to simplify:", height=250)

    if st.button("Simplify Text"):
        if not text.strip():
            st.warning("Please paste some text first.")
        else:
            with st.spinner("Summarizing…"):
                try:
                    summary = summarize_text(text)
                except Exception as e:
                    st.error(f"Summarizer error: {e}")
                    return

            card(lambda: st.markdown(f"### Simplified Summary\n{summary}"))


# ------------------------
# PAGE: IMAGE INSPECTOR
# ------------------------
def image_page():
    st.markdown("<div class='tool-cv'><h2>👁️ Image Inspector</h2></div>", unsafe_allow_html=True)

    card(lambda: st.info(
        "Upload an image and AI will identify objects, generate helpful ALT text, and detect "
        "potential accessibility concerns."
    ))

    img_file = st.file_uploader("Upload image…", type=["jpg", "jpeg", "png"])

    if img_file:
        st.image(img_file, use_column_width=True)

        with st.spinner("Analyzing image…"):
            preds = get_image_predictions(img_file)

        def display_preds():
            st.markdown("### AI Detected:")
            for _, label, prob in preds:
                st.markdown(f"<span class='badge'>{label.replace('_',' ')} – {prob*100:.1f}%</span>", unsafe_allow_html=True)

            primary = preds[0][1].replace("_", " ")
            st.markdown(f"### Suggested ALT Text:\nA photo containing **{primary}**.")

        card(display_preds)


# -----------------------------
# PAGE: INCLUSIVE LANGUAGE TOOL
# -----------------------------
def language_page():
    st.markdown("<div class='tool-ethics'><h2>🤝 Inclusive Language Scanner</h2></div>", unsafe_allow_html=True)

    card(lambda: st.info(
        "Paste text and AI will highlight non-inclusive or ableist terms and suggest better alternatives."
    ))

    text = st.text_area("Paste text to scan:", height=250)

    if st.button("Scan Text"):
        if not text.strip():
            st.warning("Please paste some text first.")
        else:
            results = scan_language(text)

            if not results:
                st.success("Great! No non-inclusive language detected.")
            else:
                for term, suggestion in results:
                    st.warning(f"**{term}** → {suggestion}")

                highlighted = highlight_terms(text)
                card(lambda: st.markdown(highlighted, unsafe_allow_html=True))


# ----------------
# MAIN NAVIGATION
# ----------------
def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Choose a Tool", ["Home", "Cognitive Simplifier", "Image Inspector", "Inclusive Language Scanner"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("Built by **Gideon Thuku** — Africa Ability Trust")

    if choice == "Home":
        about_page()
    elif choice == "Cognitive Simplifier":
        simplifier_page()
    elif choice == "Image Inspector":
        image_page()
    elif choice == "Inclusive Language Scanner":
        language_page()

    st.markdown("<div class='app-footer'>© Africa Ability Trust — Accessibility • Inclusion • Innovation</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
