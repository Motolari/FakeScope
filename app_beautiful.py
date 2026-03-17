import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import warnings
warnings.filterwarnings('ignore')
# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FakeScope — News Intelligence",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

:root {
    --bg:        #0a0a0f;
    --surface:   #12121a;
    --border:    #1e1e2e;
    --accent:    #e8ff45;
    --text:      #f0f0f5;
    --muted:     #6b6b80;
    --real:      #3ddc97;
    --fake:      #ff5f40;
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
header[data-testid="stHeader"], footer { display: none !important; }
.block-container { padding: 2rem 2.5rem !important; max-width: 1100px !important; }

[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    font-family: 'IBM Plex Sans', sans-serif !important; font-size: 0.9rem !important;
    font-weight: 400 !important; color: var(--muted) !important; padding: 0.3rem 0 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { color: var(--text) !important; }

.hero-title { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 3.2rem; letter-spacing: -0.03em; line-height: 1; color: var(--text); }
.hero-title span { color: var(--accent); }
.hero-sub { font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; color: var(--muted); letter-spacing: 0.12em; text-transform: uppercase; margin-top: 0.4rem; }

.page-header { border-bottom: 1px solid var(--border); padding-bottom: 1.25rem; margin-bottom: 2rem; }
.page-title { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: var(--text); letter-spacing: -0.02em; margin: 0; }
.page-title span { color: var(--accent); }
.page-subtitle { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: var(--muted); margin-top: 0.3rem; letter-spacing: 0.1em; text-transform: uppercase; }

.section-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.5rem; }

.card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.4rem 1.6rem; margin-bottom: 1rem; }
.card-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.4rem; }

.verdict-real { display: inline-block; background: #0d2e1e; border: 2px solid var(--real); color: var(--real); font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 800; padding: 0.5rem 1.4rem; border-radius: 8px; letter-spacing: 0.05em; }
.verdict-fake { display: inline-block; background: #2e0d0d; border: 2px solid var(--fake); color: var(--fake); font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 800; padding: 0.5rem 1.4rem; border-radius: 8px; letter-spacing: 0.05em; }

.stat-box { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1.1rem 1.3rem; text-align: center; }
.stat-value { font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 700; color: var(--accent); line-height: 1.1; }
.stat-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.25rem; }

.divider { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }

.conf-bar-wrap { background: #1e1e2e; border-radius: 4px; height: 7px; width: 100%; margin-top: 0.6rem; overflow: hidden; }
.conf-bar-fill { height: 7px; border-radius: 4px; transition: width 0.4s ease; }

.tag-real { display: inline-block; background: #0d2e1e; border: 1px solid #3ddc97; color: #3ddc97; font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; padding: 0.18rem 0.55rem; border-radius: 4px; margin: 2px; }
.tag-fake { display: inline-block; background: #2e0d0d; border: 1px solid #ff5f40; color: #ff5f40; font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; padding: 0.18rem 0.55rem; border-radius: 4px; margin: 2px; }

.stTextArea textarea { background: var(--surface) !important; color: var(--text) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; font-family: 'IBM Plex Sans', sans-serif !important; font-size: 0.95rem !important; }
.stTextArea textarea:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 2px rgba(232,255,69,0.15) !important; }
.stButton > button { background: var(--accent) !important; color: #0a0a0f !important; border: none !important; border-radius: 8px !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; font-size: 0.95rem !important; padding: 0.55rem 1.6rem !important; letter-spacing: 0.04em !important; transition: opacity 0.15s !important; }
.stButton > button:hover { opacity: 0.85 !important; }
.stSelectbox div[data-baseweb="select"] > div, .stFileUploader section { background: var(--surface) !important; border-color: var(--border) !important; color: var(--text) !important; border-radius: 8px !important; }
[data-testid="stMetricValue"] { font-family: 'Syne', sans-serif !important; font-size: 1.8rem !important; font-weight: 700 !important; color: var(--accent) !important; }
[data-testid="stMetricLabel"] { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.68rem !important; color: var(--muted) !important; text-transform: uppercase !important; letter-spacing: 0.1em !important; }

.stTabs [data-baseweb="tab-list"] { background: var(--surface) !important; border-radius: 10px !important; border: 1px solid var(--border) !important; padding: 4px !important; gap: 2px !important; }
.stTabs [data-baseweb="tab"] { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.78rem !important; color: var(--muted) !important; border-radius: 6px !important; letter-spacing: 0.05em !important; }
.stTabs [aria-selected="true"] { background: var(--accent) !important; color: #0a0a0f !important; }

@media (max-width: 768px) {
    .block-container { padding: 1rem 0.75rem !important; }
    .hero-title { font-size: 2rem !important; }
    .page-title { font-size: 1.5rem !important; }
    .verdict-real, .verdict-fake { font-size: 1.1rem !important; padding: 0.4rem 1rem !important; }
    .stButton > button { width: 100% !important; }
    .stat-value { font-size: 1.25rem !important; }
    .stTabs [data-baseweb="tab"] { font-size: 0.7rem !important; }
    [data-testid="stSidebar"] { display: none !important; }
}
.mobile-nav { display: none; }
@media (max-width: 768px) { .mobile-nav { display: block; margin-bottom: 1rem; } }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# NLTK SETUP
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def setup_nltk():
    for pkg in ['stopwords', 'wordnet', 'punkt', 'punkt_tab']:
        try:
            nltk.download(pkg, quiet=True)
        except:
            pass
    return set(stopwords.words('english')), WordNetLemmatizer()

stop_words, lemmatizer = setup_nltk()


# ─────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────
def clean_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    try:
        tokens = word_tokenize(text)
    except:
        tokens = text.split()
    cleaned = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(cleaned)


# ─────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    if os.path.exists('fake_news_model.joblib') and os.path.exists('tfidf_vectorizer.joblib'):
        return joblib.load('fake_news_model.joblib'), joblib.load('tfidf_vectorizer.joblib'), True
    return None, None, False

model, vectorizer, model_loaded = load_model()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='margin-bottom:1.8rem;'>
        <div style='font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;color:#f0f0f5;'>
            FAKE<span style='color:#e8ff45;'>SCOPE</span>
        </div>
        <div style='font-family:"IBM Plex Mono",monospace;font-size:0.62rem;color:#6b6b80;letter-spacing:0.12em;text-transform:uppercase;'>
            News Intelligence Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card-label">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        "",
        ["🔍  Predict", "📖  How It Works", "🕓  History"],
        label_visibility="collapsed"
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="card-label" style="margin-bottom:0.6rem;">About</div>', unsafe_allow_html=True)
    st.caption("Built with a Passive Aggressive Classifier + TF-IDF pipeline. Trained on balanced global & Nigerian news data.")


# ─────────────────────────────────────────────
# MOBILE NAV
# ─────────────────────────────────────────────
st.markdown("""
<style>
.mobile-nav { display: none; }
@media (max-width: 768px) {
    .mobile-nav { display: block; margin-bottom: 1rem; }
    [data-testid="stSidebar"] { display: none !important; }
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="mobile-nav">', unsafe_allow_html=True)
mobile_page = st.selectbox(
    "Navigate",
    ["🔍  Predict", "📖  How It Works", "🕓  History"],
    label_visibility="collapsed",
    key="mobile_nav"
)
st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.get("mobile_nav") and st.session_state.mobile_nav != st.session_state.get("last_sidebar_page"):
    page = mobile_page
st.session_state.last_sidebar_page = page


# ─────────────────────────────────────────────
# GLOBAL TITLE
# ─────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:2rem;padding-bottom:1.5rem;border-bottom:1px solid #1e1e2e;'>
    <div style='font-family:Syne,sans-serif;font-weight:800;font-size:3.2rem;letter-spacing:-0.03em;line-height:1;color:#f0f0f5;'>
        FAKE<span style='color:#e8ff45;'>SCOPE</span>
    </div>
    <div style='font-family:"IBM Plex Mono",monospace;font-size:0.72rem;color:#6b6b80;letter-spacing:0.15em;text-transform:uppercase;margin-top:0.4rem;'>
        News Intelligence Platform
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: PREDICT
# ─────────────────────────────────────────────
if "🔍  Predict" in page:

    st.markdown("""
    <div class='page-header'>
        <div class='page-title'>Article Verification</div>
        <div class='page-subtitle'>Paste a news article or headline to assess its credibility</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Article Text</div>', unsafe_allow_html=True)
    user_input = st.text_area("", height=180,
        placeholder="Paste a news article, headline, or paragraph here…",
        label_visibility="collapsed")

    col_btn, col_hint = st.columns([1, 5])
    with col_btn:
        analyze = st.button("Analyse Article")
    with col_hint:
        st.markdown('<span style="font-size:0.78rem;color:#9ca3af;">Minimum 30 words recommended for accurate results</span>', unsafe_allow_html=True)

    if analyze:
        if not model_loaded:
            st.error("No model found. Please place the .joblib files in the app directory.")
        elif not user_input.strip():
            st.warning("Please enter some article text before analysing.")
        else:
            with st.spinner("Analysing article…"):
                cleaned = clean_text(user_input)
                vec_input = vectorizer.transform([cleaned])
                prediction = model.predict(vec_input)[0]
                    score_raw = model.decision_function(vec_input)[0]
                    confidence = min(abs(float(score_raw)) / 3.0, 1.0) * 100

                is_real = str(prediction) in ["0", "real", "Real", "REAL"]

                if "history" not in st.session_state:
                    st.session_state.history = []
                import datetime
                st.session_state.history.insert(0, {
                    "time":       datetime.datetime.now().strftime("%H:%M:%S"),
                    "preview":    user_input[:80] + ("…" if len(user_input) > 80 else ""),
                    "verdict":    "Credible" if is_real else "Misinformation",
                    "confidence": f"{confidence:.1f}%",
                    "is_real":    is_real,
                    "words":      len(user_input.split())
                })

                st.markdown('<hr class="divider">', unsafe_allow_html=True)
                st.markdown('<div class="section-label">Result</div>', unsafe_allow_html=True)

                r1, r2, r3 = st.columns([2, 2, 2])
                with r1:
                    if is_real:
                        st.markdown('<div class="verdict-real">✓ &nbsp;Likely Credible</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="verdict-fake">✗ &nbsp;Likely Misinformation</div>', unsafe_allow_html=True)
                with r2:
                    bar_color = '#3ddc97' if is_real else '#ff5f40'
                    st.markdown(f"""
                    <div class="stat-box">
                        <div class="stat-value">{confidence:.0f}%</div>
                        <div class="stat-label">Confidence Score</div>
                        <div class="conf-bar-wrap">
                            <div class="conf-bar-fill" style="width:{confidence}%;background:{bar_color};"></div>
                        </div>
                    </div>""", unsafe_allow_html=True)
                with r3:
                    st.markdown(f"""
                    <div class="stat-box">
                        <div class="stat-value">{len(user_input.split())}</div>
                        <div class="stat-label">Word Count</div>
                    </div>""", unsafe_allow_html=True)

                feature_names = vectorizer.get_feature_names_out()
                coefs = model.coef_[0]
                tokens = cleaned.split()
                present = [(w, coefs[np.where(feature_names == w)[0][0]])
                           for w in tokens if w in feature_names]
                present_sorted = sorted(present, key=lambda x: abs(x[1]), reverse=True)[:14]

                if present_sorted:
                    st.markdown('<hr class="divider">', unsafe_allow_html=True)
                    st.markdown('<div class="section-label">Key Signals Detected</div>', unsafe_allow_html=True)
                    st.markdown('<div style="font-size:0.78rem;color:#9ca3af;margin-bottom:0.6rem;"><span style="color:#3ddc97;">Green</span> = credibility signal &nbsp;·&nbsp; <span style="color:#ff5f40;">Red</span> = misinformation signal</div>', unsafe_allow_html=True)
                    tags_html = "".join(
                        f'<span class="tag-real">{w}</span>' if s < 0 else f'<span class="tag-fake">{w}</span>'
                        for w, s in present_sorted
                    )
                    st.markdown(tags_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: HOW IT WORKS
# ─────────────────────────────────────────────
elif "📖  How It Works" in page:

    st.markdown("""
    <div style='margin-bottom:2rem;'>
        <div class='hero-title'>How It <span>Works</span></div>
        <div class='hero-sub'>Architecture · Pipeline · Design Decisions</div>
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("01", "Data Balancing", "Global news (~72k rows) is downsampled to 10k. Nigerian news is 5× oversampled to reduce class imbalance and regional bias."),
        ("02", "Text Preprocessing", "Lowercasing → URL/HTML stripping → non-alpha removal → NLTK tokenization → stopword removal → WordNet lemmatization."),
        ("03", "TF-IDF Vectorization", "Top 5,000 features selected. max_df=0.7 drops overly common terms. Sparse matrix fed to classifier."),
        ("04", "Passive Aggressive Classifier", "Online learning algorithm — updates only when it makes mistakes. Efficient for large text corpora. max_iter=50."),
        ("05", "Confidence Scoring", "Raw score from the hyperplane distance is used as confidence proxy: capped at 3.0 → scaled 0–100%."),
        ("06", "Word Signals", "Each word has a learned coefficient. Positive → Fake signal. Negative → Real signal."),
    ]

    for num, title, desc in steps:
        st.markdown(f"""
        <div class="card" style="display:flex;gap:1.4rem;align-items:flex-start;">
            <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:#e8ff45;opacity:0.25;line-height:1;min-width:2.5rem;">{num}</div>
            <div>
                <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1rem;color:#f0f0f5;margin-bottom:0.3rem;">{title}</div>
                <div style="font-size:0.88rem;color:#9090a8;line-height:1.6;">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <div class="card-label">Required Files</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.82rem;color:#9090a8;line-height:2;">
            fake_news_model.joblib &nbsp;·&nbsp; tfidf_vectorizer.joblib
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: HISTORY
# ─────────────────────────────────────────────
elif "🕓  History" in page:

    st.markdown("""
    <div class='page-header'>
        <div class='page-title'>Verification <span style='color:#e8ff45;'>History</span></div>
        <div class='page-subtitle'>Log of all articles analysed in this session</div>
    </div>""", unsafe_allow_html=True)

    history = st.session_state.get("history", [])

    if not history:
        st.markdown("""
        <div class="card" style="text-align:center;padding:2.5rem;">
            <div style="font-size:2rem;margin-bottom:0.8rem;">📭</div>
            <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#f0f0f5;margin-bottom:0.4rem;">No history yet</div>
            <div style="font-size:0.85rem;color:#6b6b80;">Articles you analyse in the Predict tab will appear here.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        total    = len(history)
        credible = sum(1 for h in history if h["is_real"])
        misinfo  = total - credible

        s1, s2, s3 = st.columns(3)
        s1.markdown(f'<div class="stat-box"><div class="stat-value">{total}</div><div class="stat-label">Total Checked</div></div>', unsafe_allow_html=True)
        s2.markdown(f'<div class="stat-box"><div class="stat-value" style="color:#3ddc97;">{credible}</div><div class="stat-label">Credible</div></div>', unsafe_allow_html=True)
        s3.markdown(f'<div class="stat-box"><div class="stat-value" style="color:#ff5f40;">{misinfo}</div><div class="stat-label">Misinformation</div></div>', unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        if total >= 2:
            st.markdown('<div class="section-label">Trend — Confidence per Check</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8, 2.2))
            fig.patch.set_facecolor('#12121a')
            ax.set_facecolor('#12121a')
            conf_vals  = [float(h["confidence"].replace('%','')) for h in reversed(history)]
            colors_seq = ['#3ddc97' if h["is_real"] else '#ff5f40' for h in reversed(history)]
            ax.bar(range(len(conf_vals)), conf_vals, color=colors_seq, edgecolor='none', width=0.6)
            ax.set_ylim(0, 105)
            ax.set_ylabel('Confidence %', color='#6b6b80', fontsize=8)
            ax.tick_params(colors='#6b6b80', labelsize=7)
            ax.set_xticks(range(len(conf_vals)))
            ax.set_xticklabels([f"#{i+1}" for i in range(len(conf_vals))], color='#6b6b80', fontsize=7)
            for spine in ax.spines.values(): spine.set_edgecolor('#1e1e2e')
            ax.legend(handles=[mpatches.Patch(color='#3ddc97', label='Credible'),
                                mpatches.Patch(color='#ff5f40', label='Misinformation')],
                      facecolor='#12121a', labelcolor='#a0a0b0', fontsize=8, framealpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('<hr class="divider">', unsafe_allow_html=True)

        st.markdown('<div class="section-label">All Checks — Most Recent First</div>', unsafe_allow_html=True)
        for i, h in enumerate(history):
            border_color  = '#3ddc97' if h["is_real"] else '#ff5f40'
            verdict_color = '#3ddc97' if h["is_real"] else '#ff5f40'
            verdict_icon  = '✓' if h["is_real"] else '✗'
            st.markdown(f"""
            <div class="card" style="border-left:3px solid {border_color};padding:1rem 1.2rem;margin-bottom:0.6rem;">
                <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem;">
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#6b6b80;">#{total - i} &nbsp;·&nbsp; {h['time']} &nbsp;·&nbsp; {h['words']} words</div>
                    <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:0.85rem;color:{verdict_color};">{verdict_icon} {h['verdict']} &nbsp;·&nbsp; {h['confidence']}</div>
                </div>
                <div style="font-size:0.88rem;color:#c0c0d0;margin-top:0.45rem;line-height:1.5;">{h['preview']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        hist_df = pd.DataFrame(history)[["time", "verdict", "confidence", "words", "preview"]]
        hist_df.columns = ["Time", "Verdict", "Confidence", "Words", "Article Preview"]
        st.download_button(
            "⬇ Export History as CSV",
            hist_df.to_csv(index=False).encode("utf-8"),
            "verification_history.csv",
            "text/csv"
        )
