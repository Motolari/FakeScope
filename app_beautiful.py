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
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
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
# PROFESSIONAL CSS — Clean Light Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

:root {
    --bg:        #0a0a0f;
    --surface:   #12121a;
    --border:    #1e1e2e;
    --accent:    #e8ff45;
    --accent2:   #ff5f40;
    --accent3:   #40d9ff;
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

.block-container {
    padding: 2rem 2.5rem !important;
    max-width: 1100px !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 400 !important;
    color: var(--muted) !important;
    padding: 0.3rem 0 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    color: var(--text) !important;
}

/* Hero title */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 3.2rem;
    letter-spacing: -0.03em;
    line-height: 1;
    color: var(--text);
}
.hero-title span { color: var(--accent); }
.hero-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.4rem;
}

/* Page header (for inner pages) */
.page-header {
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.25rem;
    margin-bottom: 2rem;
}
.page-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.02em;
    margin: 0;
}
.page-title span { color: var(--accent); }
.page-subtitle {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 0.3rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.5rem;
}

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.card-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.4rem;
}

/* Verdict badges */
.verdict-real {
    display: inline-block;
    background: #0d2e1e;
    border: 2px solid var(--real);
    color: var(--real);
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    padding: 0.5rem 1.4rem;
    border-radius: 8px;
    letter-spacing: 0.05em;
}
.verdict-fake {
    display: inline-block;
    background: #2e0d0d;
    border: 2px solid var(--fake);
    color: var(--fake);
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    padding: 0.5rem 1.4rem;
    border-radius: 8px;
    letter-spacing: 0.05em;
}

/* Tag pill */
.tag {
    display: inline-block;
    background: #1a1a26;
    border: 1px solid var(--border);
    color: var(--accent);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    margin: 2px;
}

/* Stat boxes */
.stat-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.1rem 1.3rem;
    text-align: center;
}
.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1.1;
}
.stat-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.25rem;
}

.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}

/* Streamlit overrides */
.stTextArea textarea {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.95rem !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(232,255,69,0.15) !important;
}
.stButton > button {
    background: var(--accent) !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.55rem 1.6rem !important;
    letter-spacing: 0.04em !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
.stSelectbox div[data-baseweb="select"] > div,
.stFileUploader section {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}
.stProgress > div > div { background: var(--accent) !important; }
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: var(--accent) !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.68rem !important;
    color: var(--muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
    padding: 4px !important;
    gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    color: var(--muted) !important;
    border-radius: 6px !important;
    letter-spacing: 0.05em !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: #0a0a0f !important;
}

/* Mobile */
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

/* Show mobile nav only on small screens */
.mobile-nav { display: none; }
@media (max-width: 768px) {
    .mobile-nav { display: block; margin-bottom: 1rem; }
}
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


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
model, vectorizer, model_loaded = load_model()

# ─────────────────────────────────────────────
# SIDEBAR (desktop)
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='margin-bottom:1.8rem;'>
        <div style='font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;color:#f0f0f5;'>
            FAKE<span style='color:#e8ff45;'>SCOPE</span>
        </div>
        <div style='font-family:"Share Tech Mono",monospace;font-size:0.62rem;color:#6b6b80;letter-spacing:0.12em;text-transform:uppercase;'>
            News Intelligence Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card-label">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        "",
        ["🔍  Predict", "⚙️  Train Model", "📊  Evaluate", "📖  How It Works"],
        label_visibility="collapsed"
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">Model Status</div>', unsafe_allow_html=True)
    if model_loaded:
        st.markdown('<span style="color:#3ddc97;font-family:\'Share Tech Mono\',monospace;font-size:0.78rem;">● MODEL READY</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:#ff5f40;font-family:\'Share Tech Mono\',monospace;font-size:0.78rem;">● NO MODEL FOUND</span>', unsafe_allow_html=True)
        st.caption("Go to ⚙️ Train Model to train one, or place `.joblib` files in the app directory.")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="card-label" style="margin-bottom:0.6rem;">About</div>', unsafe_allow_html=True)
    st.caption("Built with a Passive Aggressive Classifier + TF-IDF pipeline. Trained on balanced global & Nigerian news data.")

# ─────────────────────────────────────────────
# MOBILE TOP DROPDOWN NAV
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* Show mobile nav only on small screens */
.mobile-nav { display: none; }
@media (max-width: 768px) {
    .mobile-nav { display: block; margin-bottom: 1rem; }
    /* Hide sidebar on mobile */
    [data-testid="stSidebar"] { display: none !important; }
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="mobile-nav">', unsafe_allow_html=True)
mobile_page = st.selectbox(
    "Navigate",
    ["🔍  Predict", "⚙️  Train Model", "📊  Evaluate", "📖  How It Works"],
    label_visibility="collapsed",
    key="mobile_nav"
)
st.markdown('</div>', unsafe_allow_html=True)

# Use whichever nav is active (mobile overrides sidebar on small screens)
import streamlit.components.v1 as components
# Sync: on mobile the selectbox drives page, on desktop the sidebar radio does
# We detect by checking if sidebar is visible — use session state trick
if "is_mobile" not in st.session_state:
    st.session_state.is_mobile = False

# Final page selection — mobile selectbox takes priority if it was changed
if st.session_state.get("mobile_nav") and st.session_state.mobile_nav != st.session_state.get("last_sidebar_page"):
    page = mobile_page
st.session_state.last_sidebar_page = page

# ─────────────────────────────────────────────
# GLOBAL TITLE (shows on every page)
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

    tab1, tab2 = st.tabs(["Single Article", "Bulk Analysis"])

    with tab1:
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
                st.error("No model found. Please train one first in the Train Model section.")
            elif not user_input.strip():
                st.warning("Please enter some article text before analysing.")
            else:
                with st.spinner("Analysing article…"):
                    cleaned = clean_text(user_input)
                    vec_input = vectorizer.transform([cleaned])
                    prediction = model.predict(vec_input)[0]
                    score_raw = model.decision_function(vec_input)[0]
                    confidence = min(abs(float(score_raw)) / 3.0, 1.0) * 100

                st.markdown('<hr class="divider">', unsafe_allow_html=True)
                st.markdown('<div class="section-label">Result</div>', unsafe_allow_html=True)
                is_real = str(prediction) in ["0", "real", "Real", "REAL"]

                r1, r2, r3 = st.columns([2, 2, 2])
                with r1:
                    if is_real:
                        st.markdown('<div class="verdict-real">✓ &nbsp;Likely Credible</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="verdict-fake">✗ &nbsp;Likely Misinformation</div>', unsafe_allow_html=True)
                with r2:
                    bar_color = '#047857' if is_real else '#b91c1c'
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
                    st.markdown('<div style="font-size:0.78rem;color:#9ca3af;margin-bottom:0.6rem;"><span style="color:#047857;">Green</span> = credibility signal &nbsp;·&nbsp; <span style="color:#b91c1c;">Red</span> = misinformation signal</div>', unsafe_allow_html=True)
                    tags_html = "".join(
                        f'<span class="tag-real">{w}</span>' if s < 0 else f'<span class="tag-fake">{w}</span>'
                        for w, s in present_sorted
                    )
                    st.markdown(tags_html, unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section-label">Upload CSV File</div>', unsafe_allow_html=True)
        st.caption("CSV must contain a `text` column. Upload your dataset for batch prediction.")
        uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")

        if uploaded:
            if not model_loaded:
                st.error("No model found. Please train one first.")
            else:
                df_up = pd.read_csv(uploaded)
                if 'text' not in df_up.columns:
                    st.error("CSV must contain a column named `text`.")
                else:
                    with st.spinner(f"Analysing {len(df_up)} articles…"):
                        df_up['cleaned_text'] = df_up['text'].apply(clean_text)
                        vecs = vectorizer.transform(df_up['cleaned_text'])
                        df_up['prediction'] = model.predict(vecs)
                        scores = model.decision_function(vecs)
                        df_up['confidence'] = [f"{min(abs(float(s))/3.0,1.0)*100:.1f}%" for s in scores]

                    fake_count = (df_up['prediction'].astype(str).str.lower().isin(['1','fake'])).sum()
                    real_count = len(df_up) - fake_count

                    st.markdown('<hr class="divider">', unsafe_allow_html=True)
                    c1, c2, c3 = st.columns(3)
                    c1.markdown(f'<div class="stat-box"><div class="stat-value">{len(df_up)}</div><div class="stat-label">Total Articles</div></div>', unsafe_allow_html=True)
                    c2.markdown(f'<div class="stat-box"><div class="stat-value" style="color:#047857;">{real_count}</div><div class="stat-label">Credible</div></div>', unsafe_allow_html=True)
                    c3.markdown(f'<div class="stat-box"><div class="stat-value" style="color:#b91c1c;">{fake_count}</div><div class="stat-label">Misinformation</div></div>', unsafe_allow_html=True)

                    st.markdown('<hr class="divider">', unsafe_allow_html=True)
                    st.dataframe(df_up[['text','prediction','confidence']].head(50), use_container_width=True, height=300)
                    st.download_button("Download Results as CSV", df_up.to_csv(index=False).encode('utf-8'), "predictions.csv", "text/csv")


# ─────────────────────────────────────────────
# PAGE: TRAIN MODEL
# ─────────────────────────────────────────────
elif "⚙️  Train Model" in page:

    st.markdown("""
    <div class='page-header'>
        <div class='page-title'>Model Training</div>
        <div class='page-subtitle'>Upload your preprocessed dataset and configure training parameters</div>
    </div>""", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.5, 1])
    with col_left:
        st.markdown('<div class="section-label">Dataset</div>', unsafe_allow_html=True)
        st.caption("Upload `final_preprocessed_data.csv` — requires columns: `cleaned_text` and `label`.")
        train_file = st.file_uploader("", type=["csv"], key="train_upload", label_visibility="collapsed")

    with col_right:
        st.markdown('<div class="section-label">Hyperparameters</div>', unsafe_allow_html=True)
        max_features = st.select_slider("TF-IDF Vocabulary Size", options=[1000, 2000, 3000, 5000, 8000, 10000], value=5000)
        max_iter     = st.slider("Max Training Iterations", 10, 200, 50, step=10)
        test_size    = st.slider("Test Set Size (%)", 10, 40, 20, step=5)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    if st.button("Start Training"):
        if train_file is None:
            st.error("Please upload a CSV file before starting training.")
        else:
            df_train = pd.read_csv(train_file)
            if not {'cleaned_text', 'label'}.issubset(df_train.columns):
                st.error(f"Missing required columns. Found: {list(df_train.columns)}")
            else:
                df_train = df_train.dropna(subset=['cleaned_text'])
                df_train = df_train[df_train['cleaned_text'].str.strip() != ""]
                X, y = df_train['cleaned_text'], df_train['label']
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size/100, random_state=42)

                progress = st.progress(0, text="Preparing data…")
                vec = TfidfVectorizer(stop_words='english', max_df=0.7, max_features=max_features)
                progress.progress(25, text="Vectorizing text features…")
                Xtr_vec = vec.fit_transform(X_tr)
                Xte_vec = vec.transform(X_te)

                progress.progress(55, text="Training classifier…")
                clf = PassiveAggressiveClassifier(max_iter=max_iter)
                clf.fit(Xtr_vec, y_tr)

                progress.progress(80, text="Evaluating performance…")
                y_pred_t = clf.predict(Xte_vec)
                acc = accuracy_score(y_te, y_pred_t)

                progress.progress(95, text="Saving model files…")
                joblib.dump(clf, 'fake_news_model.joblib')
                joblib.dump(vec, 'tfidf_vectorizer.joblib')
                progress.progress(100, text="Complete.")

                st.success("Model trained and saved successfully.")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy", f"{acc*100:.1f}%")
                m2.metric("Training Rows", f"{len(X_tr):,}")
                m3.metric("Test Rows", f"{len(X_te):,}")
                m4.metric("Vocabulary Size", f"{max_features:,}")

                st.markdown('<hr class="divider">', unsafe_allow_html=True)
                st.markdown('<div class="section-label">Classification Report</div>', unsafe_allow_html=True)
                rep_df = pd.DataFrame(classification_report(y_te, y_pred_t, output_dict=True)).transpose().round(3)
                st.dataframe(rep_df, use_container_width=True)
                st.info("Reload the page to use your new model in the Predict section.")


# ─────────────────────────────────────────────
# PAGE: EVALUATE
# ─────────────────────────────────────────────
elif "📊  Evaluate" in page:

    st.markdown("""
    <div class='page-header'>
        <div class='page-title'>Model Evaluation</div>
        <div class='page-subtitle'>Assess model performance with confusion matrix and feature analysis</div>
    </div>""", unsafe_allow_html=True)

    if not model_loaded:
        st.warning("No model loaded. Train a model first or place `.joblib` files in the app directory.")
    else:
        st.markdown('<div class="section-label">Test Dataset</div>', unsafe_allow_html=True)
        st.caption("Upload a CSV with `cleaned_text` and `label` columns to evaluate performance.")
        eval_file = st.file_uploader("", type=["csv"], key="eval_upload", label_visibility="collapsed")

        if eval_file:
            df_ev = pd.read_csv(eval_file).dropna(subset=['cleaned_text','label'])
            X_ev = vectorizer.transform(df_ev['cleaned_text'])
            y_ev = df_ev['label']
            y_ev_pred = model.predict(X_ev)
            acc_ev = accuracy_score(y_ev, y_ev_pred)
            labels = sorted(y_ev.unique())

            e1, e2, e3 = st.columns(3)
            e1.metric("Overall Accuracy", f"{acc_ev*100:.2f}%")
            e2.metric("Total Samples", f"{len(df_ev):,}")
            e3.metric("Classes", len(labels))

            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            c_left, c_right = st.columns(2)

            with c_left:
                st.markdown('<div class="section-label">Confusion Matrix</div>', unsafe_allow_html=True)
                cm = confusion_matrix(y_ev, y_ev_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                fig.patch.set_facecolor('#ffffff')
                ax.set_facecolor('#ffffff')
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=labels, yticklabels=labels, ax=ax,
                            linewidths=0.5, linecolor='#e2e5ea',
                            annot_kws={"fontsize": 13, "fontfamily": "monospace"})
                ax.set_xlabel('Predicted Label', color='#6b7280', fontsize=9)
                ax.set_ylabel('True Label', color='#6b7280', fontsize=9)
                ax.tick_params(colors='#374151', labelsize=9)
                for spine in ax.spines.values():
                    spine.set_edgecolor('#e2e5ea')
                plt.tight_layout()
                st.pyplot(fig)

            with c_right:
                st.markdown('<div class="section-label">Feature Importance</div>', unsafe_allow_html=True)
                feature_names = vectorizer.get_feature_names_out()
                coefs = model.coef_[0]
                word_df = pd.DataFrame({'word': feature_names, 'score': coefs})
                combined = pd.concat([word_df.nsmallest(10,'score'), word_df.nlargest(10,'score')])

                fig2, ax2 = plt.subplots(figsize=(5, 4))
                fig2.patch.set_facecolor('#ffffff')
                ax2.set_facecolor('#ffffff')
                colors = ['#047857' if s < 0 else '#b91c1c' for s in combined['score']]
                ax2.barh(combined['word'], combined['score'], color=colors, edgecolor='none', height=0.65)
                ax2.set_xlabel('Coefficient Weight', color='#6b7280', fontsize=9)
                ax2.tick_params(colors='#374151', labelsize=8)
                ax2.axvline(0, color='#e2e5ea', linewidth=1.5)
                for spine in ax2.spines.values():
                    spine.set_edgecolor('#e2e5ea')
                ax2.legend(handles=[mpatches.Patch(color='#047857', label='Credibility signal'),
                                    mpatches.Patch(color='#b91c1c', label='Misinformation signal')],
                           facecolor='#ffffff', labelcolor='#374151', fontsize=8, framealpha=0.8)
                plt.tight_layout()
                st.pyplot(fig2)

            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Full Classification Report</div>', unsafe_allow_html=True)
            rep_df = pd.DataFrame(classification_report(y_ev, y_ev_pred, output_dict=True)).transpose().round(3)
            st.dataframe(rep_df, use_container_width=True)

        else:
            st.info("Upload a test CSV above to view evaluation metrics.")
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Feature Importance — Loaded Model</div>', unsafe_allow_html=True)
            feature_names = vectorizer.get_feature_names_out()
            coefs = model.coef_[0]
            word_df = pd.DataFrame({'word': feature_names, 'score': coefs})
            combined = pd.concat([word_df.nsmallest(15,'score'), word_df.nlargest(15,'score')])

            fig3, ax3 = plt.subplots(figsize=(10, 5))
            fig3.patch.set_facecolor('#ffffff')
            ax3.set_facecolor('#ffffff')
            colors = ['#047857' if s < 0 else '#b91c1c' for s in combined['score']]
            ax3.barh(combined['word'], combined['score'], color=colors, edgecolor='none', height=0.7)
            ax3.set_xlabel('Coefficient Weight', color='#6b7280')
            ax3.tick_params(colors='#374151', labelsize=9)
            ax3.axvline(0, color='#e2e5ea', linewidth=1.5)
            for spine in ax3.spines.values():
                spine.set_edgecolor('#e2e5ea')
            ax3.legend(handles=[mpatches.Patch(color='#047857', label='Credibility signal'),
                                 mpatches.Patch(color='#b91c1c', label='Misinformation signal')],
                       facecolor='#ffffff', labelcolor='#374151', framealpha=0.8)
            plt.tight_layout()
            st.pyplot(fig3)


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
        ("06", "Word Signals", "Each word has a learned coefficient. Positive → Fake signal. Negative → Real signal. Visualized in the Evaluate tab."),
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
            fake_news_model.joblib &nbsp;·&nbsp; tfidf_vectorizer.joblib<br>
            combined_news_data.csv &nbsp;·&nbsp; balanced_dataset.csv &nbsp;·&nbsp; final_preprocessed_data.csv
        </div>
    </div>
    """, unsafe_allow_html=True)
