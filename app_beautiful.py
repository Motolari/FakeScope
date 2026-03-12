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
    page_title="FakeScope — News Verifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Share+Tech+Mono&family=Rajdhani:wght@300;400;500;600;700&family=DM+Sans:wght@300;400;500&display=swap');

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
    font-family: 'Rajdhani', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-weight: 500;
    letter-spacing: 0.02em;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Sidebar nav radio labels */
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 300 !important;
    letter-spacing: 0.04em !important;
    color: var(--muted) !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    color: var(--text) !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] + div label,
[data-testid="stSidebar"] [data-testid="stRadio"] input:checked ~ label {
    color: var(--accent) !important;
    font-weight: 500 !important;
}

/* Hide default Streamlit header */
header[data-testid="stHeader"] { display: none; }

/* Title */
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
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.4rem;
}

/* Cards */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.card-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.4rem;
}
.card-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--text);
}

/* Result badge */
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

/* Divider */
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}

/* Streamlit widget overrides */
.stTextArea textarea {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
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
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.55rem 1.6rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
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
.stMetric { background: var(--surface); border-radius: 10px; padding: 0.8rem 1rem; }
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
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.78rem !important;
    color: var(--muted) !important;
    border-radius: 6px !important;
    letter-spacing: 0.05em !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: #0a0a0f !important;
}

/* ── MOBILE RESPONSIVE ── */
@media (max-width: 768px) {

    /* Hero title smaller on mobile */
    .hero-title {
        font-size: 2rem !important;
    }
    .hero-sub {
        font-size: 0.65rem !important;
    }

    /* Cards padding tighter */
    .card {
        padding: 1rem 1rem !important;
    }

    /* Verdict badges scale down */
    .verdict-real, .verdict-fake {
        font-size: 1.1rem !important;
        padding: 0.4rem 1rem !important;
    }

    /* Metrics stack better */
    [data-testid="stMetricValue"] {
        font-size: 1.3rem !important;
    }

    /* Bigger tap targets for buttons */
    .stButton > button {
        width: 100% !important;
        font-size: 1.05rem !important;
        padding: 0.75rem 1rem !important;
    }

    /* Tabs wrap properly */
    .stTabs [data-baseweb="tab-list"] {
        flex-wrap: wrap !important;
        gap: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.7rem !important;
        padding: 0.3rem 0.6rem !important;
    }

    /* Textarea bigger on mobile for easier typing */
    .stTextArea textarea {
        font-size: 1.05rem !important;
        min-height: 140px !important;
    }

    /* Sidebar nav labels bigger for tap */
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        font-size: 1.05rem !important;
        padding: 0.3rem 0 !important;
    }

    /* General text bump */
    html, body, [class*="css"] {
        font-size: 16px !important;
    }

    /* Block-level main padding */
    .block-container {
        padding: 1rem 0.8rem !important;
    }

    /* Tag pills wrap */
    .tag {
        font-size: 0.75rem !important;
        padding: 0.25rem 0.5rem !important;
    }
}
</style>""", unsafe_allow_html=True)


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
# TEXT CLEANING (mirrors your pipeline)
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
    model_path = 'fake_news_model.joblib'
    vec_path   = 'tfidf_vectorizer.joblib'
    if os.path.exists(model_path) and os.path.exists(vec_path):
        model = joblib.load(model_path)
        vec   = joblib.load(vec_path)
        return model, vec, True
    return None, None, False


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
        ["🔍  Predict", "⚙️  Train Model", "📊  Evaluate", "📖  How It Works"],
        label_visibility="collapsed"
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">Model Status</div>', unsafe_allow_html=True)

    model, vectorizer, model_loaded = load_model()
    if model_loaded:
        st.markdown('<span style="color:#3ddc97;font-family:\'IBM Plex Mono\',monospace;font-size:0.78rem;">● MODEL READY</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:#ff5f40;font-family:\'IBM Plex Mono\',monospace;font-size:0.78rem;">● NO MODEL FOUND</span>', unsafe_allow_html=True)
        st.caption("Go to ⚙️ Train Model to train one, or place `.joblib` files in the app directory.")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="card-label" style="margin-bottom:0.6rem;">About</div>', unsafe_allow_html=True)
    st.caption("Built with a Passive Aggressive Classifier + TF-IDF pipeline. Trained on balanced global & Nigerian news data.")


# ─────────────────────────────────────────────
# PAGE: PREDICT
# ─────────────────────────────────────────────
if "🔍  Predict" in page:

    st.markdown("""
    <div style='margin-bottom:2rem;'>
        <div class='hero-title'>News <span>Verifier</span></div>
        <div class='hero-sub'>Passive Aggressive Classifier · TF-IDF · NLP Pipeline</div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["  SINGLE ARTICLE  ", "  BULK CSV  "])

    # ── Single Article ──
    with tab1:
        st.markdown('<div class="card-label" style="margin-bottom:0.5rem;">Paste article text below</div>', unsafe_allow_html=True)
        user_input = st.text_area("", height=200, placeholder="Paste a news article, headline, or paragraph here…", label_visibility="collapsed")

        col_btn, col_tip = st.columns([1, 4])
        with col_btn:
            analyze = st.button("Analyze →")
        with col_tip:
            st.markdown('<span style="color:#6b6b80;font-size:0.78rem;font-family:\'IBM Plex Mono\',monospace;">min. ~30 words for best results</span>', unsafe_allow_html=True)

        if analyze:
            if not model_loaded:
                st.error("No model found. Please train one first in ⚙️ Train Model.")
            elif not user_input.strip():
                st.warning("Please enter some text.")
            else:
                with st.spinner("Analyzing…"):
                    cleaned = clean_text(user_input)
                    vec_input = vectorizer.transform([cleaned])
                    prediction = model.predict(vec_input)[0]
                    score_raw = model.decision_function(vec_input)[0]
                    confidence = min(abs(float(score_raw)) / 3.0, 1.0) * 100

                st.markdown('<hr class="divider">', unsafe_allow_html=True)

                r1, r2, r3 = st.columns([2, 1.5, 1.5])
                with r1:
                    if str(prediction) in ["0", "real", "Real", "REAL"]:
                        st.markdown('<div class="verdict-real">✓ REAL NEWS</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="verdict-fake">✗ FAKE NEWS</div>', unsafe_allow_html=True)
                with r2:
                    st.metric("Confidence", f"{confidence:.1f}%")
                with r3:
                    word_count = len(user_input.split())
                    st.metric("Word Count", word_count)

                st.markdown('<hr class="divider">', unsafe_allow_html=True)

                # Top words
                feature_names = vectorizer.get_feature_names_out()
                coefs = model.coef_[0]
                tokens = cleaned.split()
                present = [(w, coefs[np.where(feature_names == w)[0][0]])
                           for w in tokens if w in feature_names]
                present_sorted = sorted(present, key=lambda x: abs(x[1]), reverse=True)[:12]

                if present_sorted:
                    st.markdown('<div class="card-label">Key signals detected in your text</div>', unsafe_allow_html=True)
                    tags_html = ""
                    for word, score_val in present_sorted:
                        color = "#3ddc97" if score_val < 0 else "#ff5f40"
                        tags_html += f'<span class="tag" style="border-color:{color};color:{color};">{word}</span> '
                    st.markdown(tags_html, unsafe_allow_html=True)

    # ── Bulk CSV ──
    with tab2:
        st.markdown('<div class="card-label" style="margin-bottom:0.5rem;">Upload a CSV with a <code>text</code> column</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")

        if uploaded:
            if not model_loaded:
                st.error("No model found. Please train one first.")
            else:
                df_up = pd.read_csv(uploaded)
                if 'text' not in df_up.columns:
                    st.error("CSV must contain a `text` column.")
                else:
                    with st.spinner(f"Analyzing {len(df_up)} rows…"):
                        df_up['cleaned_text'] = df_up['text'].apply(clean_text)
                        vecs = vectorizer.transform(df_up['cleaned_text'])
                        df_up['prediction'] = model.predict(vecs)
                        scores = model.decision_function(vecs)
                        df_up['confidence'] = [min(abs(float(s))/3.0, 1.0)*100 for s in scores]

                    fake_count = (df_up['prediction'].astype(str).str.lower().isin(['1','fake'])).sum()
                    real_count = len(df_up) - fake_count

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Articles", len(df_up))
                    c2.metric("Real", real_count)
                    c3.metric("Fake", fake_count)

                    st.markdown('<hr class="divider">', unsafe_allow_html=True)
                    st.dataframe(
                        df_up[['text','prediction','confidence']].head(50),
                        use_container_width=True,
                        height=320
                    )

                    csv_out = df_up.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇ Download Results CSV", csv_out, "predictions.csv", "text/csv")


# ─────────────────────────────────────────────
# PAGE: TRAIN MODEL
# ─────────────────────────────────────────────
elif "⚙️  Train" in page:

    st.markdown("""
    <div style='margin-bottom:2rem;'>
        <div class='hero-title'>Train <span>Model</span></div>
        <div class='hero-sub'>Upload your dataset · configure · train in-browser</div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1.4, 1])

    with col_left:
        st.markdown('<div class="card-label" style="margin-bottom:0.5rem;">Upload preprocessed CSV</div>', unsafe_allow_html=True)
        st.caption("Needs columns: `cleaned_text` and `label`. This is your `final_preprocessed_data.csv`.")
        train_file = st.file_uploader("", type=["csv"], key="train_upload", label_visibility="collapsed")

    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Hyperparameters</div>', unsafe_allow_html=True)
        max_features = st.select_slider("TF-IDF Max Features", options=[1000, 2000, 3000, 5000, 8000, 10000], value=5000)
        max_iter     = st.slider("Classifier Max Iterations", 10, 200, 50, step=10)
        test_size    = st.slider("Test Split %", 10, 40, 20, step=5)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    if st.button("🚀  Start Training"):
        if train_file is None:
            st.error("Please upload a CSV file first.")
        else:
            df_train = pd.read_csv(train_file)
            required = {'cleaned_text', 'label'}
            if not required.issubset(df_train.columns):
                st.error(f"Missing columns. Found: {list(df_train.columns)}")
            else:
                df_train = df_train.dropna(subset=['cleaned_text'])
                df_train = df_train[df_train['cleaned_text'].str.strip() != ""]

                progress = st.progress(0, text="Splitting data…")
                X = df_train['cleaned_text']
                y = df_train['label']
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size/100, random_state=42)

                progress.progress(25, text="Vectorizing…")
                vec = TfidfVectorizer(stop_words='english', max_df=0.7, max_features=max_features)
                Xtr_vec = vec.fit_transform(X_tr)
                Xte_vec = vec.transform(X_te)

                progress.progress(55, text="Training classifier…")
                clf = PassiveAggressiveClassifier(max_iter=max_iter)
                clf.fit(Xtr_vec, y_tr)

                progress.progress(80, text="Evaluating…")
                y_pred_t = clf.predict(Xte_vec)
                acc = accuracy_score(y_te, y_pred_t)

                progress.progress(95, text="Saving model…")
                joblib.dump(clf, 'fake_news_model.joblib')
                joblib.dump(vec, 'tfidf_vectorizer.joblib')
                progress.progress(100, text="Done!")

                st.success("✓ Model trained and saved!")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy", f"{acc*100:.1f}%")
                m2.metric("Train Rows", len(X_tr))
                m3.metric("Test Rows",  len(X_te))
                m4.metric("Vocab Size", max_features)

                report = classification_report(y_te, y_pred_t, output_dict=True)
                rep_df = pd.DataFrame(report).transpose().round(3)
                st.dataframe(rep_df, use_container_width=True)

                st.info("Reload the app (press R) to use your new model in the Predict tab.")


# ─────────────────────────────────────────────
# PAGE: EVALUATE
# ─────────────────────────────────────────────
elif "📊  Evaluate" in page:

    st.markdown("""
    <div style='margin-bottom:2rem;'>
        <div class='hero-title'>Model <span>Evaluation</span></div>
        <div class='hero-sub'>Confusion matrix · word weights · feature importance</div>
    </div>
    """, unsafe_allow_html=True)

    if not model_loaded:
        st.warning("No model loaded. Train one first or place `.joblib` files in the app directory.")
    else:
        st.markdown('<div class="card-label" style="margin-bottom:0.5rem;">Upload test CSV to evaluate (needs <code>cleaned_text</code> + <code>label</code>)</div>', unsafe_allow_html=True)
        eval_file = st.file_uploader("", type=["csv"], key="eval_upload", label_visibility="collapsed")

        if eval_file:
            df_ev = pd.read_csv(eval_file).dropna(subset=['cleaned_text','label'])
            X_ev = vectorizer.transform(df_ev['cleaned_text'])
            y_ev = df_ev['label']
            y_ev_pred = model.predict(X_ev)
            acc_ev = accuracy_score(y_ev, y_ev_pred)

            e1, e2, e3 = st.columns(3)
            e1.metric("Accuracy",    f"{acc_ev*100:.2f}%")
            e2.metric("Total Rows",  len(df_ev))

            labels = sorted(y_ev.unique())
            e3.metric("Classes", len(labels))

            st.markdown('<hr class="divider">', unsafe_allow_html=True)

            c_left, c_right = st.columns(2)

            # Confusion Matrix
            with c_left:
                st.markdown('<div class="card-label">Confusion Matrix</div>', unsafe_allow_html=True)
                cm = confusion_matrix(y_ev, y_ev_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                fig.patch.set_facecolor('#12121a')
                ax.set_facecolor('#12121a')
                sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                            xticklabels=labels, yticklabels=labels, ax=ax,
                            linewidths=0.5, linecolor='#1e1e2e',
                            annot_kws={"fontsize": 13, "fontfamily": "monospace"})
                ax.set_xlabel('Predicted', color='#6b6b80', fontsize=9)
                ax.set_ylabel('Actual',    color='#6b6b80', fontsize=9)
                ax.tick_params(colors='#6b6b80')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#1e1e2e')
                plt.tight_layout()
                st.pyplot(fig)

            # Feature Importance
            with c_right:
                st.markdown('<div class="card-label">Top Feature Words</div>', unsafe_allow_html=True)
                feature_names = vectorizer.get_feature_names_out()
                coefs = model.coef_[0]
                word_df = pd.DataFrame({'word': feature_names, 'score': coefs})

                top_fake = word_df.nlargest(10, 'score')
                top_real = word_df.nsmallest(10, 'score')
                combined = pd.concat([top_real, top_fake])

                fig2, ax2 = plt.subplots(figsize=(5, 4))
                fig2.patch.set_facecolor('#12121a')
                ax2.set_facecolor('#12121a')
                colors = ['#3ddc97' if s < 0 else '#ff5f40' for s in combined['score']]
                ax2.barh(combined['word'], combined['score'], color=colors, edgecolor='none', height=0.65)
                ax2.set_xlabel('Coefficient Weight', color='#6b6b80', fontsize=9)
                ax2.tick_params(colors='#a0a0b0', labelsize=8)
                ax2.axvline(0, color='#1e1e2e', linewidth=1)
                for spine in ax2.spines.values():
                    spine.set_edgecolor('#1e1e2e')
                real_patch = mpatches.Patch(color='#3ddc97', label='Real signal')
                fake_patch = mpatches.Patch(color='#ff5f40', label='Fake signal')
                ax2.legend(handles=[real_patch, fake_patch], facecolor='#12121a', labelcolor='#a0a0b0', fontsize=8, framealpha=0.6)
                plt.tight_layout()
                st.pyplot(fig2)

            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<div class="card-label">Full Classification Report</div>', unsafe_allow_html=True)
            rep_df = pd.DataFrame(classification_report(y_ev, y_ev_pred, output_dict=True)).transpose().round(3)
            st.dataframe(rep_df, use_container_width=True)

        else:
            st.info("Upload a test CSV above to see evaluation metrics and charts.")

            # Show word importance even without test data
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<div class="card-label">Feature Importance (loaded model)</div>', unsafe_allow_html=True)
            feature_names = vectorizer.get_feature_names_out()
            coefs = model.coef_[0]
            word_df = pd.DataFrame({'word': feature_names, 'score': coefs})
            top_fake = word_df.nlargest(15, 'score')
            top_real = word_df.nsmallest(15, 'score')
            combined = pd.concat([top_real, top_fake])

            fig3, ax3 = plt.subplots(figsize=(10, 5))
            fig3.patch.set_facecolor('#12121a')
            ax3.set_facecolor('#12121a')
            colors = ['#3ddc97' if s < 0 else '#ff5f40' for s in combined['score']]
            ax3.barh(combined['word'], combined['score'], color=colors, edgecolor='none', height=0.7)
            ax3.set_xlabel('Coefficient Weight', color='#6b6b80')
            ax3.tick_params(colors='#a0a0b0', labelsize=9)
            ax3.axvline(0, color='#2a2a3a', linewidth=1.5)
            for spine in ax3.spines.values(): spine.set_edgecolor('#1e1e2e')
            real_patch = mpatches.Patch(color='#3ddc97', label='Real signal')
            fake_patch = mpatches.Patch(color='#ff5f40', label='Fake signal')
            ax3.legend(handles=[real_patch, fake_patch], facecolor='#12121a', labelcolor='#a0a0b0', framealpha=0.6)
            plt.tight_layout()
            st.pyplot(fig3)


# ─────────────────────────────────────────────
# PAGE: HOW IT WORKS
# ─────────────────────────────────────────────
elif "📖  How" in page:

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
        ("05", "Decision Function", "Raw score from the hyperplane distance is used as confidence proxy: capped at 3.0 → scaled 0–100%."),
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
