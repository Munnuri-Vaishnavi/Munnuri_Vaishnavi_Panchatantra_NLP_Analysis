"""
╔══════════════════════════════════════════════════════════════╗
║   Panchatantra NLP — Streamlit App                           ║                                                          ║
║   Run: streamlit run panchatantra_app.py                     ║
╚══════════════════════════════════════════════════════════════╝
"""

# ── Imports  ─────────────────────────────────────
import re
import io
import base64
import itertools
import random
from io import BytesIO
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS

import nltk
nltk.download('stopwords',     quiet=True)
nltk.download('punkt',         quiet=True)
nltk.download('punkt_tab',     quiet=True)
nltk.download('wordnet',       quiet=True)
nltk.download('vader_lexicon', quiet=True)

from nltk.corpus    import stopwords
from nltk.tokenize  import word_tokenize, sent_tokenize
from nltk.stem      import WordNetLemmatizer
from nltk           import bigrams
from nltk.sentiment import SentimentIntensityAnalyzer

import plotly.graph_objects as go
import plotly.express       as px
import plotly.io            as pio

# ══════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="📖 Panchatantra NLP",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel+Decorative:wght@400;700&family=Cinzel:wght@400;600;700&family=Crimson+Pro:ital,wght@0,300;0,400;0,600;1,400&display=swap');

html,body,[class*="css"]{font-family:'Crimson Pro',Georgia,serif;background:#080814;color:#e8e0f0;}
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:#0f0f22}
::-webkit-scrollbar-thumb{background:#4a3a6a;border-radius:3px}

.hero{text-align:center;padding:28px 0 6px;background:linear-gradient(180deg,#1a0a2e 0%,transparent 100%);border-radius:18px;margin-bottom:6px;}
.hero-title{font-family:'Cinzel Decorative',serif;font-size:2.4rem;font-weight:700;background:linear-gradient(135deg,#c084fc,#818cf8,#c084fc);-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:3px;margin-bottom:4px;}
.hero-sub{color:#8b7aaa;font-size:.95rem;letter-spacing:2px;font-style:italic;}

.sec{font-family:'Cinzel',serif;font-size:1rem;color:#c084fc;border-left:3px solid #c084fc;padding-left:11px;margin:24px 0 12px;letter-spacing:1px;}

.stat-row{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:16px 0;}
.scard{background:linear-gradient(135deg,#1c1040,#120e30);border:1px solid #3d2d6a;border-radius:14px;padding:18px 10px;text-align:center;position:relative;overflow:hidden;box-shadow:0 8px 28px rgba(0,0,0,.55);}
.scard::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,#c084fc,transparent);}
.snum{font-family:'Cinzel',serif;font-size:1.9rem;font-weight:700;color:#c084fc;line-height:1;}
.slabel{font-size:.68rem;color:#7a6a9a;margin-top:5px;letter-spacing:1.5px;text-transform:uppercase;}

.storycard{background:linear-gradient(135deg,#110e28,#0e0c22);border:1px solid #2d2050;border-radius:14px;padding:20px 24px;margin-bottom:16px;line-height:1.85;font-size:1.02rem;box-shadow:0 4px 20px rgba(0,0,0,.45);position:relative;overflow:hidden;}
.storycard::before{content:'';position:absolute;top:0;left:0;bottom:0;width:3px;background:linear-gradient(180deg,#c084fc,#818cf8);}
.stitle{font-family:'Cinzel',serif;color:#c084fc;font-size:1.1rem;margin-bottom:10px;font-weight:600;}
.sttext{color:#d0c8e8;line-height:1.85;}
.badge{display:inline-block;background:rgba(192,132,252,.1);border:1px solid rgba(192,132,252,.28);border-radius:18px;padding:2px 11px;font-size:.73rem;color:#c084fc;margin:3px 3px 3px 0;font-family:'Cinzel',serif;letter-spacing:.4px;}
.moral{margin-top:13px;padding-top:12px;border-top:1px solid #2d2050;color:#a594cc;font-style:italic;font-size:.96rem;}
.moral b{color:#c084fc;font-style:normal;}

.ibox{background:linear-gradient(135deg,#110e28,#0e0c22);border:1px solid #3d2d6a;border-radius:11px;padding:14px 18px;font-size:.9rem;color:#c8bce0;line-height:1.65;}
.ilabel{font-family:'Cinzel',serif;color:#c084fc;font-size:.72rem;letter-spacing:1px;text-transform:uppercase;margin-bottom:5px;}
.ival{font-size:1.08rem;color:#e8e0f0;font-weight:600;}

.qcard{background:linear-gradient(135deg,#12102a,#0e0c22);border:1px solid #3d2d6a;border-radius:14px;padding:26px 30px;margin-bottom:16px;box-shadow:0 8px 36px rgba(0,0,0,.55);}
.qq{font-family:'Cinzel',serif;font-size:1rem;color:#e8e0f0;margin-bottom:18px;line-height:1.55;}

.fcard{background:linear-gradient(135deg,#1c1040,#120e30);border:1px solid #5a3a8a;border-radius:18px;padding:36px 44px;text-align:center;min-height:200px;display:flex;flex-direction:column;justify-content:center;box-shadow:0 10px 44px rgba(192,132,252,.13);margin:10px 0;}
.flabel{font-family:'Cinzel',serif;font-size:.68rem;letter-spacing:2px;color:#7a6a9a;text-transform:uppercase;margin-bottom:14px;}
.ftext{font-size:1.2rem;color:#e8e0f0;line-height:1.65;font-style:italic;}
.fanswer{font-family:'Cinzel',serif;font-size:1.35rem;color:#c084fc;font-weight:700;}

.div{height:1px;background:linear-gradient(90deg,transparent,#3d2d6a,transparent);margin:24px 0;}

[data-testid="stSidebar"]{background:linear-gradient(180deg,#0a0818,#0f0e22);border-right:1px solid #2a1a4a;}
.slogo{font-family:'Cinzel Decorative',serif;font-size:1.05rem;color:#c084fc;text-align:center;padding:10px 0 3px;letter-spacing:2px;}
.ssub{font-size:.68rem;color:#5a4a7a;text-align:center;letter-spacing:1px;margin-bottom:14px;}

div.stButton>button{background:linear-gradient(135deg,#3d2d6a,#2a1a4a)!important;color:#c084fc!important;border:1px solid #5a3a8a!important;border-radius:9px!important;font-family:'Cinzel',serif!important;letter-spacing:.4px!important;transition:all .2s!important;}
div.stButton>button:hover{background:linear-gradient(135deg,#5a3a8a,#3d2d6a)!important;border-color:#c084fc!important;transform:translateY(-1px)!important;box-shadow:0 4px 14px rgba(192,132,252,.28)!important;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════

# ANIMALS list 
ANIMALS = [
    'monkey', 'lion', 'rabbit', 'fox', 'crow', 'tortoise', 'jackal',
    'elephant', 'snake', 'mouse', 'crocodile', 'deer', 'goat', 'tiger',
    'fish', 'dog', 'heron', 'crab', 'camel', 'bird', 'parrot', 'peacock',
    'donkey', 'crane', 'mongoose', 'hyena', 'wolf', 'hare', 'frog', 'hawk'
]

# ACTION_WORDS list 
ACTION_WORDS = [
    'tricked', 'escaped', 'saved', 'betrayed', 'warned', 'fought',
    'helped', 'planned', 'fled', 'attacked', 'sacrificed', 'cooperated',
    'defeated', 'punished', 'trusted', 'deceived', 'protected', 'united',
    'revealed', 'won'
]

# SMART_WORDS 
SMART_WORDS = {
    'intelligence', 'clever', 'wit', 'tricked', 'outsmarted', 'strategy',
    'cunning', 'wisdom', 'planned', 'deceived', 'outwitted', 'scheme',
    'calculated', 'resourceful', 'sharp'
}

# THEME_COLORS dict
THEME_COLORS = {
    'trust':          '#e74c3c', 'intelligence':   '#3498db',
    'greed':          '#f39c12', 'overconfidence': '#9b59b6',
    'identity':       '#1abc9c', 'cooperation':    '#2ecc71',
    'ego':            '#e67e22', 'gratitude':      '#16a085',
    'wisdom':         '#8e44ad', 'revenge':        '#c0392b',
    'curiosity':      '#d35400', 'sacrifice':      '#27ae60',
    'betrayal':       '#e91e63', 'strategy':       '#00bcd4',
    'survival':       '#ff5722', 'leadership':     '#607d8b',
    'justice':        '#795548', 'unity':          '#4caf50',
    'caution':        '#ffc107', 'bravery':        '#2196f3',
    'teamwork':       '#009688', 'responsibility': '#ff9800',
    'other':          '#95a5a6'
}

DARK   = "#080814"
CARD   = "#110e28"
TEXT   = "#e8e0f0"
GRID   = "#2a1a4a"
ACCENT = "#c084fc"
LAY    = dict(paper_bgcolor=DARK, plot_bgcolor=CARD,
              font=dict(color=TEXT, family="Crimson Pro, Georgia, serif"),
              margin=dict(l=40, r=20, t=50, b=40))

def dax(fig):
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, color=TEXT)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, color=TEXT)
    return fig

# ══════════════════════════════════════════════════════════
# NLP FUNCTIONS
# ══════════════════════════════════════════════════════════

@st.cache_data
def build_stopwords():
    # Exact same as notebook Step 3
    base_stopwords = set(stopwords.words('english'))
    extra_stopwords = {
        'said', 'came', 'went', 'got', 'told', 'asked', 'made', 'let',
        'also', 'back', 'away', 'upon', 'one', 'two', 'day', 'time',
        'man', 'way', 'could', 'would', 'become', 'became', 'later',
        'soon', 'then', 'there', 'worked', 'helped', 'learned', 'saved',
        'ensured', 'restored', 'removed', 'returned', 'happened',
        'followed', 'planned', 'found', 'shown', 'needed', 'matters',
        'wins', 'failed', 'escaped', 'without', 'every', 'still', 'even',
        'much', 'well', 'first', 'long', 'finally', 'simply', 'entirely',
        'rather', 'already', 'however', 'though', 'before', 'after',
        'within', 'toward', 'between', 'around', 'against', 'along'
    }
    return base_stopwords | extra_stopwords

ALL_SW = build_stopwords()

# clean_to_tokens() in notebook Step 
def clean_to_tokens(text):
    text  = str(text).lower()
    text  = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in ALL_SW and len(w) > 2]
    return words

# clean_to_string() in notebook Step 3
def clean_to_string(text):
    return ' '.join(clean_to_tokens(text))

# get_sentiment() in notebook Step 5
def get_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(str(text))['compound']

#  get_intelligence_score() in notebook Step 5
def get_intelligence_score(text):
    token_set = set(clean_to_tokens(text))
    overlap   = len(token_set & SMART_WORDS)
    score     = overlap / max(len(token_set), 1) * 100
    return round(score, 2)

#  get_main_character() in notebook Step 5
def get_main_character(text):
    text   = str(text).lower()
    counts = {animal: text.count(animal) for animal in ANIMALS if animal in text}
    return max(counts, key=counts.get) if counts else 'other'

#  count_actions() in notebook Step 5
def count_actions(text):
    text = str(text).lower()
    return {action: text.count(action) for action in ACTION_WORDS}

# Sentence-level sentiment flow (same VADER logic as notebook Step 8)
def story_sentiment_flow(story_text):
    sia       = SentimentIntensityAnalyzer()
    sentences = re.split(r'[.!?]', str(story_text))
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    return [sia.polarity_scores(s)['compound'] for s in sentences]

# Lemmatizer — same as notebook Step 4
lemmatizer = WordNetLemmatizer()
def lemmatize_tokens(token_list):
    return [lemmatizer.lemmatize(word) for word in token_list]

# TTS helper
def tts_audio(text):
    try:
        tts = gTTS(text=str(text), lang='en')
        buf = BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        return f'<audio controls style="width:100%;margin-top:8px;"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
    except Exception as e:
        return f"<p style='color:#f87171'>Audio error: {e}</p>"

# ══════════════════════════════════════════════════════════
# MAIN DATA PROCESSING 
# ══════════════════════════════════════════════════════════
@st.cache_data
def load_and_process(raw_bytes):
    df = pd.read_csv(io.BytesIO(raw_bytes))

    # Step 3 — cleaning (exact same as notebook)
    df['tokens']      = df['story'].apply(clean_to_tokens)
    df['clean_story'] = df['story'].apply(clean_to_string)
    df['word_count']  = df['story'].apply(lambda x: len(str(x).split()))

    # Step 4 — tokenization + lemmatization (exact same as notebook)
    df['word_tokens']       = df['clean_story'].apply(word_tokenize)
    df['sent_tokens']       = df['story'].apply(sent_tokenize)
    df['lemmatized_tokens'] = df['word_tokens'].apply(lemmatize_tokens)
    df['final_text']        = df['lemmatized_tokens'].apply(lambda x: ' '.join(x))
    df['bigrams_col']       = df['lemmatized_tokens'].apply(lambda x: list(bigrams(x)))

    # Step 5 — features (exact same as notebook)
    df['sentiment']      = df['story'].apply(get_sentiment)
    df['intel_score']    = df['story'].apply(get_intelligence_score)
    df['main_character'] = df['story'].apply(get_main_character)

    return df

# ══════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════
for k, v in [("cart",[]), ("quiz_idx",0), ("quiz_score",0),
             ("quiz_done",False), ("quiz_ans",None),
             ("fc_idx",0), ("fc_flipped",False), ("daily",None)]:
    if k not in st.session_state:
        st.session_state[k] = v

def add_to_cart(sid, df):
    ids = [s["story_id"] for s in st.session_state.cart]
    if sid not in ids:
        r = df[df["story_id"] == sid].iloc[0]
        st.session_state.cart.append({
            "story_id": int(sid), "title": r["title"],
            "story": r["story"], "moral": r["moral"],
            "theme": r["theme"], "emotion": r["emotion"],
        })
        st.toast(f"📚 '{r['title']}' added to cart!", icon="✅")
    else:
        st.toast("Already in your cart!", icon="ℹ️")

# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="slogo">📖 PANCHATANTRA</div>', unsafe_allow_html=True)
    st.markdown('<div class="ssub">NLP ANALYSIS SUITE</div>', unsafe_allow_html=True)
    st.markdown("---")
    import os

# Load from repo directly if running on Streamlit Cloud

   
    with open("panchatantra_full_50.csv", "rb") as f:
          
         df = load_and_process(f.read())
  
    st.markdown("---")
    n_cart = len(st.session_state.cart)
    page   = st.radio("", [
        "🏠  Overview",
        "📊  EDA Charts",
        "🧠  NLP Analysis",
        "🔍  Story Explorer",
        "🎙️  Voice Reader",
        "🎮  Story Quiz",
        "🤖  Recommender",
        "📖  Flashcard Mode",
        "🔬  Deep Analysis",
        f"🛒  My Cart ({n_cart})",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("<div style='font-size:.7rem;color:#3a2a5a;text-align:center;'>"
                "Python · NLTK · Plotly · gTTS · Streamlit</div>",
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# UPLOAD GATE
# ══════════════════════════════════════════════════════════
if not os.path.exists("panchatantra_full_50.csv") and uploaded is None:
    st.info("👈 Upload panchatantra_full_50.csv to begin.")
    st.stop()

# ══════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════
#df = load_and_process(f.read())

# ── Aggregated variables (same as notebook Steps 6–8) ───────────────────────────

# all_text = ' '.join(df['clean_story']) — same as notebook 6.1
all_text     = ' '.join(df['clean_story'])

# all_tokens — flat list of all cleaned tokens
all_tokens   = [w for t in df['tokens'] for w in t]

# emotion_order — same as notebook 6.2
emotion_order = df['emotion'].value_counts().index

# theme_order — same as notebook 6.3
theme_order  = df['theme'].value_counts().index
theme_counts = df['theme'].value_counts()
emo_counts   = df['emotion'].value_counts()

# char_counts — same as notebook 6.4: uses main_character column
char_counts  = df['main_character'].value_counts().head(12)

# action_totals — same as notebook 6.5
action_totals = Counter()
for text in df['story']:
    action_totals.update(count_actions(text))
top_actions = dict(sorted(action_totals.items(), key=lambda x: x[1], reverse=True)[:15])

# all_bigrams — same as notebook 6.5
all_bigrams = []
for text in df['clean_story']:
    tokens = text.split()
    all_bigrams.extend(list(bigrams(tokens)))
top_bigrams   = Counter(all_bigrams).most_common(12)
bigram_labels = [' '.join(pair) for pair, _ in top_bigrams]
bigram_values = [count for _, count in top_bigrams]

# TF-IDF similarity — same as notebook Step 8
tfidf_v = TfidfVectorizer(max_features=200)
tfidf_m = tfidf_v.fit_transform(df['clean_story'])
sim_mat  = cosine_similarity(tfidf_m)

# df_sent — same as notebook 6.8
df_sent    = df[['title','sentiment','emotion']].sort_values('sentiment')
bar_colors = ['#d73027' if s < 0 else '#4dac26' for s in df_sent['sentiment']]

# word_to_theme — same as notebook Step 7 (network section)
word_to_theme = {}
for _, row in df.iterrows():
    theme = str(row['theme']).strip().lower()
    for word in row['tokens']:
        if word not in word_to_theme:
            word_to_theme[word] = theme

# Daily story
if st.session_state.daily is None:
    st.session_state.daily = int(df.sample(1, random_state=42)["story_id"].values[0])

# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown('<div class="hero"><div class="hero-title">📖 PANCHATANTRA NLP</div>'
                '<div class="hero-sub">50 Ancient Indian Fables · Natural Language Processing</div></div>',
                unsafe_allow_html=True)

    # Stat cards
    st.markdown(f"""
    <div class="stat-row">
      <div class="scard"><div class="snum">{len(df)}</div><div class="slabel">Stories</div></div>
      <div class="scard"><div class="snum">{df['theme'].nunique()}</div><div class="slabel">Themes</div></div>
      <div class="scard"><div class="snum">{df['emotion'].nunique()}</div><div class="slabel">Emotions</div></div>
      <div class="scard"><div class="snum">{int(df['word_count'].mean())}</div><div class="slabel">Avg Words</div></div>
      <div class="scard"><div class="snum">{len(set(all_tokens))}</div><div class="slabel">Unique Words</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)

    # Story of the Day
    st.markdown('<div class="sec">✨ Story of the Day</div>', unsafe_allow_html=True)
    daily_row = df[df["story_id"] == st.session_state.daily].iloc[0]
    cd1, cd2  = st.columns([3,1])
    with cd1:
        st.markdown(
            f'<div class="storycard"><div class="stitle">📖 {daily_row["title"]}</div>'
            f'<div class="sttext">{daily_row["story"]}</div>'
            f'<div class="moral"><b>💡 Moral:</b> {daily_row["moral"]}</div></div>',
            unsafe_allow_html=True)
    with cd2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(f'<span class="badge">🎯 {daily_row["theme"].capitalize()}</span><br>'
                    f'<span class="badge">😮 {daily_row["emotion"].capitalize()}</span><br>'
                    f'<span class="badge">🧠 {daily_row["intel_score"]:.2f}%</span><br>'
                    f'<span class="badge">💬 {daily_row["sentiment"]:.3f}</span>',
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔀 New Story"):
            st.session_state.daily = int(df.sample(1)["story_id"].values[0])
            st.rerun()
        if st.button("🛒 Add to Cart"):
            add_to_cart(daily_row["story_id"], df)

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)

    # Word Cloud — same params as notebook 6.1
    st.markdown('<div class="sec">☁️ Word Cloud — Most Frequent Words</div>', unsafe_allow_html=True)
    wc = WordCloud(
        width=1000, height=450,
        background_color='white',
        colormap='plasma',
        max_words=120,
        collocations=False,
        random_state=42
    ).generate(all_text)
    fig_wc, ax_wc = plt.subplots(figsize=(13, 5))
    fig_wc.patch.set_facecolor('#080814')
    ax_wc.imshow(wc, interpolation='bilinear')
    ax_wc.axis('off')
    plt.tight_layout()
    st.pyplot(fig_wc, use_container_width=True)
    plt.close(fig_wc)

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)

    # Emotion + Sunburst
    ca, cb = st.columns(2)
    with ca:
        st.markdown('<div class="sec">😮 Emotion Distribution</div>', unsafe_allow_html=True)
        # Same order as notebook 6.2: emotion_order = df['emotion'].value_counts().index
        f1 = px.bar(x=list(emotion_order), y=emo_counts[emotion_order].values,
                    color=list(emotion_order),
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    labels={"x":"Emotion","y":"Number of Stories"})
        f1.update_layout(**LAY, showlegend=False)
        dax(f1); st.plotly_chart(f1, use_container_width=True)

    with cb:
        st.markdown('<div class="sec">☀️ Theme → Emotion Sunburst</div>', unsafe_allow_html=True)
        sun = df.groupby(['theme','emotion']).size().reset_index(name='count')
        f2  = px.sunburst(sun, path=['theme','emotion'], values='count',
                          color='count', color_continuous_scale='Plasma')
        f2.update_layout(**LAY)
        st.plotly_chart(f2, use_container_width=True)

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)

    # Key insights
    st.markdown('<div class="sec">💡 Key Insights</div>', unsafe_allow_html=True)
    i1,i2,i3,i4 = st.columns(4)
    best = df.loc[df['intel_score'].idxmax()]
    pos  = df.loc[df['sentiment'].idxmax()]
    with i1:
        st.markdown(f'<div class="ibox"><div class="ilabel">Most common theme</div>'
                    f'<div class="ival">{theme_counts.idxmax().capitalize()}</div>'
                    f'{theme_counts.max()} stories</div>', unsafe_allow_html=True)
    with i2:
        st.markdown(f'<div class="ibox"><div class="ilabel">Dominant emotion</div>'
                    f'<div class="ival">{emo_counts.idxmax().capitalize()}</div>'
                    f'{emo_counts.max()} stories</div>', unsafe_allow_html=True)
    with i3:
        st.markdown(f'<div class="ibox"><div class="ilabel">Most intelligent</div>'
                    f'<div class="ival">{best["title"]}</div>'
                    f'{best["intel_score"]:.2f}%</div>', unsafe_allow_html=True)
    with i4:
        st.markdown(f'<div class="ibox"><div class="ilabel">Most positive</div>'
                    f'<div class="ival">{pos["title"]}</div>'
                    f'{pos["sentiment"]:.3f}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA CHARTS  
# ══════════════════════════════════════════════════════════════════════
elif page == "📊  EDA Charts":
    st.markdown('<div class="hero"><div class="hero-title">📊 EDA CHARTS</div>'
                '<div class="hero-sub">Same charts as the Colab notebook — identical data</div></div>',
                unsafe_allow_html=True)

    # 6.2 Emotion distribution — same order as notebook
    st.markdown('<div class="sec">😮 6.2 — Emotion Distribution</div>', unsafe_allow_html=True)
    fe = px.bar(x=list(emotion_order), y=emo_counts[emotion_order].values,
                color=list(emotion_order),
                color_discrete_sequence=px.colors.qualitative.Set2,
                labels={"x":"Emotion","y":"Number of Stories"})
    fe.update_layout(**LAY, showlegend=False)
    dax(fe); st.plotly_chart(fe, use_container_width=True)

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)

    # 6.3 Theme distribution — same order as notebook
    st.markdown('<div class="sec">🎯 6.3 — Most Common Moral Themes</div>', unsafe_allow_html=True)
    ft = px.bar(x=theme_counts[theme_order].values, y=list(theme_order),
                orientation='h',
                color=list(theme_order),
                color_discrete_sequence=px.colors.qualitative.Vivid,
                labels={"x":"Number of Stories","y":"Theme"})
    ft.update_layout(**LAY, showlegend=False, yaxis=dict(autorange='reversed'))
    dax(ft); st.plotly_chart(ft, use_container_width=True)

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)

    # 6.4 Character dominance — uses main_character column, same as fixed notebook
    st.markdown('<div class="sec">🦁 6.4 — Character Dominance</div>', unsafe_allow_html=True)
    st.caption("Main animal per story — same as notebook (uses main_character column, not str.contains)")
    fc = px.bar(x=char_counts.values, y=char_counts.index, orientation='h',
                color=char_counts.values, color_continuous_scale='Plasma',
                labels={"x":"Number of Stories as Main Character","y":"Animal"})
    fc.update_layout(**LAY, coloraxis_showscale=False, yaxis=dict(autorange='reversed'))
    dax(fc); st.plotly_chart(fc, use_container_width=True)

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)

    # 6.5 Action words + Bigrams — same logic as notebook
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sec">⚡ 6.5a — Top Action Words</div>', unsafe_allow_html=True)
        fa = px.bar(x=list(top_actions.keys()), y=list(top_actions.values()),
                    color=list(top_actions.values()), color_continuous_scale='Magma',
                    labels={"x":"Action Word","y":"Total Occurrences"})
        fa.update_layout(**LAY, coloraxis_showscale=False, xaxis_tickangle=-35)
        dax(fa); st.plotly_chart(fa, use_container_width=True)

    with c2:
        st.markdown('<div class="sec">🔗 6.5b — Top Bigrams</div>', unsafe_allow_html=True)
        # Same as notebook: Counter(all_bigrams).most_common(12)
        fb = px.bar(x=bigram_labels, y=bigram_values,
                    color=bigram_values, color_continuous_scale='Plasma',
                    labels={"x":"Bigram (Word Pair)","y":"Frequency"})
        fb.update_layout(**LAY, coloraxis_showscale=False, xaxis_tickangle=-40)
        dax(fb); st.plotly_chart(fb, use_container_width=True)

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)

    # 6.6 Story length — same bins=12 as notebook
    st.markdown('<div class="sec">📏 6.6 — Story Length Distribution</div>', unsafe_allow_html=True)
    fl = px.histogram(df, x='word_count', nbins=12,
                      color_discrete_sequence=['steelblue'],
                      labels={"word_count":"Word Count","count":"Number of Stories"})
    fl.add_vline(x=df['word_count'].mean(), line_dash='dash', line_color='red',
                 annotation_text=f"Mean = {df['word_count'].mean():.0f} words",
                 annotation_font_color='red')
    fl.update_layout(**LAY); dax(fl)
    st.plotly_chart(fl, use_container_width=True)
    st.caption(f"Average: {df['word_count'].mean():.1f} words | "
               f"Min: {df['word_count'].min()} | Max: {df['word_count'].max()}")

# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — NLP ANALYSIS  
# ══════════════════════════════════════════════════════════════════════
elif page == "🧠  NLP Analysis":
    st.markdown('<div class="hero"><div class="hero-title">🧠 NLP ANALYSIS</div>'
                '<div class="hero-sub">Intelligence · Sentiment · Flow</div></div>',
                unsafe_allow_html=True)

    # 6.7 Intelligence score — same as notebook: top 15, horizontal bar
    st.markdown('<div class="sec">🧠 6.7 — Story Intelligence Score (Top 15)</div>',
                unsafe_allow_html=True)
    # Same as notebook: sort descending, head(15), horizontal bar, invert y-axis
    top_intel = df[['title','intel_score','theme']].sort_values('intel_score', ascending=False).head(15)
    fi = px.bar(top_intel, x='intel_score', y='title', orientation='h',
                color='intel_score', color_continuous_scale='YlOrRd',
                labels={"intel_score":"Intelligence Score (%)","title":"Story"})
    fi.update_layout(**LAY, coloraxis_showscale=False, yaxis=dict(autorange='reversed'))
    dax(fi); st.plotly_chart(fi, use_container_width=True)

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)

    # 6.8 Sentiment — same as notebook: sorted, red=negative green=positive
    st.markdown('<div class="sec">💬 6.8 — Sentiment Score per Story</div>',
                unsafe_allow_html=True)
    st.caption("Red = Negative | Green = Positive — same colour logic as notebook")
    fs = px.bar(df_sent, x='title', y='sentiment',
                color='sentiment', color_continuous_scale='RdYlGn',
                hover_data=['emotion'],
                labels={"title":"Story Title","sentiment":"Compound Sentiment Score"})
    fs.update_layout(**LAY, coloraxis_showscale=False, xaxis_tickangle=-75,
                     xaxis=dict(tickfont=dict(size=7)), height=420)
    dax(fs); st.plotly_chart(fs, use_container_width=True)

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)

    # Sentiment flow — same VADER sentence-level logic as dashboard Step 8
    st.markdown('<div class="sec">📈 Sentiment Flow per Story (Sentence-level)</div>',
                unsafe_allow_html=True)
    selected = st.multiselect(
        "Select stories to compare (max 6):",
        df['title'].tolist(),
        default=df['title'].tolist()[:3],
        max_selections=6
    )
    if selected:
        fc2   = px.colors.qualitative.Pastel
        fflow = go.Figure()
        for i, title in enumerate(selected):
            row  = df[df['title'] == title].iloc[0]
            flow = story_sentiment_flow(row['story'])
            fflow.add_trace(go.Scatter(
                x=list(range(len(flow))), y=flow,
                mode='lines+markers', name=title[:25],
                line=dict(color=fc2[i % len(fc2)], width=2.5),
                marker=dict(size=6)
            ))
        fflow.update_layout(**LAY, height=400,
                            xaxis_title='Sentence Index',
                            yaxis_title='Sentiment Score',
                            legend=dict(bgcolor=CARD, bordercolor=GRID))
        dax(fflow); st.plotly_chart(fflow, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 4 — STORY EXPLORER
# ══════════════════════════════════════════════════════════════════════
elif page == "🔍  Story Explorer":
    st.markdown('<div class="hero"><div class="hero-title">🔍 STORY EXPLORER</div>'
                '<div class="hero-sub">Filter · Search · Read · Save</div></div>',
                unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    with f1: sel_th  = st.multiselect("Theme",   sorted(df['theme'].unique()),   default=[])
    with f2: sel_emo = st.multiselect("Emotion", sorted(df['emotion'].unique()), default=[])
    with f3: search  = st.text_input("Search story text", placeholder="e.g. lion, betrayal...")

    sort_by = st.selectbox("Sort by", ["story_id","intel_score","sentiment","word_count"])
    filt    = df.copy()
    if sel_th:  filt = filt[filt['theme'].isin(sel_th)]
    if sel_emo: filt = filt[filt['emotion'].isin(sel_emo)]
    if search:  filt = filt[filt['story'].str.lower().str.contains(search.lower())]
    filt = filt.sort_values(sort_by, ascending=(sort_by == 'story_id'))

    st.caption(f"Showing {len(filt)} of {len(df)} stories")
    st.markdown("<div class='div'></div>", unsafe_allow_html=True)

    for _, row in filt.iterrows():
        emoji = "💚" if row['sentiment']>0.1 else ("❤️" if row['sentiment']<-0.1 else "🟡")
        st.markdown(
            f'<div class="storycard">'
            f'<div class="stitle">#{int(row["story_id"])} · {row["title"]}</div>'
            f'<div class="sttext">{row["story"]}</div><br>'
            f'<span class="badge">🎯 {row["theme"].capitalize()}</span>'
            f'<span class="badge">😮 {row["emotion"].capitalize()}</span>'
            f'<span class="badge">🧠 {row["intel_score"]:.2f}%</span>'
            f'<span class="badge">{emoji} {row["sentiment"]:.3f}</span>'
            f'<span class="badge">📏 {row["word_count"]} words</span>'
            f'<div class="moral"><b>💡 Moral:</b> {row["moral"]}</div>'
            f'</div>', unsafe_allow_html=True
        )
        if st.button("🛒 Add to Cart", key=f"exp_{row['story_id']}"):
            add_to_cart(row['story_id'], df)

# ══════════════════════════════════════════════════════════════════════
# PAGE 5 — VOICE READER
# ══════════════════════════════════════════════════════════════════════
elif page == "🎙️  Voice Reader":
    st.markdown('<div class="hero"><div class="hero-title">🎙️ VOICE READER</div>'
                '<div class="hero-sub">Listen to Panchatantra Stories</div></div>',
                unsafe_allow_html=True)

    sel_story = st.selectbox("Choose a story:", df['title'].tolist())
    row       = df[df['title'] == sel_story].iloc[0]

    c1, c2 = st.columns([3,1])
    with c1:
        st.markdown(
            f'<div class="storycard"><div class="stitle">📖 {row["title"]}</div>'
            f'<div class="sttext">{row["story"]}</div>'
            f'<div class="moral"><b>💡 Moral:</b> {row["moral"]}</div></div>',
            unsafe_allow_html=True)

    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec">🎛️ Controls</div>', unsafe_allow_html=True)
        read_what = st.radio("Read:", ["Full Story","Moral Only","Title + Moral"])
        if read_what == "Full Story":
            read_text = f"{row['title']}. {row['story']} The moral is: {row['moral']}"
        elif read_what == "Moral Only":
            read_text = f"The moral of {row['title']} is: {row['moral']}"
        else:
            read_text = f"{row['title']}. {row['moral']}"

        if st.button("▶️ Play", use_container_width=True):
            with st.spinner("Generating audio..."):
                audio_html = tts_audio(read_text)
            st.markdown(audio_html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<span class="badge">🎯 {row["theme"].capitalize()}</span><br>'
                    f'<span class="badge">📏 {row["word_count"]} words</span>',
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🛒 Add to Cart", use_container_width=True):
            add_to_cart(row['story_id'], df)

# ══════════════════════════════════════════════════════════════════════
# PAGE 6 — STORY QUIZ
# ══════════════════════════════════════════════════════════════════════
elif page == "🎮  Story Quiz":
    st.markdown('<div class="hero"><div class="hero-title">🎮 STORY QUIZ</div>'
                '<div class="hero-sub">Guess the Moral · Test Your Knowledge</div></div>',
                unsafe_allow_html=True)

    TOTAL_Q = 10

    if st.session_state.quiz_done:
        sc = st.session_state.quiz_score
        st.markdown(f'<div class="qcard" style="text-align:center;">'
                    f'<div style="font-family:Cinzel,serif;font-size:3rem;color:#c084fc;">{sc}/{TOTAL_Q}</div>'
                    f'<div style="color:#a594cc;margin-top:8px;font-size:1.05rem;">'
                    f'{"🏆 Excellent!" if sc>=8 else "👍 Good effort!" if sc>=5 else "📖 Keep reading!"}'
                    f'</div></div>', unsafe_allow_html=True)
        if st.button("🔄 Play Again", use_container_width=True):
            st.session_state.quiz_idx  = 0
            st.session_state.quiz_score= 0
            st.session_state.quiz_done = False
            st.session_state.quiz_ans  = None
            st.rerun()
    else:
        random.seed(st.session_state.quiz_idx + 42)
        sample  = df.sample(frac=1, random_state=st.session_state.quiz_idx+42).reset_index(drop=True)
        q_row   = sample.iloc[st.session_state.quiz_idx % len(sample)]
        # Wrong answers from DIFFERENT theme to avoid confusion
        wrong   = df[(df['story_id'] != q_row['story_id']) &
                     (df['theme'] != q_row['theme'])]['moral'].sample(3, random_state=st.session_state.quiz_idx+1).tolist()
        options = wrong + [q_row['moral']]
        random.shuffle(options)

        st.progress((st.session_state.quiz_idx)/TOTAL_Q,
                    text=f"Question {st.session_state.quiz_idx+1} of {TOTAL_Q} · Score: {st.session_state.quiz_score}")

        st.markdown(f'<div class="qcard"><div class="qq">📖 Story: <b style="color:#c084fc">'
                    f'{q_row["title"]}</b><br><br>'
                    f'<span style="color:#a594cc;font-style:italic;">"{q_row["story"][:200]}..."</span>'
                    f'<br><br>❓ What is the moral of this story?</div></div>',
                    unsafe_allow_html=True)

        answered = st.session_state.quiz_ans is not None
        for i, opt in enumerate(options):
            if not answered:
                if st.button(f"{'ABCD'[i]}. {opt}", key=f"opt_{i}", use_container_width=True):
                    st.session_state.quiz_ans = opt
                    if opt == q_row['moral']:
                        st.session_state.quiz_score += 1
                    st.rerun()
            else:
                if opt == q_row['moral']:
                    st.success(f"✅ {'ABCD'[i]}. {opt}")
                elif opt == st.session_state.quiz_ans:
                    st.error(f"❌ {'ABCD'[i]}. {opt}")
                else:
                    st.write(f"{'ABCD'[i]}. {opt}")

        if answered:
            nc1, nc2 = st.columns(2)
            with nc1:
                if st.button("➡️ Next Question", use_container_width=True):
                    st.session_state.quiz_idx += 1
                    st.session_state.quiz_ans  = None
                    if st.session_state.quiz_idx >= TOTAL_Q:
                        st.session_state.quiz_done = True
                    st.rerun()
            with nc2:
                if st.button("🛒 Save this Story", use_container_width=True):
                    add_to_cart(q_row['story_id'], df)

# ══════════════════════════════════════════════════════════════════════
# PAGE 7 — RECOMMENDER (uses TF-IDF sim_mat from notebook )
# ══════════════════════════════════════════════════════════════════════
elif page == "🤖  Recommender":
    st.markdown('<div class="hero"><div class="hero-title">🤖 RECOMMENDER</div>'
                '<div class="hero-sub">Find Stories Similar to Ones You Love</div></div>',
                unsafe_allow_html=True)

    sel  = st.selectbox("Select a story you liked:", df['title'].tolist())
    idx  = df[df['title'] == sel].index[0]
    scores = sorted(enumerate(sim_mat[idx]), key=lambda x: x[1], reverse=True)
    scores = [(i,s) for i,s in scores if i != idx][:5]

    liked = df.iloc[idx]
    st.markdown(f'<div class="storycard"><div class="stitle">✅ You liked: {liked["title"]}</div>'
                f'<span class="badge">🎯 {liked["theme"].capitalize()}</span>'
                f'<span class="badge">😮 {liked["emotion"].capitalize()}</span>'
                f'<div class="moral"><b>💡 Moral:</b> {liked["moral"]}</div></div>',
                unsafe_allow_html=True)

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)
    st.markdown('<div class="sec">🎯 Top 5 Similar Stories</div>', unsafe_allow_html=True)

    for rank, (i, score) in enumerate(scores, 1):
        row = df.iloc[i]
        cr1, cr2 = st.columns([4,1])
        with cr1:
            st.markdown(
                f'<div class="storycard"><div class="stitle">#{rank} · {row["title"]}'
                f' <span style="color:#7a6a9a;font-size:.8rem;">({score:.0%} match)</span></div>'
                f'<div class="sttext">{row["story"][:300]}...</div><br>'
                f'<span class="badge">🎯 {row["theme"].capitalize()}</span>'
                f'<span class="badge">😮 {row["emotion"].capitalize()}</span>'
                f'<span class="badge">🧠 {row["intel_score"]:.2f}%</span>'
                f'<div class="moral"><b>💡 Moral:</b> {row["moral"]}</div></div>',
                unsafe_allow_html=True)
        with cr2:
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            if st.button("🛒 Save", key=f"rec_{i}", use_container_width=True):
                add_to_cart(row['story_id'], df)

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)
    st.markdown('<div class="sec">🔍 Moral Keyword Matcher</div>', unsafe_allow_html=True)
    keyword = st.text_input("Search morals by keyword:", placeholder="e.g. trust, courage, wisdom...")
    if keyword:
        matches = df[df['moral'].str.lower().str.contains(keyword.lower())]
        if len(matches):
            st.success(f"Found {len(matches)} stories with '{keyword}' in the moral:")
            for _, row in matches.iterrows():
                st.markdown(f'<div class="storycard"><div class="stitle">{row["title"]}</div>'
                            f'<div class="moral"><b>💡 Moral:</b> {row["moral"]}</div></div>',
                            unsafe_allow_html=True)
                if st.button("🛒 Add", key=f"kw_{row['story_id']}"):
                    add_to_cart(row['story_id'], df)
        else:
            st.warning(f"No morals found with '{keyword}'")

# ══════════════════════════════════════════════════════════════════════
# PAGE 8 — FLASHCARD MODE
# ══════════════════════════════════════════════════════════════════════
elif page == "📖  Flashcard Mode":
    st.markdown('<div class="hero"><div class="hero-title">📖 FLASHCARD MODE</div>'
                '<div class="hero-sub">Read the Moral · Guess the Story</div></div>',
                unsafe_allow_html=True)

    total = len(df)
    idx   = st.session_state.fc_idx % total
    row   = df.iloc[idx]

    st.progress((idx+1)/total, text=f"Card {idx+1} of {total}")

    st.markdown(f'<div class="fcard"><div class="flabel">💡 Which story has this moral?</div>'
                f'<div class="ftext">"{row["moral"]}"</div></div>',
                unsafe_allow_html=True)

    ff1, ff2, ff3, ff4 = st.columns(4)
    with ff1:
        if st.button("🔍 Reveal", use_container_width=True):
            st.session_state.fc_flipped = True
    with ff2:
        if st.button("➡️ Next", use_container_width=True):
            st.session_state.fc_idx    += 1
            st.session_state.fc_flipped = False
            st.rerun()
    with ff3:
        if st.button("⬅️ Prev", use_container_width=True):
            st.session_state.fc_idx    = max(0, st.session_state.fc_idx-1)
            st.session_state.fc_flipped = False
            st.rerun()
    with ff4:
        if st.button("🔀 Random", use_container_width=True):
            st.session_state.fc_idx    = random.randint(0, total-1)
            st.session_state.fc_flipped = False
            st.rerun()

    if st.session_state.fc_flipped:
        st.markdown(
            f'<div class="fcard" style="margin-top:12px;">'
            f'<div class="flabel">✅ Answer</div>'
            f'<div class="fanswer">{row["title"]}</div><br>'
            f'<span class="badge">🎯 {row["theme"].capitalize()}</span>'
            f'<span class="badge">😮 {row["emotion"].capitalize()}</span>'
            f'<div style="margin-top:12px;color:#a594cc;font-style:italic;font-size:.94rem;">'
            f'"{row["story"][:150]}..."</div></div>',
            unsafe_allow_html=True)
        if st.button("🛒 Save to Cart", use_container_width=True):
            add_to_cart(row['story_id'], df)

# ══════════════════════════════════════════════════════════════════════
# PAGE 9 — DEEP ANALYSIS
# ══════════════════════════════════════════════════════════════════════
elif page == "🔬  Deep Analysis":
    st.markdown('<div class="hero"><div class="hero-title">🔬 DEEP ANALYSIS</div>'
                '<div class="hero-sub">Emotion Arc · Theme Wordclouds · Character Map</div></div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🎭 Emotion Arc", "☁️ Theme Wordclouds", "🗺️ Character Map"])

    # Tab 1 — Emotion Arc
    with tab1:
        st.markdown('<div class="sec">🎭 Emotion Arc — Which emotions lead to positive endings?</div>',
                    unsafe_allow_html=True)

        def end_sentiment(story):
            sia   = SentimentIntensityAnalyzer()
            sents = sent_tokenize(str(story))
            if len(sents) < 3: return 0
            return sia.polarity_scores(" ".join(sents[-3:]))['compound']

        df['end_sentiment'] = df['story'].apply(end_sentiment)
        df['outcome']       = df['end_sentiment'].apply(
            lambda x: "✅ Positive Ending" if x>0.05
            else ("❌ Negative Ending" if x<-0.05 else "➡️ Neutral Ending")
        )

        arc_df = df.groupby(['emotion','outcome']).size().reset_index(name='count')
        fa2 = px.bar(arc_df, x='emotion', y='count', color='outcome', barmode='group',
                     color_discrete_map={
                         "✅ Positive Ending":"#4ade80",
                         "❌ Negative Ending":"#f87171",
                         "➡️ Neutral Ending":"#94a3b8"
                     }, labels={"emotion":"Starting Emotion","count":"Stories"})
        fa2.update_layout(**LAY, xaxis_tickangle=-35, height=400)
        dax(fa2); st.plotly_chart(fa2, use_container_width=True)

        oc = df['outcome'].value_counts()
        o1,o2,o3 = st.columns(3)
        for col,label in zip([o1,o2,o3],
                             ["✅ Positive Ending","❌ Negative Ending","➡️ Neutral Ending"]):
            count = oc.get(label, 0)
            with col:
                st.markdown(f'<div class="ibox"><div class="ilabel">{label}</div>'
                            f'<div class="ival">{count} stories</div>'
                            f'{count/len(df)*100:.0f}% of all stories</div>',
                            unsafe_allow_html=True)

    # Tab 2 — Per-theme wordclouds (uses clean_story, same as notebook 6.1)
    with tab2:
        st.markdown('<div class="sec">☁️ Per-Theme Word Cloud</div>', unsafe_allow_html=True)
        sel_theme  = st.selectbox("Select a theme:", sorted(df['theme'].unique()))
        theme_text = ' '.join(df[df['theme'] == sel_theme]['clean_story'])

        if theme_text.strip():
            twc = WordCloud(
                width=1000, height=380,
                background_color='white',
                colormap='plasma',
                max_words=80,
                collocations=False,
                random_state=42
            ).generate(theme_text)
            fig_twc, ax_twc = plt.subplots(figsize=(12, 4))
            fig_twc.patch.set_facecolor('#080814')
            ax_twc.imshow(twc, interpolation='bilinear')
            ax_twc.axis('off')
            ax_twc.set_title(f'Theme: {sel_theme.capitalize()}',
                             color='#c084fc', fontsize=13, pad=10)
            plt.tight_layout()
            st.pyplot(fig_twc, use_container_width=True)
            plt.close(fig_twc)

            theme_stories = df[df['theme']==sel_theme][['title','emotion','intel_score','sentiment','word_count']]
            st.markdown(f"**{len(theme_stories)} stories in '{sel_theme}' theme:**")
            st.dataframe(theme_stories.set_index('title'), use_container_width=True)

    # Tab 3 — Character Journey Map
    with tab3:
        st.markdown('<div class="sec">🗺️ Character Journey Map</div>', unsafe_allow_html=True)
        sel_animal   = st.selectbox("Select a character:", sorted(ANIMALS))
        char_stories = df[df['story'].str.lower().str.contains(sel_animal)].copy()

        if len(char_stories) == 0:
            st.warning(f"No stories found with '{sel_animal}'")
        else:
            st.success(f"**{sel_animal.capitalize()}** appears in **{len(char_stories)}** stories")
            char_stories['story_num'] = range(1, len(char_stories)+1)

            fj = go.Figure()
            fj.add_trace(go.Scatter(
                x=char_stories['story_num'], y=char_stories['sentiment'],
                mode='lines+markers+text',
                text=char_stories['title'].str[:18],
                textposition='top center',
                textfont=dict(size=8, color='#c084fc'),
                line=dict(color=ACCENT, width=2.5),
                marker=dict(size=13, color=char_stories['sentiment'],
                            colorscale='RdYlGn', showscale=True,
                            colorbar=dict(title='Sentiment',
                                         tickfont=dict(color=TEXT))),
                hovertemplate="<b>%{text}</b><br>Sentiment: %{y:.3f}<extra></extra>"
            ))
            fj.update_layout(**LAY, height=420,
                             xaxis_title='Story Appearance',
                             yaxis_title='Story Sentiment',
                             xaxis=dict(tickmode='linear'))
            dax(fj); st.plotly_chart(fj, use_container_width=True)

            for _, row in char_stories.iterrows():
                emoji = "💚" if row['sentiment']>0.1 else ("❤️" if row['sentiment']<-0.1 else "🟡")
                st.markdown(
                    f'<div class="storycard"><div class="stitle">{row["title"]}</div>'
                    f'<div class="sttext">{row["story"][:220]}...</div><br>'
                    f'<span class="badge">🎯 {row["theme"].capitalize()}</span>'
                    f'<span class="badge">{emoji} {row["sentiment"]:.3f}</span>'
                    f'<div class="moral"><b>💡 Moral:</b> {row["moral"]}</div></div>',
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 10 — MY CART
# ══════════════════════════════════════════════════════════════════════
elif page.startswith("🛒"):
    st.markdown('<div class="hero"><div class="hero-title">🛒 MY READING CART</div>'
                '<div class="hero-sub">Your Saved Panchatantra Stories</div></div>',
                unsafe_allow_html=True)

    if not st.session_state.cart:
        st.markdown('<div class="fcard"><div class="flabel">Your cart is empty</div>'
                    '<div class="ftext">Browse stories and click 🛒 Add to Cart</div></div>',
                    unsafe_allow_html=True)
    else:
        themes_c   = [s['theme'] for s in st.session_state.cart]
        most_theme = Counter(themes_c).most_common(1)[0][0] if themes_c else '—'

        st.markdown(f"""<div class="stat-row" style="grid-template-columns:repeat(3,1fr);">
          <div class="scard"><div class="snum">{len(st.session_state.cart)}</div><div class="slabel">Saved Stories</div></div>
          <div class="scard"><div class="snum">{len(set(themes_c))}</div><div class="slabel">Unique Themes</div></div>
          <div class="scard"><div class="snum">{most_theme.capitalize()}</div><div class="slabel">Top Theme</div></div>
        </div>""", unsafe_allow_html=True)

        # Export
        export_text = ""
        for s in st.session_state.cart:
            export_text += (f"{'='*60}\n{s['title']}\n{'='*60}\n"
                            f"{s['story']}\n\nMoral: {s['moral']}\n"
                            f"Theme: {s['theme']} | Emotion: {s['emotion']}\n\n")
        st.download_button("📥 Export as Text File", data=export_text,
                           file_name="my_panchatantra_stories.txt",
                           mime="text/plain", use_container_width=True)

        st.markdown("<div class='div'></div>", unsafe_allow_html=True)

        to_remove = None
        for i, s in enumerate(st.session_state.cart):
            cc1, cc2 = st.columns([5,1])
            with cc1:
                st.markdown(
                    f'<div class="storycard"><div class="stitle">#{i+1} · {s["title"]}</div>'
                    f'<div class="sttext">{s["story"]}</div><br>'
                    f'<span class="badge">🎯 {s["theme"].capitalize()}</span>'
                    f'<span class="badge">😮 {s["emotion"].capitalize()}</span>'
                    f'<div class="moral"><b>💡 Moral:</b> {s["moral"]}</div></div>',
                    unsafe_allow_html=True)
            with cc2:
                st.markdown("<br><br><br>", unsafe_allow_html=True)
                if st.button("🗑️ Remove", key=f"rm_{i}", use_container_width=True):
                    to_remove = i

        if to_remove is not None:
            st.session_state.cart.pop(to_remove)
            st.rerun()

        st.markdown("<div class='div'></div>", unsafe_allow_html=True)
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.cart = []
            st.rerun()
