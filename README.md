
<div align="center">


# 📖 Panchatantra Stories — NLP Analysis & Interactive Dashboard

### Applying Modern NLP Techniques to 2500-Year-Old Ancient Indian Wisdom  


<br>

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-85EA2D?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![gTTS](https://img.shields.io/badge/Text--to--Speech-gTTS-blue?style=for-the-badge)

</div>
<div align="center">

## 🚀 Explore the Project

<table>
<tr>

<td align="center" width="300" height="220">

### 🌐 Live Application  
<br>
Experience the full interactive dashboard  
<br><br>

<a href="YOUR_STREAMLIT_LINK">
<img src="https://img.icons8.com/color/96/streamlit.png" width="60"/>
<br><br>
<b>▶ Launch App</b>
</a>

</td>

<td align="center" width="300" height="220">

### 📓 Notebook (Colab)  
<br>
Explore the NLP pipeline & implementation  
<br><br>

<a href="YOUR_COLAB_LINK">
<img src="https://img.icons8.com/color/96/google-colab.png" width="60"/>
<br><br>
<b>📊 Open Notebook</b>
</a>

</td>

</tr>
</table>

</div>

## 📌 Project Overview

This project performs a **complete end-to-end Natural Language Processing (NLP) analysis** on **50 stories from the Panchatantra** — one of the world's oldest collections of moral fables, originating in ancient India around 200 BCE.

The goal is to apply real-world NLP techniques to a rich literary dataset and present findings through an **interactive multi-page Streamlit application** — complete with voice narration, a story quiz, and intelligent recommendations.

> *"The Panchatantra is not just a collection of stories — it is a structured system of wisdom. This project decodes that system using data."*

---
## 🎯 What Makes This Project Unique

| Feature | Description |
|--------|------------|
| 🎙️ Voice Reader | Listen to any story using Text-to-Speech (gTTS) |
| 🎮 Story Quiz | 10-question MCQ game — guess the moral from the story |
| 🤖 Story Recommender | TF-IDF cosine similarity finds stories similar to your favourite |
| 📖 Flashcard Mode | Read the moral → flip to reveal the story title |
| 🔬 Deep Analysis | Emotion arc, per-theme wordclouds, character journey map |
| 🛒 Reading Cart | Save favourite stories and export as a text file |
| 🔍 Story Explorer | Filter by theme, emotion, search text, sort by any metric |
| ✨ Story of the Day | Random featured story on the homepage every session |

---

## 🗂️ Project Structure

```bash
panchatantra-nlp-analysis/
│
├── 📄 panchatantra_app.py              ← Streamlit app (10 pages, 28+ features)
├── 📓 panchatantra_nlp_analysis.ipynb  ← Google Colab NLP notebook
├── 📊 panchatantra_full_50.csv         ← Dataset (50 stories)
├── 📄 requirements.txt                 ← Python dependencies
└── 📄 README.md                        ← This file
```

## 📊 Dataset

The dataset was custom built for this project — not taken from any existing source.

## 📑 Columns
| Column   | Type   | Description                                                |
| -------- | ------ | ---------------------------------------------------------- |
| story_id | int    | Unique story number (1–50)                                 |
| title    | string | Story title                                                |
| story    | string | Full story paragraph (~200 words, realistic noise added)   |
| moral    | string | The lesson of the story                                    |
| emotion  | string | Emotion label (fear, joy, anger, betrayal, surprise, etc.) |
| theme    | string | Moral theme (trust, intelligence, cooperation, etc.)       |

### 📈 Dataset stats

- 📚 50 stories · 22 unique themes · 12 emotion types  
- 📝 Average ~200 words per story  
- ✨ Realistic noise: apostrophes, contractions, commas, dashes  
- 📖 Stories written as natural paragraphs — not simple sentences  

## 🔍 NLP Pipeline (Google Colab Notebook)

```text
Raw Text
   │
   ▼
Step 1 ── Text Cleaning        (lowercase, remove punctuation, regex)
   │
   ▼
Step 2 ── Stopword Removal     (NLTK + 40 custom folk-tale stopwords)
   │
   ▼
Step 3 ── Tokenization         (word_tokenize + sent_tokenize)
   │
   ▼
Step 4 ── Lemmatization        (WordNetLemmatizer)
   │
   ▼
Step 5 ── Bigram Extraction    (NLTK bigrams)
   │
   ▼
Step 6 ── Feature Engineering  (sentiment, intel_score, main_character)
   │
   ▼
Step 7 ── Visualisation        (8 charts saved as PNG)
   │
   ▼
Step 8 ── Interactive Dashboard (Plotly + HTML export)
   ▼
Step 8 ── Interactive Dashboard (Plotly + HTML export)
```

## 📊 Analyses Performed

| # | Analysis | Method |
|--|----------|--------|
| 1 | Text Cleaning | Regex + NLTK stopwords |
| 2 | Tokenization | word_tokenize, sent_tokenize |
| 3 | Lemmatization | WordNetLemmatizer |
| 4 | Bigram Extraction | nltk.bigrams |
| 5 | Word Cloud | WordCloud library |
| 6 | Emotion Distribution | Dataset emotion column |
| 7 | Moral Theme Analysis | Dataset theme column |
| 8 | Character Dominance | main_character detection |
| 9 | Top Action Words | Custom verb counting |
| 10 | Story Length | Word count histogram |
| 11 | Intelligence Score | Smart-word density formula |
| 12 | Sentiment Analysis | VADER SentimentIntensityAnalyzer |
| 13 | Sentiment Flow | Sentence-level VADER scoring |
| 14 | Story Similarity | TF-IDF + Cosine Similarity |
| 15 | Emotion Arc | Ending sentiment detection |
| 16 | Theme Wordclouds | Per-theme clean_story join |
| 17 | Character Journey | Animal-filtered story sentiment timeline |
| 18 | Moral Keyword Matcher | String search on moral column |

---

## 🖥️ Streamlit App — 10 Pages

| Page | Features |
|------|---------|
| 🏠 Overview | Stat cards · Word Cloud · Emotion chart · Sunburst · Story of the Day · Key Insights |
| 📊 EDA Charts | Themes · Character Dominance · Action Words · Bigrams · Length Distribution |
| 🧠 NLP Analysis | Sentiment bar · Intelligence Score · Sentiment Flow selector |
| 🔍 Story Explorer | Filter by theme/emotion · Search text · Sort · Full story cards · Cart |
| 🎙️ Voice Reader | TTS audio · Full/Moral/Title read modes · Cart |
| 🎮 Story Quiz | 10 MCQ questions · Score tracker · Wrong answers from different themes |
| 🤖 Recommender | TF-IDF similarity · Top 5 matches · Moral keyword matcher · Cart |
| 📖 Flashcard Mode | Show moral → flip → reveal title · Next/Prev/Random · Cart |
| 🔬 Deep Analysis | Emotion Arc · Per-theme Wordclouds · Character Journey Map |
| 🛒 My Cart | View saved stories · Stats · Export as .txt · Remove / Clear all |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|--------|
| Python 3.10+ | Core language |
| Pandas | Data manipulation |
| NLTK | Tokenization, lemmatization, sentiment |
| Scikit-learn | TF-IDF vectoriser, cosine similarity |
| Plotly | Interactive charts |
| Matplotlib | Static charts, wordcloud rendering |
| WordCloud | Word cloud generation |
| gTTS | Text-to-Speech audio |
| Streamlit | Web application framework |
| NetworkX | Word co-occurrence network (Colab) |

---

## 🚀 Run Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/YourUsername/panchatantra-nlp-analysis.git
cd panchatantra-nlp-analysis
```
### 2️⃣ Install dependencies
``` bash
pip install -r requirements.txt
```
### 3️⃣ Run the Streamlit app
``` bash
streamlit run panchatantra_app.py
```
### 4️⃣ Open in browser
``` bash
http://localhost:8501
```
## 📓 Run the Colab Notebook

Follow these steps to explore the NLP pipeline:

1. Click the **Open in Colab** badge at the top of this README  
2. Upload `panchatantra_full_50.csv` when prompted  
3. Run all cells → **Runtime → Run all**  
4. The dashboard will auto-download as `panchatantra_dashboard.html`  

---

## 🧠 Key Findings

- 😨 **Fear** is the most dominant emotion across Panchatantra stories  
- 🤝 **Trust** is the most recurring moral theme  
- 🦁 **Lion** and 🐍 **Snake** are the most frequently featured characters  
- 🧠 Stories with **intelligence themes** score highest on the custom intelligence metric  
- 📈 Around **60% of stories** end with positive sentiment regardless of starting emotion  
- 📝 Average story length is **~200 words** with low variance — consistent narrative style  

---

## 🎓 Skills Demonstrated
✅ End-to-end NLP pipeline design  
✅ Text preprocessing — cleaning, tokenization, lemmatization, bigrams  
✅ Sentiment analysis using VADER  
✅ TF-IDF vectorisation and cosine similarity  
✅ Custom dataset creation with realistic noise injection  
✅ Feature engineering — intelligence score, character detection  
✅ Interactive data visualisation with Plotly  
✅ Multi-page Streamlit application development  
✅ Session state management in Streamlit  
✅ Streamlit Cloud deployment  
✅ Text-to-Speech integration using gTTS 

---

## 👤 Author



### Munnuri Vaishnavi

📧 your.email@example.com  

🔗 [LinkedIn](#)  

🐙 [GitHub](#)

<div align="center">

### 𝐁𝐮𝐢𝐥𝐭 𝐰𝐢𝐭𝐡 ❤️ 𝐮𝐬𝐢𝐧𝐠 𝐏𝐲𝐭𝐡𝐨𝐧 · 𝐍𝐋𝐓𝐊 · 𝐏𝐥𝐨𝐭𝐥𝐲 · 𝐒𝐭𝐫𝐞𝐚𝐦𝐥𝐢𝐭  
⭐ *If you found this project useful, please give it a star!* ⭐  

</div>
