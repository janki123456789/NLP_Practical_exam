<div align="center">

# ✈️ Airline Tweet Analytics
### From Raw Text to a Streamlit App — NLP Practical Exam (Set B)

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Gensim](https://img.shields.io/badge/Gensim-Word2Vec-red)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)

**Classify airline tweets as Negative 😠 · Neutral 😐 · Positive 😊**
</div>
---

## 📌 Project Overview

Passengers post thousands of tweets every day — praising the crew, complaining about delays, shouting about lost baggage. A social media team cannot read them all.

This project builds a small, working NLP system that **cleans raw tweets → converts them into numbers → learns from them → predicts sentiment**, all wrapped in a simple **Streamlit web app**.

---
### Dataset Link

- [Tweets.csv](Tweets.csv)

## 🔗 Important Links

| Resource | Link |
|---|---|
| 📊 Dataset (Kaggle – Twitter US Airline Sentiment) | https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment |
| 🚀 Deployed Streamlit link | STREAMLIT_APP_URL |

> If deployment is blocked on the lab network, replace the app link with: **Deployment blocked — local demo in video**

---

## 🗂️ Dataset

- **Source:** Kaggle – *Twitter US Airline Sentiment* (`Tweets.csv`)
- **Size:** 14,640 tweets about 6 US airlines
- **Columns used:** `airline`, `airline_sentiment`, `negativereason`, `text`
- **Class split:** Negative 9,178 (62.69%) · Neutral 3,099 · Positive 2,363

---

## 🧭 Steps Covered

| Step | Topic | What was done |
|:---:|---|---|
| 1 | **NLP Fundamentals, Preprocessing & RegEx** | Exploration, mention/hashtag/URL extraction, `clean_text()` pipeline, stemming vs lemmatization, word frequency |
| 2 | **Feature Extraction** | Bag-of-Words, TF-IDF and N-Grams on a tiny corpus and on real tweets |
| 3 | **Word Embeddings** | Word2Vec CBoW & Skip-gram, similarity, PCA visualisation |
| 4 | **Text Classification** | 3-class sentiment with 4 feature methods, macro-F1 comparison |
| 5 | **Different NLP Approaches & POS Tagging** | Heuristic vs TextBlob vs ML; nouns from negative tweets |
| 6 | **Streamlit App** | Interactive sentiment predictor with probability chart |
| 7 | **Comparison & Reflection** | Summary table, reflection, GitHub submission |

---

## 📊 Results Summary

### 🔍 Step 1 & 2 — Exploration and Features
| Finding | Result |
|---|---|
| Negative tweets | **62.69%** (class imbalance warning) |
| Highest negative share | **US Airways – 77.69%** |
| Avg. tweet length (words) | Negative 19.69 · Neutral 14.41 · Positive 14.01 |
| Top @mentions | `@united` (3,765), `@AmericanAir` (2,926), `@USAirways` (2,907), `@SouthwestAir` (2,409), `@JetBlue` (2,198) |
| Top hashtags | `#DestinationDragons` (75), `#fail` (57), `#jetblue` (35), `#UnitedAirlines` (35), `#customerservice` (34) |
| Tweets with URL | 1,173 (38.19% negative) |
| Tweets with flight number | 606 |

| Method | Features | Sparsity % |
|---|:---:|:---:|
| Bag-of-Words | 3,000 | 99.74 |
| TF-IDF | 3,000 | 99.74 |
| Bigrams (negative tweets) | 3,000 | 99.92 |

### 🧬 Step 3 — Word2Vec
| Item | Value |
|---|---|
| Avg. tokens per tweet | 8.88 |
| Unique words | 10,135 |
| Settings | `vector_size=100, window=5, min_count=3, epochs=15, seed=42` |
| Vocabulary size (CBoW / Skip-gram) | _fill after re-running with the tweet tokens_ |

### 🎯 Step 4 — Sentiment Classification (test set = 2,928 tweets)
| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|:---:|:---:|:---:|:---:|
| Majority Baseline | 0.6267 | 0.2089 | 0.3333 | 0.2568 |
| 🏆 **TF-IDF + N-Grams (1,2)** | **0.7640** | 0.7066 | 0.7343 | **0.7162** |
| Bag-of-Words | 0.7497 | 0.6936 | 0.7293 | 0.7053 |
| TF-IDF | 0.7493 | 0.6938 | 0.7277 | 0.7051 |
| Averaged Word2Vec | _update after Word2Vec re-run_ | | | |

**Best model – per-class report**

| Class | Precision | Recall | F1 |
|---|:---:|:---:|:---:|
| Negative | 0.90 | 0.80 | 0.85 |
| Neutral | 0.54 | 0.70 | 0.61 |
| Positive | 0.68 | 0.70 | 0.69 |

🔑 **Strongest words** — Negative: `hour`, `not`, `worst`, `hold`, `no`, `hr`, `delayed`, `bag` · Positive: `great`, `thanks`, `thank`, `awesome`, `love`, `amazing`, `best`, `thx`

### 🔍 Step 5 — Three Approaches (first 1,000 test tweets)
| Approach | Needs training data? | Accuracy | Macro F1 |
|---|:---:|:---:|:---:|
| Heuristic (rule-based) | No | 0.467 | 0.5069 |
| TextBlob (API-style) | No | 0.466 | 0.4671 |
| **Machine Learning** | Yes | **0.781** | **0.7425** |

**POS tagging (500 negative tweets)** — top nouns: `flight`, `service`, `customer`, `plane`, `time`, `hold`, `hour`.
Top `negativereason`: Customer Service Issue (158), Late Flight (87), Can't Tell (62), Bad Flight (42), Cancelled Flight (41).
➡️ Nouns such as *service/customer* and *hour/time/hold* match the *Customer Service* and *Late Flight* themes.

---

## 🖥️ Streamlit App

The app lets a team member paste a tweet and instantly see:

- 😠 / 😐 / 😊 **predicted sentiment**
- 📊 **probability bar chart** for all 3 classes
- 🔎 **RegEx panel** listing `@mentions` and `#hashtags`
- 📌 Sidebar with **3 example tweets** and an *About* note
- ⚠️ `st.warning('Please enter a tweet')` for empty input


## 📁 Repository Structure

```
nlp-practical-exam-set-b/
├── NLP_Practical_Exam_SetB.ipynb   # Steps 1–5 (fully executed)
├── app.py                          # Streamlit app
├── text_utils.py                   # clean_text() helper used by the app
├── sentiment_pipeline.pkl          # Best saved pipeline (TF-IDF + N-Grams + LogReg)
├── word2vec_skipgram.model         # Skip-gram model (+ any .npy files)
├── requirements.txt
├── summary_report.md
├── README.md
└── outputs/                        # Plots + Streamlit screenshots
```

---

## ⚙️ How to Run

**1️⃣ Clone the repository**
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/nlp-practical-exam-set-b.git
cd nlp-practical-exam-set-b
```

**2️⃣ Install requirements**
```bash
pip install -r requirements.txt
```

**3️⃣ Download NLTK data (once)**
```python
import nltk
for p in ['punkt', 'punkt_tab', 'stopwords', 'wordnet',
          'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
    nltk.download(p)
```

**4️⃣ Run the notebook** — place `Tweets.csv` (from Kaggle) next to it, then open `NLP_Practical_Exam_SetB.ipynb` and run all cells.

**5️⃣ Run the Streamlit app**
```bash
streamlit run app.py
```

> ⚠️ A `.pkl` file saved with one scikit-learn version may fail on another. `requirements.txt` pins the version used for training.

---

## 🧠 Key Learnings

- Accuracy alone is misleading on imbalanced data — **macro F1** and the **majority baseline (62.67%)** give the real picture.
- Negation words (`not`, `no`, `nor`) were **kept** during stop-word removal because they flip sentiment.
- **TF-IDF + N-Grams** won because bigrams such as *not happy* capture meaning single words miss.
- Trained ML (macro F1 0.74) clearly beat the heuristic (0.51) and TextBlob (0.47) on airline-specific language.
- Neutral tweets are the hardest class; sarcasm and Hinglish (e.g. *"flight late hai, bakwaas service"*) remain weak spots.

## 🔮 Next Steps
More labelled data · better cleaning · Hinglish support · fine-tuning a Hugging Face Transformer (e.g. BERT).

---

## 🎥 Video

[▶️ Watch the 5–10 min walkthrough](VIDEO_URL) 

---

# 👨‍💻 Author

Janki Dholariya
