import streamlit as st
import pandas as pd
import joblib
import re
import nltk

from nltk.tokenize import word_tokenize
from text_cleaning import clean_text


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    words = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    stop_words = stop_words - {'not', 'no', 'nor'}

    words = [w for w in words if w not in stop_words]

    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]

    return ' '.join(words)


# ---------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------

st.set_page_config(
    page_title="Airline Sentiment AI",
    page_icon="✈️",
    layout="wide"
)

# ---------------------------------
# 2. CUSTOM CSS
# ---------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #F4F7FB;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #142B4A;
    text-align: center;
}

.sub-title {
    text-align: center;
    color: #64748B;
    font-size: 17px;
    margin-bottom: 25px;
}

.hero {
    background: linear-gradient(120deg, #142B4A, #2563EB);
    padding: 30px;
    border-radius: 18px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 32px;
}

.hero p {
    color: #E0ECFF;
    font-size: 16px;
}

.result-card {
    background: white;
    padding: 22px;
    border-radius: 15px;
    border-left: 5px solid #2563EB;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.07);
    margin-bottom: 15px;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    text-align: center;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.06);
}

.metric-card h2 {
    color: #142B4A;
}

.section-title {
    color: #142B4A;
    font-size: 23px;
    font-weight: 700;
}

div.stButton > button {
    background: #2563EB;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 22px;
    font-weight: bold;
}

div.stButton > button:hover {
    background: #142B4A;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------
# 3. LOAD MACHINE LEARNING MODEL
# ---------------------------------

@st.cache_resource
def load_model():
    return joblib.load("sentiment_pipeline.pkl")

model = load_model()


# ---------------------------------
# 4. SENTIMENT LABELS
# ---------------------------------

labels = {
    0: "Negative 😠",
    1: "Neutral 😐",
    2: "Positive 😊"
}

sentiment_colors = {
    0: "#DC2626",
    1: "#D97706",
    2: "#16A34A"
}


# ---------------------------------
# 5. HEADER
# ---------------------------------

st.markdown("""
<div class="hero">
    <h1>✈️ Airline Tweet Sentiment Analysis</h1>
    <p>AI Powered Customer Feedback Analysis Using NLP & Machine Learning</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<p class="sub-title">
Analyze airline passenger tweets and understand customer opinions instantly.
</p>
""", unsafe_allow_html=True)


# ---------------------------------
# 6. SIDEBAR
# ---------------------------------

st.sidebar.title("✈️ Airline AI")

st.sidebar.markdown("### About Project")

st.sidebar.write(
    "This application uses Natural Language Processing "
    "and Machine Learning to classify airline tweets "
    "into Negative, Neutral and Positive sentiments."
)

st.sidebar.divider()

st.sidebar.markdown("### 💡 Try Example Tweets")

if st.sidebar.button("😠 Negative Example"):
    st.session_state["tweet_input"] = (
        "My flight is delayed and the service is terrible."
    )

if st.sidebar.button("😐 Neutral Example"):
    st.session_state["tweet_input"] = (
        "My flight is scheduled for 8 PM today."
    )

if st.sidebar.button("😊 Positive Example"):
    st.session_state["tweet_input"] = (
        "Amazing service and friendly crew. Thank you!"
    )

st.sidebar.divider()

st.sidebar.markdown("### 🛠️ Technologies")

st.sidebar.write(
    "Python | NLP | NLTK | Scikit-learn | "
    "TF-IDF | Logistic Regression | Streamlit"
)

st.sidebar.info("Developed for Airline Tweet Analytics Project")


# ---------------------------------
# 7. TWEET INPUT
# ---------------------------------

st.markdown(
    '<p class="section-title">📝 Enter Your Tweet</p>',
    unsafe_allow_html=True
)

tweet = st.text_area(
    "Write your airline experience:",
    key="tweet_input",
    height=130,
    placeholder="Example: My flight was delayed and customer service was poor..."
)

col1, col2 = st.columns([1, 5])

with col1:
    analyse = st.button("🔍 Analyse Tweet", use_container_width=True)

with col2:
    st.caption("Enter a tweet and click Analyse to get the AI prediction.")


# ---------------------------------
# 8. SENTIMENT ANALYSIS
# ---------------------------------

if analyse:

    if not tweet.strip():

        st.warning("Please enter a tweet")

    else:

        cleaned = clean_text(tweet)

        if not cleaned.strip():
            st.warning("Please enter meaningful tweet text.")

        else:

            prediction = model.predict([cleaned])[0]
            prediction = int(prediction)

            st.divider()

            st.markdown(
                '<p class="section-title">📊 Sentiment Analysis Result</p>',
                unsafe_allow_html=True
            )

            # Prediction Card

            color = sentiment_colors.get(prediction, "#2563EB")
            label = labels.get(prediction, str(prediction))

            st.markdown(f"""
            <div class="result-card" style="border-left-color:{color};">
                <h3 style="color:#64748B;">Predicted Sentiment</h3>
                <h1 style="color:{color};">{label}</h1>
                <p style="color:#64748B;">
                    AI model has analyzed your airline tweet.
                </p>
            </div>
            """, unsafe_allow_html=True)


            # ---------------------------------
            # 9. PROBABILITY CHART
            # ---------------------------------

            st.markdown(
                '<p class="section-title">📈 Sentiment Probability</p>',
                unsafe_allow_html=True
            )

            if hasattr(model, "predict_proba"):

                probs = model.predict_proba([cleaned])[0]
                classes = model.classes_

                sentiment_names = {
                    0: "Negative",
                    1: "Neutral",
                    2: "Positive"
                }

                prob_data = pd.DataFrame({
                    "Sentiment": [
                        sentiment_names.get(int(c), str(c))
                        for c in classes
                    ],
                    "Probability": probs
                })

                prob_data["Probability (%)"] = (
                    prob_data["Probability"] * 100
                ).round(2)

                chart_data = prob_data.set_index("Sentiment")[
                    ["Probability (%)"]
                ]

                st.bar_chart(chart_data, horizontal=True)

                st.dataframe(
                    prob_data[["Sentiment", "Probability (%)"]],
                    use_container_width=True,
                    hide_index=True
                )

                confidence = max(probs) * 100

                st.metric(
                    "Model Confidence",
                    f"{confidence:.2f}%"
                )


            # ---------------------------------
            # 10. TWEET INFORMATION
            # ---------------------------------

            st.divider()

            st.markdown(
                '<p class="section-title">🔎 Tweet Information</p>',
                unsafe_allow_html=True
            )

            mentions = re.findall(r'@\w+', tweet)
            hashtags = re.findall(r'#\w+', tweet)

            word_count = len(tweet.split())

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <h2>{word_count}</h2>
                    <p>Word Count</p>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <h2>{len(mentions)}</h2>
                    <p>Mentions</p>
                </div>
                """, unsafe_allow_html=True)

            with c3:
                st.markdown(f"""
                <div class="metric-card">
                    <h2>{len(hashtags)}</h2>
                    <p>Hashtags</p>
                </div>
                """, unsafe_allow_html=True)

            st.write("**Mentions:**", mentions if mentions else "No mentions found")

            st.write("**Hashtags:**", hashtags if hashtags else "No hashtags found")


            # ---------------------------------
            # 11. POS TAGGING
            # ---------------------------------

            with st.expander("🔤 View POS Tagging"):

                nltk.download(
                    "averaged_perceptron_tagger_eng",
                    quiet=True
                )

                words = word_tokenize(cleaned)
                tags = nltk.pos_tag(words)

                pos_df = pd.DataFrame(
                    tags,
                    columns=["Word", "POS Tag"]
                )

                st.dataframe(
                    pos_df,
                    use_container_width=True,
                    hide_index=True
                )


            # ---------------------------------
            # 12. SUGGESTED ACTION
            # ---------------------------------

            st.divider()

            st.markdown(
                '<p class="section-title">💡 Suggested Action</p>',
                unsafe_allow_html=True
            )

            if prediction == 0:

                st.error(
                    "Customer Care: Contact the passenger "
                    "and resolve their complaint as soon as possible."
                )

            elif prediction == 1:

                st.info(
                    "Monitor Feedback: Keep tracking the passenger's "
                    "feedback and provide relevant information."
                )

            elif prediction == 2:

                st.success(
                    "Appreciate Customer: Thank the passenger "
                    "for sharing positive feedback."
                )


# ---------------------------------
# 13. FOOTER
# ---------------------------------

st.divider()

st.markdown("""
<div style="text-align:center; padding:15px; color:#64748B;">
    <b>Airline Tweet Sentiment Analysis</b><br>
    Built with Python | NLP | Machine Learning | Streamlit<br>
    AI-Based Customer Feedback Analysis ✈️
</div>
""", unsafe_allow_html=True)
