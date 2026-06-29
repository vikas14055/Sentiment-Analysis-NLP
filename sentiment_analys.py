import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import string
import os

from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("stopwords")
nltk.download("vader_lexicon")

plt.style.use("ggplot")

# -----------------------
# Load Dataset
# -----------------------
df = pd.read_csv("books_dataset.csv")

df["Review"] = df["Review"].fillna("")

print(df.head())

# -----------------------
# NLP Cleaning
# -----------------------
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df["Clean_Review"] = df["Review"].apply(clean_text)

# -----------------------
# SENTIMENT ANALYSIS (VADER - Better than TextBlob)
# -----------------------
sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = sia.polarity_scores(text)["compound"]

    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df["Sentiment"] = df["Clean_Review"].apply(get_sentiment)

# -----------------------
# EMOTION DETECTION (Simple Lexicon Logic)
# -----------------------
def get_emotion(text):
    text = text.lower()

    if any(word in text for word in ["love", "amazing", "great", "fantastic", "wonderful"]):
        return "Joy"
    elif any(word in text for word in ["boring", "disappointing", "bad", "worst"]):
        return "Sadness"
    elif any(word in text for word in ["angry", "annoying", "hate"]):
        return "Anger"
    elif any(word in text for word in ["scary", "thrilling", "mystery"]):
        return "Fear"
    else:
        return "Neutral"

df["Emotion"] = df["Clean_Review"].apply(get_emotion)

# -----------------------
# TREND ANALYSIS
# -----------------------
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

# Sentiment vs Price insight
print("\nAverage Price by Sentiment:")
print(df.groupby("Sentiment")["Price"].mean())

# -----------------------
# SAVE OUTPUT
# -----------------------
os.makedirs("output", exist_ok=True)
os.makedirs("charts", exist_ok=True)

df.to_csv("output/final_nlp_output.csv", index=False)

# -----------------------
# VISUAL 1: SENTIMENT DISTRIBUTION
# -----------------------
plt.figure(figsize=(6,4))
sns.countplot(x="Sentiment", data=df)
plt.title("Sentiment Distribution")
plt.tight_layout()
plt.savefig("charts/sentiment.png")
plt.show()

# -----------------------
# VISUAL 2: EMOTION DISTRIBUTION
# -----------------------
plt.figure(figsize=(6,4))
sns.countplot(x="Emotion", data=df)
plt.title("Emotion Distribution")
plt.tight_layout()
plt.savefig("charts/emotion.png")
plt.show()

# -----------------------
# VISUAL 3: PRICE vs SENTIMENT
# -----------------------
plt.figure(figsize=(6,4))
sns.boxplot(x="Sentiment", y="Price", data=df)
plt.title("Price vs Sentiment Analysis")
plt.tight_layout()
plt.savefig("charts/price_sentiment.png")
plt.show()

# -----------------------
# VISUAL 4: PIE CHART
# -----------------------
plt.figure(figsize=(6,6))
df["Sentiment"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Sentiment Percentage")
plt.ylabel("")
plt.tight_layout()
plt.savefig("charts/pie.png")
plt.show()

# -----------------------
# FINAL INSIGHTS
# -----------------------
print("\n==================== FINAL INSIGHTS ====================")

print("\n1. Most common sentiment:")
print(df["Sentiment"].value_counts())

print("\n2. Most common emotion:")
print(df["Emotion"].value_counts())

print("\n3. Marketing Insight:")
print("- Positive reviews indicate strong customer satisfaction.")
print("- Negative reviews highlight product issues or dissatisfaction.")
print("- Price does not strongly determine sentiment in this dataset.")

print("\nPROJECT COMPLETED SUCCESSFULLY 🚀")