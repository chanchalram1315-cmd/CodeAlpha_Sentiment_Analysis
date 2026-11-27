import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import os

# Load dataset
df = pd.read_csv("data/reviews.csv")

# Create output folders
os.makedirs("outputs", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)

# Display dataset
print("Dataset loaded successfully!")

print("\nFirst 5 Reviews:")
print(df.head())

print("\nTotal Reviews:", len(df))

# Download VADER sentiment lexicon
nltk.download("vader_lexicon")

from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Analyze sentiment
def analyze_sentiment(review):
    score = sia.polarity_scores(review)["compound"]

    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis
df["Sentiment"] = df["Review"].apply(analyze_sentiment)

# Display results
print("\nSentiment Analysis Results:")
print(df[["Review", "Sentiment"]].to_string(index=False))

print("\nSentiment Counts:")
print(df["Sentiment"].value_counts())

# Save sentiment analysis results

df.to_csv("outputs/sentiment_results.csv", index=False)

print("\nSentiment results saved successfully!")

# Sentiment Distribution Chart

sentiment_counts = df["Sentiment"].value_counts()

plt.figure(figsize=(8, 5))

sns.barplot(
    x=sentiment_counts.index,
    y=sentiment_counts.values
)

plt.title("Sentiment Distribution of Product Reviews")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")

plt.tight_layout()

plt.savefig("visualizations/sentiment_distribution.png", dpi=300)
plt.show()

print("Sentiment distribution chart saved!")

# Sentiment Percentage Pie Chart

plt.figure(figsize=(7, 7))

plt.pie(
    sentiment_counts.values,
    labels=sentiment_counts.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Percentage of Customer Sentiments")
plt.tight_layout()

plt.savefig("visualizations/sentiment_pie_chart.png", dpi=300)
plt.show()

print("Sentiment pie chart saved!")

# Sentiment Score Analysis

df["Compound_Score"] = df["Review"].apply(
    lambda review: sia.polarity_scores(review)["compound"]
)

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="Compound_Score",
    bins=10,
    kde=True
)

plt.title("Distribution of Sentiment Scores")
plt.xlabel("Compound Sentiment Score")
plt.ylabel("Number of Reviews")

plt.tight_layout()

plt.savefig("visualizations/sentiment_score_distribution.png", dpi=300)
plt.show()

print("Sentiment score chart saved!")

# Combined Sentiment Analysis Dashboard

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Bar Chart
sns.barplot(
    x=sentiment_counts.index,
    y=sentiment_counts.values,
    ax=axes[0]
)
axes[0].set_title("Sentiment Distribution")
axes[0].set_xlabel("Sentiment")
axes[0].set_ylabel("Number of Reviews")

# Pie Chart
axes[1].pie(
    sentiment_counts.values,
    labels=sentiment_counts.index,
    autopct="%1.1f%%",
    startangle=90
)
axes[1].set_title("Customer Sentiment Percentage")

fig.suptitle("Product Reviews Sentiment Analysis", fontsize=16)
plt.tight_layout()

plt.savefig(
    "visualizations/sentiment_dashboard.png",
    dpi=300
)
plt.show()

print("Sentiment dashboard created successfully!")