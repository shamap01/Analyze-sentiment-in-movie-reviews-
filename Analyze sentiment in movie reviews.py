import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.sentiment import SentimentIntensityAnalyzer

# Load the dataset
df = pd.read_csv("movie_reviews.csv")

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Remove special characters and digits
    text = re.sub(r'\W|\d', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    words = word_tokenize(text)
    # Remove stop words
    words = [word for word in words if word not in stop_words]
    # Stemming
    words = [lemmatizer.lemmatize(word) for word in words]
    # Join the words
    text = ' '.join(words)
    return text

# Apply preprocessing to the reviews
df['review'] = df['review'].apply(preprocess_text)

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['review'])

# K-Means Clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Assign cluster labels to each review
df['cluster'] = kmeans.labels_

# Sentiment Analysis
sia = SentimentIntensityAnalyzer()
df['sentiment'] = df['review'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Print the cluster labels and sentiment scores for each review
print(df[['review', 'cluster', 'sentiment']])
