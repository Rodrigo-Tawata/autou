# train.py
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

nltk.download('stopwords')
nltk.download('rslp')

stop_pt = set(stopwords.words('portuguese'))
stemmer = RSLPStemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zà-ú0-9\s]', ' ', text)
    tokens = [w for w in text.split() if w not in stop_pt]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)

df = pd.read_csv("data/examples.csv")  # csv com colunas text,label
df['text_proc'] = df['text'].astype(str).apply(preprocess)

X = df['text_proc']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=2)),
    ('clf', LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
])

pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)
print(classification_report(y_test, pred))

joblib.dump(pipeline, "models/classifier.joblib")
print("Modelo salvo em models/classifier.joblib")
