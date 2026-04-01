import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.pipeline import Pipeline

# Data load karo
fake = pd.read_csv('data/Fake.csv')
real = pd.read_csv('data/True.csv')

fake['label'] = 'FAKE'
real['label'] = 'REAL'

df = pd.concat([fake, real])
df = df.sample(frac=1).reset_index(drop=True)

X = df['title'] + ' ' + df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline banao
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=50000, ngram_range=(1,2))),
    ('clf', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Model save karo
with open('model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("Model saved successfully!")