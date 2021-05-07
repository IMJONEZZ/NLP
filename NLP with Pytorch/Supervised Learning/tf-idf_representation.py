from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt

corpus = ["Time flies flies like an arrow.",
          "Fruit flies like a banana."]
tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(corpus).toarray()
vocab = ['an', 'arrow', "banana", "flies", "fruit", "like", "time"]

plt.figure(figsize=(16,9))
sns.heatmap(tfidf, annot=True, cbar=False, xticklabels=vocab, yticklabels=['Sentence 1', 'Sentence 2'])
plt.show()