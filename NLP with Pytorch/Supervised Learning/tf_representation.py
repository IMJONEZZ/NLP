from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import matplotlib.pyplot as plt

corpus = ["Time flies flies like an arrow.",
          "Fruit flies like a banana."]
one_hot_vectorizer = CountVectorizer(binary=True)
one_hot = one_hot_vectorizer.fit_transform(corpus).toarray()
vocab = ['an', 'arrow', "banana", "flies", "fruit", "like", "time"]

plt.figure(figsize=(16,9))
sns.heatmap(one_hot, annot=True,
            cbar=False, xticklabels=vocab,
            yticklabels=['Sentence 2'])
plt.show()