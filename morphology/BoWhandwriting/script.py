from goldman_emma_raw import goldman_docs
from henson_matthew_raw import henson_docs
from wu_tingfang_raw import wu_docs
# import sklearn modules here:
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Setting up the combined list of friends' writing samples
friends_docs = goldman_docs + henson_docs + wu_docs
# Setting up labels for your three friends
friends_labels = [1] * 154 + [2] * 141 + [3] * 166

# Print out a document from each friend:
#print(goldman_docs[10])
#print(henson_docs[10])
#print(wu_docs[10])

mystery_postcard = """
My friend,
From the 10th of July to the 13th, a fierce storm raged, clouds of
freeing spray broke over the ship, incasing her in a coat of icy mail,
and the tempest forced all of the ice out of the lower end of the
channel and beyond as far as the eye could see, but the _Roosevelt_
still remained surrounded by ice.
Hope to see you soon.
"""

# Create bow_vectorizer:
bow_vectorizer = CountVectorizer()
# Define friends_vectors:
friends_vectors = bow_vectorizer.fit_transform(friends_docs)
# Define mystery_vector: 
mystery_vector = bow_vectorizer.transform([mystery_postcard])

# Define friends_classifier:
friends_classifier = MultinomialNB()
# Train the classifier:
friends_classifier.fit(friends_vectors, friends_labels)
# Change predictions:
predictions = friends_classifier.predict(mystery_vector)

mystery_friend = predictions[0] if predictions[0] else "someone else"

probs = friends_classifier.predict_proba(mystery_vector)
# Uncomment the print statement:
print(mystery_postcard)
print(probs)
print("The postcard was from {}!".format(mystery_friend))
