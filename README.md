# NLP

## Quick Explanation:
This repo contains reference work that I've done for my career and schooling that I figure could be helpful for more than just myself. All of the code is written in Python 3.5+ and  
> I've taken special care to make sure it's compatible through the latest versions  
>>(currently 3.8.x).

There are 3 main folders at this point:
- Natural Language Processing with Classification and Vector Spaces
    - This folder contains the basics for these subjects:
        - Preprocessing text for NLP
        - Visualizing Datasets, Naive Bayes classifiers, and Word Embeddings
        - Using Logistic Regression for Sentiment Analysis
        - Building a Naive Bayes classifier for Sentiment Analysis
        - Principle Component Analysis TODO: Make this one easier to understand, also fix the end of the code
        - How to do Linear Algebra with Numpy (useful for NLP)
        - The HashTable data structure and its use for rudimentary machine translation
        - Final Project: Machine Translation and Locality Sensitive Hashing
        - Adapted from the NLP specialization from deeplearning.ai (Stanford University professors' side hustle)
- Natural Language Processing with Probabalistic Model
    - This folder takes the concepts from the first folder one step further, allowing for slightly more complex implementations:
        - Minimum Edit Distance - Autocorrect: Contains data to go from building a vocabulary from text all the way to utilizing Levenshtein distance to detect whether a given word is incorrect and suggest replacements
        - Part of Speech Tagging: How to do it with probabilities, and what its limitations are
        - N-grams: How do you guess what words are coming next? Build an autocomplete system and use probabilities to provide suggestions
        - Some good advice for working with text in python and using the Numpy framework.
        - What is a Continuous Bag of Words model, and how does it differ from the Bag of Words model in the NLP folder?
        - How to get Word embeddings for use in other applications, like the Naive Bayes Classifier in the Classification folder
        - Final Project: Creating Word Embeddings from Shakespeare text with a Continuous Bag of Words Model
- NLP
    - This folder contains its own README because it concerns primarily language modelling. From bag of words all the way to neural language models, there are easy-to-follow and change implementations with fun little examples to go with them. 
    - Adapted from the NLP specialization at Codecademy.