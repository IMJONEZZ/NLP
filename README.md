# NLP

## Quick Explanation:
This repo contains reference work that I've done for my career and schooling that I figure could be helpful for more than just myself. All of the code is written in Python 3.5+ and  
> I've taken special care to make sure it's compatible through the latest versions  
>>(currently 3.8.x).

## Publications:
This folder contains publications, complete with code to run for yourselves.
- Machine Learning Based Chat Analysis (Libchat)
    - November 2020
    - Includes Toy Dataset
    - Does not include production model
> Abstract:
> The BYU library implmented a machine learning based tool to perform various text analysis tasks on transcripts of chat-based interactions between patrons and librarians. These text analysis tasks included estimating patron satisfaction, and classifying queries into various categories such as Research/Reference, Directional, Tech/Troubleshooting, Policy/Procedure, and others. An accuracy of 78% or better was achieved for each category. This paper details the implementation details and explores potential applications for the text analysis tool.

## Linguistics
- This folder contains basic linguistic principles required to evaluate any NLP task and provide better QA than any BLEU score could give
    - From ancient writing systems to modern semantics, this folder contains pictures and presentations explaining everything you need to be able to understand where language came from, why it's here, and where it's going.
    - Chris, I'm a computer science major, so how can this possibly help me? I'll tell you fine fellow! Have you ever wondered why machine translation between English and Arabic or English and Japanese is uncharacteristically bad? Simply put, most of the people who have worked on these systems have not known the intricacies of Alphabets vs Abjads vs Logo-Syllabaries. This is not a diss or an accusation, but we are now at a point where we can apply mathematics to the linguistics in order to improve specific languages. This requires a solid foundation in both parts, otherwise we will end up with the same or similar results compared with what we've gotten before.

## Natural Language Processing with Classification and Vector Spaces
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
## Natural Language Processing with Probabalistic Models
- This folder takes the concepts from the first folder one step further, allowing for slightly more complex implementations:
    - Minimum Edit Distance - Autocorrect: Contains data to go from building a vocabulary from text all the way to utilizing Levenshtein distance to detect whether a given word is incorrect and suggest replacements
    - Part of Speech Tagging: How to do it with probabilities, and what its limitations are
    - N-grams: How do you guess what words are coming next? Build an autocomplete system and use probabilities to provide suggestions
    - Some good advice for working with text in python and using the Numpy framework.
    - What is a Continuous Bag of Words model, and how does it differ from the Bag of Words model in the NLP folder?
    - How to get Word embeddings for use in other applications, like the Naive Bayes Classifier in the Classification folder
    - Final Project: Creating Word Embeddings from Shakespeare text with a Continuous Bag of Words Model
## Natural Language Processing with Sequence Models
- Welcome to deep neural networks for completing the same tasks we've already learned about! Why are DNN's better? They're quicker, require less code, and are many times more accurate.
    - Intro to Trax: Why not PyTorch or TensorFlow?
    - Sentiment Analysis on a DNN
    - Autocorrect on a GRU (deep N-grams as opposed to the shallow ones in previous sections)
    - LSTMs, RNNs, and GRUs: What's the difference, and what are the use cases?
    - Named Entity Recognition with an LSTM - Want to build a bot to summarize academic papers for you?
    - One-shot learning, how to use a threshold and one set of weights to avoid retraining on every new piece of data
    - Siamese Networks - How do we determine whether an utterance means the same thing as another utterance (e.g. How old are you vs. What is your age)
    - Final Project: Question similarity on Siamese LSTMs using Quora's Question dataset.
## Natural Language Processing with Attention Models
- While the last folder (sequence models) contains most of the models that are currently deployed or being deployed by businesses to production, this folder contains methods for state-of-the-art with attention models. Big thanks to Google's DeepMind team for coming up with this.
    - Attention: What is it? How do you do it? What types are there? All of these questions are answered.
    - Neural Machine Translation with Attention - build upon other MT projects contained in this repo (Locality Sensitive Hashing, seq2seq, both deep and shallow) and add attention to machine translation. If you're extra quick, you'll be able to do it with a transformer too!
    - Translation scoring metrics - learn about BLEU, ROUGE, and F1. Which should you use when? What are the pros and cons to each?
    - Summarize text with a transformer network built from scratch.
    - Byte Pair Encoding and SentencePiece
    - Transfer Learning - What are the limits? When should you Transfer Learn instead of training from scratch? How do you do it?
    - Answer Questions with T5 with a finetuned Question Answer model
## NLP
- This folder contains its own README because it concerns primarily language modelling. From bag of words all the way to neural language models, there are easy-to-follow and change implementations with fun little examples to go with them. 
    - Adapted from the NLP specialization at Codecademy.