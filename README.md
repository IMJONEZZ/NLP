### NLP

This notebook contains example implementations of a bunch of helpful NLP program implementations such as:
- Bag of Words: A statistical language model that uses wordcounts to make other important conclusions from text
- Levenshtein Distance: An algorithm for measuring the distance between words by measuring how many changes in spelling it would take to change one word into another.
- Markov Chains: A stochastic (basically random) language model that chains events together to determine the probability of subsequent events.
- Naive Bayes: A statistical classifier used to create assumptions between features passed in. Using those features and assumptions, it is able to predict which is the most likely between several classes.
- N-Grams: Used for parsing entire phrases, language prediction, and translation. N-Grams looks at a sequence of an unknown number (n) of units (letter, morphemes, words, sentences, anything) and predicts the probability of each unit appearing again.
- Topic Modelling: A technique for prioritizing less frequently used terms as *topics* using **term frequency-inverse document frequency (tf-idf)**. When working with larger text like News articles or Wikipedia, this is extremely helpful for determining topics that aren't the most common words like 'the' or 'is.'
- Word Embeddings: With the assumption from linguist John Rupert - "You shall know a word by the company it keeps," word embeddings utilize large vectors of words built upon frequency and context. Using these vectors, it's possible to begin to determine word usage and meaning by comparing the cosine distance between vectors. Between similar words that cosine distance will be small, and very different words will have a large cosine distance.
- Neural Machine Translation: A very simple implementation using Tensorflow and Keras for training a translation language model. Includes training texts for Russian, Slovak, and Czech.

## Bag of Words

Running the program on its defaults will analyze the word counts for a portion of *Through the Looking-Glass* by Lewis Carroll. This will work for any plain text document though, just change the import statement and looking_glass_text variable.

## Bag of Words - Handwriting Analysis

Using both Bag of Words and Naive Bayes in conjunction, this program can predict which of a list of possible authors wrote a given piece of text, using training data from each of the authors. The default program compares data from Emma Goldman, Matthew Henson, and Tingfang Wu, though it can be used with any authors if the text is presented in a similar format.

## Iliad Analysis

Using Regex parsing and simple grammars written for Noun Phrases and Verb Phrases, this program tokenizes and tags every sentence within *The Iliad* by Homer, then outputs the 30 most common Noun Phrases and Verb Phrases. Using those phrases, we can deduce different things about the character of each person described in the book. Feel free to try this out with Shakespeare or any text that meets your fancy!

## Levenshtein Distance

The program here is fairly simple, all it does is calculate and output the Levenshtein distance between any two words. The default words are: squanker, target, code, dot, chunk, and skunk. Test it out on your favorite words!

## Markov Chain

Using a custom Markov Chain class, this program generates text (50 words by default, edit max_length to change) based on 3 imported documents. By default the paragraph generated draws from Louisa May Alcott's 3 most famous books. Try it out using your favorite singers and see if it sounds like them! Disclaimer: this is not even close to the level that GPT-2 or GPT-3 run on.

## Naive Bayes

This program performs low-level sentiment analysis on reviews. Try changing the review passed in, did it get it right? For a nice comparison of several other algorithms for sentiment analysis, check out https://quicksentiment.herokuapp.com

## N-Grams

After preprocessing, this program outputs the most common bigrams, trigrams, and n-grams, where *n* is the number passed into the third function call. Unlike the original Bag of Words, this looks at the entire text of *Through the Looking Glass*, although it's useful for any text.

## NMT

In order to run, add a .txt file containing parrallel sentences in a similar format to the 3 already in the folder to the same folder. Edit the data_path variable in preprocessing.py to point to your file. In training_model.py, make sure the model being saved matches the model being loaded in test_function.py. Make sure to name the model something descriptive so that you can keep good track of them.

## Presidential Analysis

This program uses Word Embeddings to analyze the inaugural speeches of every president of the United States so far. Because this is a slightly complex program, the program contains many examples of how to run the different functions on any of the presidents.

TODO: Mine more speeches from presidents **OUTSIDE** of the inaugurations.

## Topic Model

When run, this program compares 2 different Topic Model algorithms: Bag of Words and TF-IDF. Defaults compare topic modelling using several chapters picked from *The Adventures of Sherlock Holmes* by Sir Arthur Conan Doyle. Compare the outputs on different texts to get an intuition for whether BoW or tf-idf is better for your specific project.

## Word2Vec

Using the classic *Romeo and Juliet*, this program embeds all of the words in the play, then provides several functions to compare them.

## Preprocess

An example of fairly standard tokenization and part-of-speech tagging using the Natural Language ToolKit.

## Treeparsing

Showcases parsing trees also using nltk. Add a text replacing Lorem Ipsum, and check out whether it did a good job!