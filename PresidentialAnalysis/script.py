import os
import gensim
import spacy
from president_helper import read_file, process_speeches, merge_speeches, get_president_sentences, get_presidents_sentences, most_frequent_words

# get list of all speech files
files = [file for file in os.listdir() if file[-4:] == '.txt']
#print(f'These are the files in this folder: {files}\n')

# read each speech file
speeches = [read_file(file) for file in files]

# preprocess each speech
processed_speeches = process_speeches(speeches)

# merge speeches
all_sentences = merge_speeches(processed_speeches)

# view most frequently used words
#most_freq_words = most_frequent_words(all_sentences)
#print(f' Here are the most frequent words in all presidential speeches: {most_freq_words}\n')

# create gensim model of all speeches
#all_prez_embeddings = gensim.models.Word2Vec(all_sentences,size=96,window=5,min_count=1,workers=2,sg=1)

# view words similar to freedom
#similar_to_freedom = all_prez_embeddings.most_similar('freedom',topn=20)
#print(f'These are the most similar words to freedom from all presidents of the United States: {similar_to_freedom}\n')

# get President Roosevelt sentences
#roosevelt_sentences = get_president_sentences("franklin-d-roosevelt")
#print(roosevelt_sentences)

# view most frequently used words of Roosevelt
#roosevelt_most_freq_words = most_frequent_words(roosevelt_sentences)
#print(f' These are the most frequent words used by FDR: {roosevelt_most_freq_words}\n')

# create gensim model for Roosevelt
#roosevelt_embeddings = gensim.models.Word2Vec(roosevelt_sentences,size=96,window=5,min_count=1,workers=2,sg=1)

# view words similar to freedom for Roosevelt
#roosevelt_similar_to_freedom = roosevelt_embeddings.most_similar("freedom",topn=20)
#print(roosevelt_similar_to_freedom)

# get sentences of multiple presidents
#rushmore_prez_sentences = get_presidents_sentences(["washington","jefferson","lincoln","theodore-roosevelt"])

# view most frequently used words of presidents
#rushmore_most_freq_words = most_frequent_words(rushmore_prez_sentences)
#print(f'The most frequent words used by the Rushmore presidents: {rushmore_most_freq_words}\n')

# create gensim model for the presidents
#rushmore_embeddings = gensim.models.Word2Vec(rushmore_prez_sentences,size=96,window=5,min_count=1,workers=2,sg=1)

# view words similar to freedom for presidents
#rushmore_similar_to_freedom = rushmore_embeddings.most_similar("freedom",topn=20)
#print(f'These are the most similar words to freedom from the Rushmore presidents: {rushmore_similar_to_freedom}\n')

trump_sentences = get_president_sentences("trump")
trump_most_freq_words = most_frequent_words(trump_sentences)
trump_embeddings = gensim.models.Word2Vec(trump_sentences,size=96,window=5,min_count=1,workers=2,sg=1)
#trump_similar_to_fake = trump_embeddings.most_similar("fake",topn=20)

'''bom_sentences = get_president_sentences("bom")
unique_word = []
for sentence in bom_sentences:
    for word in sentence:
        if word in unique_word:
            continue
        else:
            unique_word.append(word)
most_freq_bom_words = most_frequent_words(bom_sentences)
all_bom_embeddings = gensim.models.Word2Vec(bom_sentences,size=96,window=5,min_count=1,workers=2,sg=1)
similar_to_christ = all_bom_embeddings.most_similar("christ",topn=20)
similar_to_prophet = all_bom_embeddings.most_similar("prophet",topn=20)
similar_to_lamanite = all_bom_embeddings.most_similar("lamanite",topn=20)'''

allofit = [trump_most_freq_words]

with open('results_trump.txt', 'w', encoding='utf-8') as f:
    for item in allofit:
        f.write(f"This is its own section:\nIt is {str(len(item))} items long\n{str(item)}\n")