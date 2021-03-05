def tokenize_with_spacy(text="Mary, don't slap the green witch"):
    import spacy

    nlp = spacy.load('en_core_web_sm')

    cleaned = [str(token) for token in nlp(text.lower())]
    print(cleaned)
    return cleaned
    

def tokenize_with_nltk(tweet=u"Snow White and the Seven Degrees #MakeAMovieCold@midnight:-)"):
    from nltk.tokenize import TweetTokenizer

    tokenizer = TweetTokenizer()
    cleaned = tokenizer.tokenize(tweet.lower())
    print(cleaned)
    return cleaned