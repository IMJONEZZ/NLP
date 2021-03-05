def lemmatization_with_spacy(text=u"he was running late"):
    import spacy

    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    lemmas = []
    for token in doc:
        print(f"{token} --> {token.lemma_}")
        lemmas.append(token.lemma_)

    return lemmas