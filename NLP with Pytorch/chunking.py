def noun_chunking_with_spacy(text=u"Mary slapped the green witch."):
    import spacy

    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    for chunk in doc.noun_chunks:
        print(f"{chunk} - {chunk.label_}")

    return doc.noun_chunks