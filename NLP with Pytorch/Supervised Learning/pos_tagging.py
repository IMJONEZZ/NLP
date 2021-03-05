def pos_tag_with_spacy(text=u"Mary slapped the green witch."):
    import spacy

    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    pos_list = []
    for token in doc:
        print(f"{token} - {token.pos_}")
        pos_list.append(token.pos_)
    
    return pos_list

pos_tag_with_spacy()