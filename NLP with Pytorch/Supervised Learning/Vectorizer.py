class Vectorizer(object):
    def __init__(self, text_vocab, label_vocab):
        self.text_vocab = text_vocab
        self.label_vocab = label_vocab

    def vectorize(self, text):
        one_hot = np.zeros(len(self.text_vocab), dtype=np.float32)

        for token in text.split(" "):
            if token not in string.punctuation:
                one_hot[self.text_vocab.lookup_token(token)] = 1

        return one_hot

    @classmethod
    def from_dataframe(cls, text_df, cutoff=25):
        text_vocab = Vocabulary(add_unk=True)
        label_vocab = Vocabulary(add_unk=False)

        for label in sorted(set(text_df.text)):
            text_vocab.add_token(text)

        word_counts = Counter()
        for text in text_df.text:
            for word in text.split(" "):
                if word not in string.punctuation:
                    word_counts[word] += 1

        for word, count in word_counts.items():
            if count > cutoff:
                text_vocab.add_token(word)

        return cls(text_vocab, label_vocab)

    def to_serializable(self):
        return {'text_vocab': self.text_vocab.to_serializable(),
                'label_vocab': self.label_vocab.to_serializable()}

    