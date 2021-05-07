from torch.utils.data import Dataset

class MyDataset(Datset):
    def __init__(self, dataframe, vectorizer):
        self.dataframe = dataframe
        self._vectorizer = vectorizer

        self.train_df = self.dataframe[self.dataframe.split=='train']
        self.train_size = len(self.train_df)

        self.val_df = self.dataframe[self.dataframe.split=='val']
        self.val_size = len(self.val_df)

        self.test_df = self.dataframe[self.dataframe.split=='test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {'train': (self.train_df, self.train_size),
                            'val': (self.val_df, self.val_size),
                            'test': (self.test_df, self.test_size)}
        
        self.set_split('train')

    @classmethod
    def load_dataset_and_make_vectorizer(cls, csv):
        dataframe = pd.read_csv(csv)
        return cls(dataframe, MyVectorizer.from_dataframe(dataframe))

    def get_vectorizer(self):
        return self._vectorizer

    def set_split(self, split='train'):
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        row = self._target_df.iloc[index]

        vector = self._vectorizer.vectorize(row.text)

        label_index = self._vectorizer(rating_vocab.lookup_token(row.label))

        return {'x_data': vector,
                'y_target': label_index}

    def get_num_batches(self, batch_size):
        return len(self) // batch_size