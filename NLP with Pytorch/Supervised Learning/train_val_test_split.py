import collections
import numpy as np
import pandas as pd
import re

from argparse import Namespace

def train_val_test_split(train_dataset_path, list_of_train_names, test_dataset_path, list_of_test_names, output_csv_path):

    args = Namespace(
        raw_train_dataset_csv=train_dataset_path,
        raw_test_dataset_csv=test_dataset_path,
        train_proportion=0.7,
        val_proportion=0.3,
        output_munged_csv=output_csv_path,
        seed=1337
    )

    train = pd.read_csv(args.raw_train_dataset_csv,
    header=None, names=list_of_train_names)
    train = train[~pd.isnull(train.text)]
    test = pd.read_csv(args.raw_test_dataset_csv,
    header=None, names=list_of_test_names)
    test = test[~pd.isnull(test.text)]

    print(set(train.label))

    by_label = collections.defaultdict(list)
    for _, row in review_subset.iterrows():
        by_label[row.label].append(row.to_dict())

    final_list = []
    np.random.seed(args.seed)

    for _, item_list in sorted(by_label.items()):
        np.random.shuffle(item_list)

        n_total = len(item_list)
        n_train = int(args.train_proportion * n_total)
        n_val = int(args.val_proportion * n_total)

        for item in item_list[:n_train]:
            item['split'] = 'train'

        for item in item_list[n_train:n_train+n_val]:
            item['split'] = 'val'

        final_list.extend(item_list)

    final_texts = pd.DataFrame(final_list)

    return final_texts