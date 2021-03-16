import torch
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
from transformers import BertTokenizer, BertForSequenceClassification

def load_data(device, source_folder = 'data'):
    print("load_data - device:", device)

    # Fields

    label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
    text_field = Field(tokenize='spacy', lower=True, include_lengths=True, batch_first=True)
    fields = [('text', text_field), ('label', label_field)]

    # TabularDataset

    train, valid, test = TabularDataset.splits(path=source_folder, train='train.csv', validation='valid.csv',
                                            test='test.csv', format='CSV', fields=fields, skip_header=True)
    novel = TabularDataset(
        path = source_folder+'/novel.csv',
        format = 'csv',
        skip_header = True,
        fields = fields
    )

    # Iterators
    train_iter = BucketIterator(train, batch_size=16, sort_key=lambda x: len(x.text),
                                device=device, train=True, sort=True, sort_within_batch=True)
    valid_iter = BucketIterator(valid, batch_size=16, sort_key=lambda x: len(x.text),
                                device=device, train=True, sort=True, sort_within_batch=True)
    test_iter = Iterator(test, batch_size=16, device=device, train=False, shuffle=False, sort=False)
    novel_iter = Iterator(novel, batch_size=1, device=device, sort=False, sort_within_batch=False, repeat=False, shuffle=False)
    # Vocabulary
    text_field.build_vocab(train, min_freq = 3)
    return train_iter, valid_iter, test_iter, novel_iter, text_field.vocab