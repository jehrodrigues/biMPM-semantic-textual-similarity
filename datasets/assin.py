"""
Data comes from SemEval-2014 Task 1: Evaluation of Compositional Distributional Semantic Models
on Full Sentences through Semantic Relatedness and Entailment
http://alt.qcri.org/semeval2014/task1/index.php?id=data-and-tools
"""
import math

import numpy as np
import torch
from torchtext.data import BucketIterator, Field, interleave_keys, RawField
from torchtext.data.dataset import TabularDataset
from torchtext.data.pipeline import Pipeline
from torchtext.vocab import Vectors
from myvocab import myVectors
from torchtext.vocab import FastText
from torchtext.vocab import GloVe


def get_class_probs(sim, *args):
    """
    Convert a single label into class probabilities.
    """
    class_probs = np.zeros(ASSIN.num_classes)
    ceil, floor = math.ceil(sim), math.floor(sim)
    if ceil == floor:
        class_probs[floor - 1] = 1
    else:
        class_probs[floor - 1] = ceil - sim
        class_probs[ceil - 1] = sim - floor

    return class_probs


class ASSIN(TabularDataset):

    name = 'assin'
    num_classes = 5

    def __init__(self, path, format, fields, skip_header=True, **kwargs):
        super(ASSIN, self).__init__(path, format, fields, skip_header, **kwargs)

        # We want to keep a raw copy of the sentence for some models and for debugging
        RAW_TEXT_FIELD = RawField()
        for ex in self.examples:
            raw_sentence_a, raw_sentence_b = ex.sentence_a[:], ex.sentence_b[:]
            setattr(ex, 'raw_sentence_a', raw_sentence_a)
            setattr(ex, 'raw_sentence_b', raw_sentence_b)

        self.fields['raw_sentence_a'] = RAW_TEXT_FIELD
        self.fields['raw_sentence_b'] = RAW_TEXT_FIELD

    @staticmethod
    def sort_key(ex):
        return interleave_keys(
            len(ex.sentence_a), len(ex.sentence_b))

    @classmethod
    def splits(cls, text_field, label_field, id_field, path='data/assin', root='', train='train/ASSIN_train.txt',
               validation='dev/ASSIN_dev.txt', test='test/ASSIN_test.txt', **kwargs):

        return super(ASSIN, cls).splits(path, root, train, validation, test,
                                       format='tsv',
                                       fields=[('id', id_field), ('sentence_a', text_field), ('sentence_b', text_field),
                                               ('relatedness_score', label_field)], #, ('entailment', None)
                                       skip_header=True)

    @classmethod
    def iters(cls, batch_size=64, device=-1, shuffle=True, vectors='glove.300d'):
        cls.TEXT = Field(sequential=True, tokenize='spacy', lower=True, batch_first=True)
        cls.LABEL = Field(sequential=False, use_vocab=False, batch_first=True, tensor_type=torch.FloatTensor, postprocessing=Pipeline(get_class_probs))
        cls.ID = Field(sequential=False, use_vocab=False, batch_first=True, tensor_type=torch.FloatTensor)

        train, val, test = cls.splits(cls.TEXT, cls.LABEL, cls.ID)

        vectors = Vectors(name='glove_s300.txt', url='http://143.107.183.175:22980/download.php?file=embeddings/glove/glove_s300.zip')

        cls.TEXT.build_vocab(train, vectors)

        return BucketIterator.splits((train, val, test), batch_size=batch_size, shuffle=shuffle, repeat=False, device=device)
