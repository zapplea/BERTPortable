# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create masked LM/next sentence masked_lm TF examples for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import pandas as pd
from collections import Counter
import pickle
import numpy as np
import gc

# flags = tf.flags
#
# FLAGS = flags.FLAGS
#
# flags.DEFINE_string("input_file", None,
#                     "Input raw text file (or comma-separated list of files).")
#
# flags.DEFINE_string(
#     "output_file", None,
#     "Output TF example file (or comma-separated list of files).")
#
# flags.DEFINE_string("vocab_file", None,
#                     "The vocabulary file that the BERT model was trained on.")
#
# flags.DEFINE_bool(
#     "do_lower_case", True,
#     "Whether to lower case the input text. Should be True for uncased "
#     "models and False for cased models.")
#
# flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")
#
# flags.DEFINE_integer("max_predictions_per_seq", 20,
#                      "Maximum number of masked LM predictions per sequence.")
#
# flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")
#
# flags.DEFINE_integer(
#     "dupe_factor", 10,
#     "Number of times to duplicate the input data (with different masks).")
#
# flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")
#
# flags.DEFINE_float(
#     "short_seq_prob", 0.1,
#     "Probability of creating sequences which are shorter than the "
#     "maximum length.")


class TrainingInstance(object):
  """A single training instance (sentence pair)."""

  def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
               is_random_next):
    self.tokens = tokens
    self.segment_ids = segment_ids
    self.is_random_next = is_random_next
    self.masked_lm_positions = masked_lm_positions
    self.masked_lm_labels = masked_lm_labels

  def __str__(self):
    s = ""
    s += "tokens: %s\n" % (" ".join(
        [x for x in self.tokens]))
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
    s += "is_random_next: %s\n" % self.is_random_next
    s += "masked_lm_positions: %s\n" % (" ".join(
        [str(x) for x in self.masked_lm_positions]))
    s += "masked_lm_labels: %s\n" % (" ".join(
        [x for x in self.masked_lm_labels]))
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()



def write_instance_to_example_files(instances, vocab, max_seq_length,
                                    max_predictions_per_seq, output_file):
  # DONE: need to save data to pickle
  """Create TF example files from `TrainingInstance`s."""
  input_ids_collection = []
  input_mask_collection = []
  segment_ids_collection = []
  masked_lm_positions_collection = []
  masked_lm_ids_collection = []
  masked_lm_weights_collection = []
  next_sentence_label_collection = []

  with open(output_file,'wb') as outf:
    for (inst_index, instance) in enumerate(instances):
      # EXPL: convert token to id
      input_ids = convert_tokens_to_ids(instance.tokens,vocab)
      input_mask = [1] * len(input_ids)
      segment_ids = list(instance.segment_ids)
      assert len(input_ids) <= max_seq_length

      while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

      assert len(input_ids) == max_seq_length
      assert len(input_mask) == max_seq_length
      assert len(segment_ids) == max_seq_length

      masked_lm_positions = list(instance.masked_lm_positions)
      masked_lm_ids = convert_tokens_to_ids(instance.masked_lm_labels,vocab)
      masked_lm_weights = [1.0] * len(masked_lm_ids)

      while len(masked_lm_positions) < max_predictions_per_seq:
        masked_lm_positions.append(0)
        masked_lm_ids.append(0)
        masked_lm_weights.append(0.0)

      next_sentence_label = 1 if instance.is_random_next else 0

      input_ids_collection.append(np.array(input_ids).astype('int64'))
      input_mask_collection.append(np.array(input_mask).astype('int64'))
      segment_ids_collection.append(np.array(segment_ids).astype('int64'))
      masked_lm_positions_collection.append(np.array(masked_lm_positions).astype('int64'))
      masked_lm_ids_collection.append(np.array(masked_lm_ids).astype('int64'))
      masked_lm_weights_collection.append(np.array(masked_lm_weights).astype('int64'))
      next_sentence_label_collection.append(np.array([next_sentence_label]).astype('int64'))
    pickle.dump((input_ids_collection,
                 input_mask_collection,
                 segment_ids_collection,
                 masked_lm_positions_collection,
                 masked_lm_ids_collection,
                 masked_lm_weights_collection,
                 next_sentence_label_collection),
                outf)

def convert_tokens_to_ids(tokens,vocab):
  output = []
  for token in tokens:
    output.append(vocab[token])
  return output

def split_sentence(data, config):
  new_data = []
  for review in data:
    new_review = []
    for sentence in review:
      sentence = sentence.split(' ')
      if len(sentence) > config['corpus']['max_sentence_len']:
        multiple = len(sentence) // config['corpus']['max_sentence_len']
        mod = len(sentence) % config['corpus']['max_sentence_len']
        if mod == 0:
          rng = multiple
        else:
          rng = multiple + 1
        for i in range(rng):
          start = i * config['corpus']['max_sentence_len']
          stop = start + config['corpus']['max_sentence_len'] - 1
          new_review.append(' '.join(sentence[start:stop]))
      else:
        new_review.append(' '.join(sentence))
    new_data.append(new_review)
  return new_data

def prepare_corpus(config):
  """
  prepare vocab, all documents. eliminate words which is too long(max 11)
  :return:
  """
  # RECTIFY: bert will merge the review to a large sentence TODO: cut a long sentence to pieces with maximum length as 200
  all_documents = []
  word_counts = Counter()
  for input_filePath in config['corpus']['input_filePaths']:
    review_collection = pd.read_pickle(input_filePath)[:,1]
    review_collection = split_sentence(review_collection,config)
    for review in review_collection:
      all_documents.append([])
      for sentence in review:
        sentence = sentence.split(' ')
        for i in range(len(sentence)):
          word = sentence[i]
          if len(list(word))>config['corpus']['max_word_len']:
            sentence[i] = config['corpus']['unknown_word']
        word_counts.update(sentence)
        all_documents[-1].append(sentence)

  words = [word for word, count in word_counts.most_common(config['corpus']['vocab_size'])
                  if count >= config['corpus']['min_word_occurance']]
  words2 = ['[PAD]','[UNK]','[MASK]','[CLS]','[SEP]']
  words2.extend(words)
  # generate word vocabulary
  word_to_id = {word: i for i, word in enumerate(words2)}
  print('word vocab size: ', len(word_to_id))
  print('[PAD]: ', word_to_id['[PAD]'])
  print('[UNK]: ', word_to_id['[UNK]'])
  print('[MASK]: ', word_to_id['[MASK]'])
  print('[CLS]: ', word_to_id['[CLS]'])
  print('[SEP]: ', word_to_id['[SEP]'])
  return all_documents,word_to_id



def create_training_instances(all_documents,vocab, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng):
  """Create `TrainingInstance`s from raw text."""

  # Input file format:
  # (1) One sentence per line. These should ideally be actual sentences, not
  # entire paragraphs or arbitrary spans of text. (Because we use the
  # sentence boundaries for the "next sentence prediction" task).
  # (2) Blank lines between documents. Document boundaries are needed so
  # that the "next sentence prediction" task doesn't span between documents.

  rng.shuffle(all_documents)

  vocab_words = list(vocab.keys())
  instances = []
  for _ in range(dupe_factor):
    for document_index in range(len(all_documents)):
      instances.extend(
          create_instances_from_document(
              all_documents, document_index, max_seq_length, short_seq_prob,
              masked_lm_prob, max_predictions_per_seq, vocab_words, rng))
  rng.shuffle(instances)
  return instances


def create_instances_from_document(
    all_documents, document_index, max_seq_length, short_seq_prob,
    masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
  """Creates `TrainingInstance`s for a single document."""
  document = all_documents[document_index]

  # Account for [CLS], [SEP], [SEP]
  max_num_tokens = max_seq_length - 3

  # We *usually* want to fill up the entire sequence since we are padding
  # to `max_seq_length` anyways, so short sequences are generally wasted
  # computation. However, we *sometimes*
  # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
  # sequences to minimize the mismatch between pre-training and fine-tuning.
  # The `target_seq_length` is just a rough target however, whereas
  # `max_seq_length` is a hard limit.
  target_seq_length = max_num_tokens
  if rng.random() < short_seq_prob:
    target_seq_length = rng.randint(2, max_num_tokens)

  # We DON'T just concatenate all of the tokens from a document into a long
  # sequence and choose an arbitrary split point because this would make the
  # next sentence prediction task too easy. Instead, we split the input into
  # segments "A" and "B" based on the actual "sentences" provided by the user
  # input.
  instances = []
  current_chunk = []
  current_length = 0
  i = 0
  while i < len(document):
    segment = document[i]
    current_chunk.append(segment)
    current_length += len(segment)
    if i == len(document) - 1 or current_length >= target_seq_length:
      if current_chunk:
        # `a_end` is how many segments from `current_chunk` go into the `A`
        # (first) sentence.
        a_end = 1
        if len(current_chunk) >= 2:
          a_end = rng.randint(1, len(current_chunk) - 1)

        tokens_a = []
        for j in range(a_end):
          tokens_a.extend(current_chunk[j])

        tokens_b = []
        # Random next
        is_random_next = False
        if len(current_chunk) == 1 or rng.random() < 0.5:
          is_random_next = True
          target_b_length = target_seq_length - len(tokens_a)

          # This should rarely go for more than one iteration for large
          # corpora. However, just to be careful, we try to make sure that
          # the random document is not the same as the document
          # we're processing.
          for _ in range(10):
            random_document_index = rng.randint(0, len(all_documents) - 1)
            if random_document_index != document_index:
              break

          random_document = all_documents[random_document_index]
          random_start = rng.randint(0, len(random_document) - 1)
          for j in range(random_start, len(random_document)):
            tokens_b.extend(random_document[j])
            if len(tokens_b) >= target_b_length:
              break
          # We didn't actually use these segments so we "put them back" so
          # they don't go to waste.
          num_unused_segments = len(current_chunk) - a_end
          i -= num_unused_segments
        # Actual next
        else:
          is_random_next = False
          for j in range(a_end, len(current_chunk)):
            tokens_b.extend(current_chunk[j])
        truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

        assert len(tokens_a) >= 1
        assert len(tokens_b) >= 1

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
          tokens.append(token)
          segment_ids.append(0)

        tokens.append("[SEP]")
        segment_ids.append(0)

        for token in tokens_b:
          tokens.append(token)
          segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        (tokens, masked_lm_positions,
         masked_lm_labels) = create_masked_lm_predictions(
             tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
        instance = TrainingInstance(
            tokens=tokens,
            segment_ids=segment_ids,
            is_random_next=is_random_next,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        instances.append(instance)
      current_chunk = []
      current_length = 0
    i += 1

  return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
  """Creates the predictions for the masked LM objective."""

  cand_indexes = []
  for (i, token) in enumerate(tokens):
    if token == "[CLS]" or token == "[SEP]":
      continue
    cand_indexes.append(i)

  rng.shuffle(cand_indexes)

  output_tokens = list(tokens)

  num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(tokens) * masked_lm_prob))))

  masked_lms = []
  covered_indexes = set()
  for index in cand_indexes:
    if len(masked_lms) >= num_to_predict:
      break
    if index in covered_indexes:
      continue
    covered_indexes.add(index)

    masked_token = None
    # 80% of the time, replace with [MASK]
    if rng.random() < 0.8:
      masked_token = "[MASK]"
    else:
      # 10% of the time, keep original
      if rng.random() < 0.5:
        masked_token = tokens[index]
      # 10% of the time, replace with random word
      else:
        masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

    output_tokens[index] = masked_token

    masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)

  return (output_tokens, masked_lm_positions, masked_lm_labels)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
  """Truncates a pair of sequences to a maximum sequence length."""
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_num_tokens:
      break

    trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
    assert len(trunc_tokens) >= 1

    # We want to sometimes truncate from the front and sometimes from the
    # back to add more randomness and avoid biases.
    if rng.random() < 0.5:
      del trunc_tokens[0]
    else:
      trunc_tokens.pop()


def main(config):
  rng = random.Random(config['training_data']['random_seed'])
  all_documents,vocab = prepare_corpus(config)
  print('prepared corpus')
  print('len all documents: ',len(all_documents))
  mod = 2
  step = int(len(all_documents)/mod)

  for i in range(mod):
    start = i*step
    end = start+step-1
    print('start: %d, end: %d'%(start,end))
    part_of_all_documents = all_documents[start:end]
    instances = create_training_instances(
        part_of_all_documents, vocab, config['corpus']['max_sentence_len'], config['training_data']['dupe_factor'],
        config['training_data']['short_seq_prob'], config['training_data']['masked_lm_prob'],
        config['training_data']['max_predictions_per_seq'],rng)
    print("instance sample 77: ",instances[77])
    # DONE: check [MASK],[CLS],[SEP]. add ['[PAD]','[UNK]','[MASK]','[CLS]','[SEP]'] at top
    # DONE: check where the max sequence length is used. It seems that in instance, the document is merged to
    # DONE: one big sentence.
    write_instance_to_example_files(instances, vocab, config['corpus']['max_sentence_len'],
                                    config['training_data']['max_predictions_per_seq'],
                                    config['training_data']['output_file']%i)


if __name__ == "__main__":
  # DONE: check where to convert token word to id, especially [CLS],[SEP], [MASK]. there is a function which will convert token to id
  # DONE: check it use multiple sentences or merge them to a big sentence. Merge a review to a big sentence.
  config = {'corpus':{'input_filePaths':[
                                         '/datastore/liu121/sentidata2/data/meituan_jieba/testa_cut.pkl',
                                         '/datastore/liu121/sentidata2/data/meituan_jieba/testb_cut.pkl',
                                         '/datastore/liu121/sentidata2/data/meituan_jieba/train_cut.pkl',
                                         '/datastore/liu121/sentidata2/data/meituan_jieba/val_cut.pkl',
                                        ],
                      'vocab_size':2000000,
                      'min_word_occurance':1,
                      'max_word_len':11,
                      'unknown_word':'[UNK]',
                      'max_sentence_len':1144},
            'training_data':{'dupe_factor':10,
                             'short_seq_prob':0.1,
                             'masked_lm_prob':0.15,
                             'max_predictions_per_seq':20,
                             'random_seed':12345,
                             'output_file':'/datastore/liu121/bert/train_data_%d.pkl'}
            }
  main(config)