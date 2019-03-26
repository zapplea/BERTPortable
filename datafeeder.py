import numpy as np
import pickle

class Dataset:
    def __init__(self, input_ids, input_mask, segment_ids,masked_lm_positions,masked_lm_ids, masked_lm_weights, next_sentence_labels, **kwargs):
        self.dataset_len = len(input_ids)
        self.batch_size = kwargs['batch_size']
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_ids = masked_lm_ids
        self.masked_lm_weights = masked_lm_weights
        self.next_sentence_labels = next_sentence_labels
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.count < self.dataset_len:
            if self.count + self.batch_size < self.dataset_len:
                input_ids_batch=self.input_ids[self.count: self.count + self.batch_size]
                input_mask_batch = self.input_mask[self.count:self.count + self.batch_size]
                segment_ids_batch = self.segment_ids[self.count:self.count + self.batch_size]
                masked_lm_positions_batch = self.masked_lm_positions[self.count:self.count + self.batch_size]
                masked_lm_ids_batch = self.masked_lm_ids[self.count:self.count + self.batch_size]
                masked_lm_weights_batch = self.masked_lm_weights[self.count:self.count + self.batch_size]
                next_sentence_labels_batch = self.next_sentence_labels[self.count:self.count + self.batch_size]
            else:
                input_ids_batch = self.input_ids[self.count:]
                input_mask_batch = self.input_mask[self.count:]
                segment_ids_batch = self.segment_ids[self.count:]
                masked_lm_positions_batch = self.masked_lm_positions[self.count:]
                masked_lm_ids_batch = self.masked_lm_ids[self.count:]
                masked_lm_weights_batch = self.masked_lm_weights[self.count:]
                next_sentence_labels_batch = self.next_sentence_labels[self.count:]
            self.count += self.batch_size
        else:
            raise StopIteration
        return input_ids_batch, input_mask_batch, segment_ids_batch, masked_lm_positions_batch, \
               masked_lm_ids_batch, masked_lm_weights_batch, next_sentence_labels_batch

class DataFeeder:
    def __init__(self,config,index):
        self.config = config

        self.train_input_ids,self.train_input_mask,self.train_segment_ids,\
        self.train_masked_lm_positions,self.train_masked_lm_ids, self.train_masked_lm_weights, \
        self.train_next_sentence_labels = self.load_data(index)


    def load_data(self,index):
        with open(self.config['data']['train_dataset_filePath']%index,'rb') as f:
            # the data is tuple
            data = pickle.load(f)
        return data

    def dataset_generator(self):
        dataset = Dataset(self.train_input_ids,self.train_input_mask,self.train_segment_ids,
                          self.train_masked_lm_positions,self.train_masked_lm_ids, self.train_masked_lm_weights,
                          self.train_next_sentence_labels,batch_size=self.config['data']['batch_size'])
        return dataset