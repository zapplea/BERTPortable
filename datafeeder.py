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
    def __init__(self,config):
        self.config = config

        self.train_input_ids,self.train_input_mask,self.train_segment_ids,\
        self.train_masked_lm_positions,self.train_masked_lm_ids, self.train_masked_lm_weights, \
        self.train_next_sentence_labels = self.load_data('train')

        self.val_input_ids, self.val_input_mask, self.val_segment_ids, \
        self.val_masked_lm_positions, self.val_masked_lm_ids, self.val_masked_lm_weights, \
        self.val_next_sentence_labels = self.load_data('val')

    def load_data(self,mod):
        if mod == 'train':
            with open(self.config['data']['train_dataset_filePath'],'rb') as f:
                # the data is dic
                data = pickle.load(f)
        elif mod == 'val':
            with open(self.config['data']['val_dataset_filePath'],'rb') as f:
                # the data is dic
                data = pickle.load(f)
        else:
            raise ValueError('The value of mod can only be train or val')
        return data

    def dataset_generator(self,mod):
        if mod == 'train':
            dataset = Dataset(self.train_input_ids,self.train_input_mask,self.train_segment_ids,
                              self.train_masked_lm_positions,self.train_masked_lm_ids, self.train_masked_lm_weights,
                              self.train_next_sentence_labels,batch_size=self.config['data']['batch_size'])
        elif mod == 'val':
            dataset = Dataset(self.val_input_ids, self.val_input_mask, self.val_segment_ids,
                              self.val_masked_lm_positions, self.val_masked_lm_ids, self.val_masked_lm_weights,
                              self.val_next_sentence_labels,batch_size=self.config['data']['batch_size'])
        else:
            raise ValueError('The value of mod can only be train or val')
        return dataset