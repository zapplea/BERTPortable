import pickle
import gc

class Cut:
    def __init__(self,config,index):
        self.config = config

        self.train_input_ids,self.train_input_mask,self.train_segment_ids,\
        self.train_masked_lm_positions,self.train_masked_lm_ids, self.train_masked_lm_weights, \
        self.train_next_sentence_labels = self.load_data(index)


    def load_data(self,index):
        print(self.config['data']['train_dataset_filePath']%index)
        with open(self.config['data']['train_dataset_filePath']%index,'rb') as f:
            # the data is tuple
            data = pickle.load(f)
        return data

    def anal(self):
        print(len(self.train_input_ids))

if __name__ == "__main__":
    config = {'data':{'batch_size':10,
                      'train_dataset_filePath':'/datastore/liu121/bert_trail/train_data/train_data_%d.pkl',
                      'train_file_num':5
                      }}
    for i in range(5):
        cut = Cut(config,i)
        cut.anal()
        del cut
        gc.collect()