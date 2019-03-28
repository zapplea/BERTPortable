from datafeeder import DataFeeder
import numpy as np

config ={'data':{'batch_size':10,
                      'train_dataset_filePath':'/datastore/liu121/bert_trail/train_data/train_data_%d.pkl',
                      'train_file_num':5
                      }}

for i in range(config['data']['train_file_num']):
    df = DataFeeder(config,i)
    dataset = df.dataset_generator()
    print('total dataset size: ',np.array(df.train_input_ids).shape[0])
    for input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights, next_sentence_labels in dataset:
        print('batch size: ',np.array(input_ids).shape)
