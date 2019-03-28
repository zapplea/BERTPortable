from datafeeder import DataFeeder
import numpy as np
import math

def generate_feed_dict(config,tower_data):
    train_mod = math.ceil(len(tower_data['input_ids']) / config['model']['gpu_num'])
    for k in range(config['model']['gpu_num']):
        start = k * train_mod
        end = start + train_mod
        for key in tower_data:
            print(np.array(tower_data[key][start:end]).shape)
            break

config ={'data':{'batch_size':10,
                      'train_dataset_filePath':'/datastore/liu121/bert_trail/train_data/train_data_%d.pkl',
                      'train_file_num':5
                      },
         'model':{'gpu_num':2}}

for i in range(config['data']['train_file_num']):
    df = DataFeeder(config,i)
    dataset = df.dataset_generator()
    print('total dataset size: ',np.array(df.train_input_ids).shape[0])
    for input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights, next_sentence_labels in dataset:
        print('batch size: ',np.array(input_ids).shape)
        tower_data = {'input_ids': input_ids,
                      'input_mask': input_mask,
                      'segment_ids': segment_ids,
                      'masked_lm_positions': masked_lm_positions,
                      'masked_lm_ids': masked_lm_ids,
                      'masked_lm_weights': masked_lm_weights,
                      'next_sentence_labels': next_sentence_labels}
        generate_feed_dict(config,tower_data)
        print('=====================================')
    exit()
