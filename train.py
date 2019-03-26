import tensorflow as tf
import math
import pickle

class Train:
    def __init__(self,config,datafeeder):
        self.config = config
        self.df = datafeeder

    def train(self,model_dict):
        train_op = model_dict['train_op']
        init = tf.global_variables_initializer()
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        with tf.Session(graph=tf.get_default_graph(), config=sess_config) as sess:
            sess.run(init)
            for i in range(self.config['train']['epoch']):
                print('epoch: %d'%i)
                for j in range(self.config['data']['train_file_num']):
                    print('data file: ',self.config['data']['train_dataset_filePath']%j)
                    df = self.df(self.config,j)
                    dataset = df.dataset_generator()
                    for input_ids,input_mask,segment_ids,masked_lm_positions,masked_lm_ids,masked_lm_weights,next_sentence_labels in dataset:
                        tower_data = {'input_ids': input_ids,
                                      'input_mask': input_mask,
                                      'segment_ids': segment_ids,
                                      'masked_lm_positions': masked_lm_positions,
                                      'masked_lm_ids': masked_lm_ids,
                                      'masked_lm_weights': masked_lm_weights,
                                      'next_sentence_labels': next_sentence_labels}
                        feed_dict = self.generate_feed_dict(model_dict['tower_inputs'],tower_data)
                        sess.run(train_op,feed_dict=feed_dict)
                # after each epoch, the variable weights will be saved for once.
                self.save_variables_value(sess)

    def generate_feed_dict(self,tower_inputs,tower_data):
        feed_dict = {}
        train_mod = math.ceil(len(tower_data['input_ids']) / self.config['model']['gpu_num'])
        for k in range(self.config['model']['gpu_num']):
            start = k * train_mod
            end = start + train_mod
            for key in tower_data:
                feed_dict[tower_inputs[k][key]] = tower_data[key][start:end]
        return feed_dict

    def save_variables_value(self,sess):
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        values = {}
        for var in vars:
            value = sess.run(var)
            values[var.name] = value
        with open(self.config['train']['varval_filePath']) as f:
            pickle.dump(values,f)
        print('save variables value to: %s'%(self.config['train']['varval_filePath']))