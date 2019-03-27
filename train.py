import tensorflow as tf
import math
import pickle
import numpy as np

class Train:
    def __init__(self,config,datafeeder):
        self.config = config
        self.df = datafeeder
        self.report_file = open(self.config['train']['report_filePath'],'w')

    def report(self,info,file=None,mod='std'):
        if mod=='std':
            print(info)
        else:
            file.write(info+'\n')
            file.flush()

    def train(self,model_dict):
        with tf.get_default_graph().as_default():
            train_op = model_dict['train_op']
            avg_metrics = model_dict['avg_metrics']
            eval_metrics = model_dict['eval_metrics']
            init_global = tf.global_variables_initializer()
            init_local = tf.local_variables_initializer()
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        with tf.Session(graph=tf.get_default_graph(), config=sess_config) as sess:
            sess.run(init_global)
            sess.run(init_local)
            for i in range(self.config['train']['epoch']):
                self.report('epoch: %d'%i,self.report_file,'report')
                self.report('===================',self.report_file,'report')
                metrics_value_ls = []
                for j in range(self.config['data']['train_file_num']):
                    self.report('data file: %s'%self.config['data']['train_dataset_filePath']%j,self.report_file,'report')
                    df = self.df(self.config,j)
                    dataset = df.dataset_generator()
                    count = 0
                    for input_ids,input_mask,segment_ids,masked_lm_positions,masked_lm_ids,masked_lm_weights,next_sentence_labels in dataset:
                        print('count: ',count)
                        count+=1
                        if count == 10:
                            break
                        tower_data = {'input_ids': input_ids,
                                      'input_mask': input_mask,
                                      'segment_ids': segment_ids,
                                      'masked_lm_positions': masked_lm_positions,
                                      'masked_lm_ids': masked_lm_ids,
                                      'masked_lm_weights': masked_lm_weights,
                                      'next_sentence_labels': next_sentence_labels}
                        feed_dict = self.generate_feed_dict(model_dict['tower_inputs'],tower_data)
                        _, avg_metrics_value= sess.run([train_op,avg_metrics],feed_dict=feed_dict)
                        print('eval_metrics type: ',type(eval_metrics))
                        eval_metrics_val = sess.run(eval_metrics,feed_dict=feed_dict)
                        print(eval_metrics_val)
                        print('masked_lm_accuracy shape: ',eval_metrics_val[0].shape)
                        metrics_value_ls.append(avg_metrics_value)
                        mean_metrics = np.mean(metrics_value_ls, axis=0)
                        print('mean_metrics: ',tuple(mean_metrics))
                        print('mean_metrics shape: ',mean_metrics.shape)
                        exit()
                    break
                masked_lm_accuracy,\
                masked_lm_mean_loss,\
                next_sentence_accuracy,\
                next_sentence_mean_loss = tuple(np.mean(metrics_value_ls,axis=0))
                self.report('masked_lm_accuracy: %f'%masked_lm_accuracy, self.report_file, 'report')
                self.report('masked_lm_mean_loss: %f' %masked_lm_mean_loss , self.report_file, 'report')
                self.report('next_sentence_accuracy: %f' %next_sentence_accuracy, self.report_file, 'report')
                self.report('next_sentence_mean_loss: %f' %next_sentence_mean_loss, self.report_file, 'report')
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