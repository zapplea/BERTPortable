from multiGPU_graph import GraphBuilder
from train import Train
from datafeeder import DataFeeder
import argparse

def main(config):
    # TODO: construct metrics
    # DONE: output all variables value to pickle, so that we can use it to initialize other models.
    train = Train(config,DataFeeder)
    gb = GraphBuilder(config)
    model_dict = gb.build_graph()
    train.train(model_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_num',type=int,default=2)
    args = parser.parse_args()
    config = {'model':{'max_seq_length':1144,
                       'max_predictions_per_seq':20,
                       'gpu_num':args.gpu_num,
                       'is_training':True,
                       'hidden_dropout_prob':0.1,
                       'attention_probs_dropout_prob':0.1,
                       'vocab_size':80335,# 400634
                       'hidden_size':300, # embedding size
                       'initializer_range':0.02,
                       'type_vocab_size':16,
                       'max_position_embeddings':1144, #max seq length
                       'num_hidden_layers':12,
                       'num_attention_heads':12,
                       'hidden_act':'relu', # linear/relu/gelu/tanh
                       'lr':5e-5,
                       'intermediate_size':3072,
                       'num_warmup_steps':10000,
                       },
              'train':{'epoch':100000,
                       'early_stop_limit':5,
                       'varval_filePath':'/datastore/liu121/bert_trail/weights/var_weights_val.pkl',
                       'report_filePath':'/datastore/liu121/bert_trail/report/report.txt'
                       },
              'data':{'batch_size':10,
                      'train_dataset_filePath':'/datastore/liu121/bert_trail/train_data/train_data_%d.pkl',
                      'train_file_num':5
                      }
             }
    main(config)