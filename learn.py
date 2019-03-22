import multiGPU_graph
import train

def main():
    # TODO: construct metrics
    pass

if __name__ == "__main__":
    config = {'model':{'max_seq_length':None,
                       'max_predictions_per_seq':None,
                       'gpu_num':2,
                       'is_training':True,
                       'hidden_dropout_prob':0.1,
                       'attention_probs_dropout_prob':0.1,
                       'vocab_size':None,
                       'hidden_size':None,
                       'initializer_range':0.02,
                       'type_vocab_size':16,
                       'max_position_embeddings':512, #max seq length
                       'num_hidden_layers':12,
                       'num_attention_heads':12,
                       'hidden_act':'relu', # linear/relu/gelu/tanh
                       'lr':5e-5,
                       'num_warmup_steps':10000,
                       },
              'train':{'epoch':100000,},
              'data':{}
             }