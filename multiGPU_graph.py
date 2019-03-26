import tensorflow as tf
import modeling
import optimization

class GraphBuilder:
    def __init__(self,config):
        self.config = config

    def average_gradients(self, tower_grads):
        # calculate average gradient for each shared variable across all GPUs
        # shape of tower_grads: [((grad0_gpu0, var0_gpu0), (grad1_gpu0,var1_gpu0),...),
        #                        ((grad0_gpu1, var0_gpu1), (grad1_gpu1,var1_gpu1),...),
        #                        ((grad0_gpu2, var0_gpu2), (grad1_gpu2,var1_gpu2),...),
        #                        ...]
        # zip(tower_grads)-->[((grad0_gpu0, var0_gpu0),(grad0_gpu1, var0_gpu1),(grad0_gpu2, var0_gpu2)),
        #                     ((grad1_gpu0, var1_gpu0),(grad1_gpu1, var1_gpu1),(grad1_gpu2, var0_gpu2)),
        #                     ... ...]
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            # We need to average the gradients across each GPU.

            g0, v0 = grad_and_vars[0]

            if g0 is None:
                # no gradient for this variable, skip it
                average_grads.append((g0, v0))
                continue

            if isinstance(g0, tf.IndexedSlices):
                # If the gradient is type IndexedSlices then this is a sparse
                #   gradient with attributes indices and values.
                # To average, need to concat them individually then create
                #   a new IndexedSlices object.
                indices = []
                values = []
                for g, v in grad_and_vars:
                    indices.append(g.indices)
                    values.append(g.values)
                all_indices = tf.concat(indices, 0)
                avg_values = tf.concat(values, 0) / len(grad_and_vars)
                # deduplicate across indices
                av, ai = self._deduplicate_indexed_slices(avg_values, all_indices)
                grad = tf.IndexedSlices(av, ai, dense_shape=g0.dense_shape)

            else:
                # a normal tensor can just do a simple average
                grads = []
                for g, v in grad_and_vars:
                    # Add 0 dimension to the gradients to represent the tower.
                    expanded_g = tf.expand_dims(g, 0)
                    # Append on a 'tower' dimension which we will average over
                    grads.append(expanded_g)

                # Average over the 'tower' dimension.
                grad = tf.concat(grads, 0)
                grad = tf.reduce_mean(grad, 0)

            # the Variables are redundant because they are shared
            # across towers. So.. just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)

            average_grads.append(grad_and_var)

        assert len(average_grads) == len(list(zip(*tower_grads)))

        return average_grads

    def _deduplicate_indexed_slices(self,values, indices):
        """Sums `values` associated with any non-unique `indices`.
        Args:
          values: A `Tensor` with rank >= 1.
          indices: A one-dimensional integer `Tensor`, indexing into the first
          dimension of `values` (as in an IndexedSlices object).
        Returns:
          A tuple of (`summed_values`, `unique_indices`) where `unique_indices` is a
          de-duplicated version of `indices` and `summed_values` contains the sum of
          `values` slices associated with each unique index.
        """
        unique_indices, new_index_positions = tf.unique(indices)
        summed_values = tf.unsorted_segment_sum(
            values, new_index_positions,
            tf.shape(unique_indices)[0])
        return (summed_values, unique_indices)

    def clip_grads(self, grads, all_clip_norm_val = 1.0,):
        # FIXME: delete everything about do_summaries. We don't need it in here.
        # grads = [(grad1, var1), (grad2, var2), ...]
        def _clip_norms(grad_and_vars, val,):
            # grad_and_vars is a list of (g, v) pairs
            grad_tensors = [g for g, v in grad_and_vars]
            scaled_val = val

            clipped_tensors, g_norm = tf.clip_by_global_norm(grad_tensors, scaled_val)

            ret = []
            for t, (g, v) in zip(clipped_tensors, grad_and_vars):
                ret.append((t, v))

            return ret

        ret = _clip_norms(grads, all_clip_norm_val,)

        assert len(ret) == len(grads)

        return ret

    def average_loss(self, graph, gpu_num):
        # FIXME: need to fix
        attr_total_loss = []
        senti_total_loss = []
        joint_total_loss = []
        for i in range(gpu_num):
            attr_total_loss.append(graph.get_collection('attr_loss')[i])
            senti_total_loss.append(graph.get_collection('senti_loss')[i])
            joint_total_loss.append(graph.get_collection('joint_loss')[i])
        attr_loss = tf.reduce_mean(attr_total_loss, axis=0)
        senti_loss = tf.reduce_mean(senti_total_loss,axis=0)
        joint_loss = tf.reduce_mean(joint_total_loss, axis=0)
        return attr_loss, senti_loss, joint_loss

    def compute_grads(self,loss,tower_grads):
        # all var
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # attribute
        var_list = []
        for var in vars:
            var_list.append(var)
            print(var.name)
        print('==========================')

        grads = tf.gradients(loss,var_list=var_list,aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
        tower_grads.append(grads)



    def build_graph(self):
        tower_grads = []
        tower_inputs = []
        with tf.get_default_graph().device('/cpu:0'):
            for k in range(self.config['model']['gpu_num']):
                with tf.device('/gpu:%d' % k):
                    print('gpu No.: %d'%k)
                    with tf.variable_scope('BERT', reuse=k > 0):
                        # TODO: output placeholder.
                        # EXPL: get input
                        input_ids, input_mask, segment_ids, \
                        masked_lm_positions, masked_lm_ids, masked_lm_weights, \
                        next_sentence_labels = modeling.input_fn(self.config)
                        tower_inputs.append({'input_ids':input_ids,
                                             'input_mask':input_mask,
                                             'segment_ids':segment_ids,
                                             'masked_lm_positions':masked_lm_positions,
                                             'masked_lm_ids':masked_lm_ids,
                                             'masked_lm_weights':masked_lm_weights,
                                             'next_sentence_labels':next_sentence_labels})
                        # EXPL: get model
                        model = modeling.BertModel(self.config,input_ids=input_ids,input_mask=input_mask,token_type_ids=segment_ids,scope='bert')
                        # EXPL: get masked loss
                        (masked_lm_loss,masked_lm_example_loss, masked_lm_log_probs) = \
                            get_masked_lm_output(self.config, model.get_sequence_output(),
                                                 model.get_embedding_table(),
                                                 masked_lm_positions,
                                                 masked_lm_ids,
                                                 masked_lm_weights)
                        # EXPL: get next sentence loss
                        (next_sentence_loss, next_sentence_example_loss,next_sentence_log_probs) = \
                            get_next_sentence_output(self.config, model.get_pooled_output(),
                                                     next_sentence_labels)
                        # EXPL: get loss
                        total_loss = masked_lm_loss + next_sentence_loss
                        # TODO: check the whole net to see whether the variables' name is given.
                        # TODO: set a mechanism to check the variable name.
                        # TODO: test whether name of tf.layers.dense variable has the same name when use twice under the same scope.

                        self.compute_grads(total_loss,tower_grads)
            # TODO: initialize with checkpoint when k == 0
            avg_grads_vars = self.average_gradients(tower_grads)
            global_step = tf.train.get_or_create_global_step()
            opt = optimization.create_optimizer(init_lr=self.config['model']['lr'],
                                                num_train_steps=self.config['model']['epoch'],
                                                num_warmup_steps=self.config['model']['num_warmup_steps'],
                                                global_step=global_step)
            train_op = opt.apply_gradients(avg_grads_vars, global_step=global_step)
            new_global_step = global_step + 1
            train_op = tf.group(train_op, [global_step.assign(new_global_step)])

            # TODO: check the shape of these values and then merge results from different gpu to get the eval.
            # eval_metrics = (metric_fn, [
            #     masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
            #     masked_lm_weights, next_sentence_example_loss,
            #     next_sentence_log_probs, next_sentence_labels
            # ])

        return {'train_op':train_op,'tower_inputs':tower_inputs}

def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
              masked_lm_weights, next_sentence_example_loss,
              next_sentence_log_probs, next_sentence_labels):
    """Computes the loss and accuracy of the model."""
    masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                     [-1, masked_lm_log_probs.shape[-1]])
    masked_lm_predictions = tf.argmax(
        masked_lm_log_probs, axis=-1, output_type=tf.int32)
    masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
    masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
    masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
    masked_lm_accuracy = tf.metrics.accuracy(
        labels=masked_lm_ids,
        predictions=masked_lm_predictions,
        weights=masked_lm_weights)
    masked_lm_mean_loss = tf.metrics.mean(
        values=masked_lm_example_loss, weights=masked_lm_weights)

    next_sentence_log_probs = tf.reshape(
        next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
    next_sentence_predictions = tf.argmax(
        next_sentence_log_probs, axis=-1, output_type=tf.int32)
    next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
    next_sentence_accuracy = tf.metrics.accuracy(
        labels=next_sentence_labels, predictions=next_sentence_predictions)
    next_sentence_mean_loss = tf.metrics.mean(
        values=next_sentence_example_loss)

    return {
        "masked_lm_accuracy": masked_lm_accuracy,
        "masked_lm_loss": masked_lm_mean_loss,
        "next_sentence_accuracy": next_sentence_accuracy,
        "next_sentence_loss": next_sentence_mean_loss,
    }

def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config['model']['hidden_size'],
          activation=modeling.get_activation(bert_config['model']['hidden_act']),
          kernel_initializer=modeling.create_initializer(
              bert_config['model']['initializer_range']))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config['model']['vocab_size']],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config['model']['vocab_size'], dtype=tf.float32)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(tf.cast(label_weights,dtype='float32') * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + tf.constant(1e-5,dtype='float32')
    loss = numerator / denominator

  return (loss, per_example_loss, log_probs)


def get_next_sentence_output(bert_config, input_tensor, labels):
  """Get loss and log probs for the next sentence prediction."""

  # Simple binary classification. Note that 0 is "next sentence" and 1 is
  # "random sentence". This weight matrix is not used after pre-training.
  with tf.variable_scope("cls/seq_relationship"):
    output_weights = tf.get_variable(
        "output_weights",
        shape=[2, bert_config['model']['hidden_size']],
        initializer=modeling.create_initializer(bert_config['model']['initializer_range']))
    output_bias = tf.get_variable(
        "output_bias", shape=[2], initializer=tf.zeros_initializer())

    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    labels = tf.reshape(labels, [-1])
    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + tf.cast(flat_offsets,dtype='int64'), [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor