from __future__ import division
import numpy as np
import tensorflow as tf
import logging
import tqdm
from sklearn.metrics import classification_report,accuracy_score
from cell import biLSTM, AttentionCell, MatchLSTMCell, get_last_layer,padding_batch
from data_util import minibatches
from sklearn.metrics import classification_report,accuracy_score

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

from ptpython.repl import embed


class InferModel(object):
    def __init__(self, *args):
        self.config = args[0]
        self.glove_embeddings = args[1]
        self.vocab = args[2]
        self.embedding_size = self.config.embedding_size
        self.state_size = self.config.state_size

        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)

        N = self.config.batch_size
        self.q = tf.placeholder(tf.int64, [None, None], name='q')
        self.x = tf.placeholder(tf.int64, [None, None], name='x')
        self.fx = tf.placeholder(tf.int64, [None, None], name ='fx')
        self.q_mask = tf.placeholder(tf.bool, [None, None], name='q_mask')
        self.x_mask = tf.placeholder(tf.bool, [None, None], name='x_mask')
        self.fx_mask = tf.placeholder(tf.bool, [None, None], name='fx_mask')
        self.a = tf.placeholder(tf.int32, [None, None], name='a')
        self.a_mask = tf.placeholder(tf.bool, [None, None], name='a_mask')
        self.y = tf.placeholder(tf.int64, [None], name='y')
        self.JX = tf.placeholder(tf.int32, shape=(), name='JX')
        self.JQ = tf.placeholder(tf.int32, shape=(), name='JQ')
        self.JA = tf.placeholder(tf.int32, shape=(), name='JA')
        self.JFX = tf.placeholder(tf.int32, shape=(), name='JFX')
        self.learning_rate = tf.placeholder(tf.float32,shape=(),name="learning_rate")
        self.keep_prob = tf.placeholder(tf.float32,shape=(),name="keep_prob")

        # TODO Decay learning rate

        self.data_size = tf.constant(self.config.dataset_size,name="dataset_size")
        self.word_emb_mat = tf.get_variable(dtype=tf.float32, initializer=self.glove_embeddings, name="word_emb_mat")
        self.fx_emb_mat = tf.Variable(tf.truncated_normal([2,15]), name='fx_emb')

        with tf.variable_scope("infer"):
            question, context, answer = self.setup_embeddings() #[ N,?,100]
            fx = self.setup_feature_embeddings() #[N,JFX,15]

            with tf.variable_scope("concatenate_context_embeddings"):
                '''
                Concatenate overlap features
                '''
                f1,f2,f3 = tf.split(fx,3,axis=1)
                if self.config.overlap:
                    logger.info("Using overlap features")
                    context = tf.concat([context,f1,f2,f3],axis=2)

            with tf.variable_scope("encoder"):

                logger.info("Encoder")
                self.question_repr = self.encode(question,self.q_mask,scope="question_repr")#[N,JQ,2d]
                self.context_repr = self.encode(context,self.x_mask,scope="context_repr")#[N,JX,2d]
                self.answer_repr = self.encode(answer,self.a_mask,scope="answer_repr")#[N,JA,2d]

            with tf.variable_scope("aligned_question_embeddings"):
                '''
                Aligned Question Embeddings from arXiv:1704.00051v2
                captures soft alignment between embeddings
                '''
                if self.config.ques_aligned_context:
                    logger.info("Using Question Aligned embeddings")
                    question_repr_t = tf.transpose(self.question_repr,[0,2,1]) #[N,2d,JQ]
                    aligned_attention = tf.exp(tf.matmul(self.context_repr,question_repr_t)) #[N,JX,JQ]
                    aligned_attention_sum = tf.tile(tf.reduce_sum(aligned_attention,axis=2,keep_dims=True),[1,1,self.JQ])
                    self.q_aligned_attn = tf.divide(aligned_attention,tf.subtract(aligned_attention_sum,aligned_attention),name="q_alignments") #[N,JX,JQ]
                    self.q_aligned_context = tf.multiply(self.context_repr, tf.reduce_sum(self.q_aligned_attn,2,keep_dims=True)) #[N,JX,2d]
                    # Concatenating with context_repr
                    self.context_repr = tf.add(self.context_repr,self.q_aligned_context)

            # Multiple hops over attention (ReasoNet)
            with tf.variable_scope("question_over_context") as scope:
                '''
                Multihop model based on ReasoNet arXiv:1609.05284v3
                '''

                # Question over context attention
                state_qx_fw, state_qx_bw = tf.split(self.context_repr, 2, axis=2)
                attention_cell_qx_fw = MatchLSTMCell(state_size=self.state_size, state_is_tuple=True, encoder_input=state_qx_fw,
                                                  encoder_input_size= self.JX, encoder_mask=self.x_mask)
                attention_cell_qx_bw = MatchLSTMCell(state_size=self.state_size, state_is_tuple=True,
                                                  encoder_input=state_qx_bw, encoder_input_size=self.JX, encoder_mask=self.x_mask)

                #  Context over question attention
                state_xq_fw, state_xq_bw = tf.split(self.question_repr, 2, axis=2)
                attention_cell_xq_fw = MatchLSTMCell(state_size=self.state_size, state_is_tuple=True, encoder_input=state_xq_fw,
                                                  encoder_input_size= self.JQ, encoder_mask=self.q_mask)
                attention_cell_xq_bw = MatchLSTMCell(state_size=self.state_size, state_is_tuple=True,
                                                  encoder_input=state_xq_bw, encoder_input_size=self.JQ, encoder_mask=self.q_mask)

                self.config.include_answer = False
                if self.config.include_answer:
                    attend_on_x = tf.concat([question, answer], 1)  # [N,JA+JQ,2d]
                    attend_on_x_mask = tf.concat([self.q_mask, self.a_mask], 1)  #[N,JA+JQ]
                else:
                    attend_on_x = self.question_repr  # [N,JQ,2d]
                    attend_on_x_mask = self.q_mask  #[N,JQ]

                attend_on_q = self.context_repr #[N,JX,2d]
                attend_on_q_mask = self.x_mask

                if self.config.num_hops > 0 :
                    logger.info("Multi Hops")
                    gru_cell = tf.contrib.rnn.GRUCell(2*self.config.state_size)
                    qseq_len = tf.reshape(tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1), [-1, ])
                    # Initalize the first state of gru state
                    self.inner_state = get_last_layer(self.question_repr,qseq_len-1)
                    for i in range(self.config.num_hops):
                        logger.info("hop %d"%i)
                        logger.info("question over context attention")
                        with tf.variable_scope("question_over_context"):
                            self.q_on_x, _, _ = biLSTM(attend_on_x, attend_on_x_mask,
                                                              cell_fw=attention_cell_qx_fw, cell_bw=attention_cell_qx_bw,
                                                              dropout=self.keep_prob, state_size=self.config.state_size)  #[N,JQ,2*d]

                        logger.info("context over question attention")
                        with tf.variable_scope("context_over_question"):
                            self.x_on_q, _, _ = biLSTM(attend_on_q, attend_on_q_mask,
                                                              cell_fw=attention_cell_xq_fw, cell_bw=attention_cell_xq_bw,
                                                              dropout=self.keep_prob, state_size=self.config.state_size)  #[N,JX,2*d]
                        concat_attentions = tf.concat([self.q_on_x,self.x_on_q],1) #[N, JX + JQ, 2*d]
                        reduced_attention = tf.reduce_sum(concat_attentions,1) #[N,2d]
                        # Calculate new inner state
                        _,self.inner_state = gru_cell(reduced_attention,self.inner_state)
                        # adjust question repr with inner state
                        attend_on_x = self.question_repr * tf.expand_dims(self.inner_state,1)
                        attend_on_q = self.context_repr * tf.expand_dims(self.inner_state,1)
                        scope.reuse_variables()
                    # get final attended layer
                    self.attended_repr = self.q_on_x #[N,JQ,2*d]
                else:
                    logger.info("Single Hops")
                    logger.info("question over context attention")
                    self.attended_repr, _, _ = biLSTM(attend_on_x, attend_on_x_mask,
                                                      cell_fw=attention_cell_qx_fw, cell_bw=attention_cell_qx_bw,
                                                      dropout=self.keep_prob, state_size=self.config.state_size)  #[N,JQ,2*d]

            with tf.variable_scope("self_alignment"):
                '''
                Self Alignment Layer from arXiv:1705.02798v1
                '''
                if self.config.self_alignment:
                    logger.info("Self Alignment Layer")
                    attended_repr_t = tf.transpose(self.attended_repr,[0,2,1])
                    self.self_alignment = tf.matmul(self.attended_repr,attended_repr_t) # [N,JQ,JQ]
                    diag = tf.ones_like(tf.cast(self.q_mask,tf.float32))
                    self.self_alignment = tf.transpose(tf.matrix_set_diag(self.self_alignment,diag),[0,2,1],name="self_alignment")
                    self.self_qalign_repr = tf.matmul(self.self_alignment,self.question_repr,name="qalign")

            with tf.variable_scope("aggregation_layer"):
                '''
                aggregation of context aware question and self aligned questions
                '''
                if self.config.self_alignment:
                    self.final_layer = tf.concat([self.attended_repr,self.self_qalign_repr],1)
                    self.final_mask = tf.concat([self.q_mask,self.q_mask],1)
                else:
                    self.final_layer = self.attended_repr
                    self.final_mask = self.q_mask

            if self.config.use_decoder:
                logger.info("Decoding")
                self.decode_repr = self.decode(self.final_layer,self.final_mask)
                decode_len = tf.reshape(tf.reduce_sum(tf.cast(self.final_mask, tf.int32), axis=1), [-1, ])
            else:
                self.decode_repr = self.final_layer
                decode_len = tf.reshape(tf.reduce_sum(tf.cast(self.final_mask, tf.int32), axis=1), [-1, ])

            self.preds = get_last_layer(self.decode_repr, decode_len-1)

            with tf.variable_scope("MLP_layer"):
                self.logits = tf.contrib.layers.fully_connected(self.preds, self.config.num_classes)

            self.pred_softmax= tf.nn.softmax(self.logits, name="pred_softmax")

            self.loss = self.setup_loss(self.logits, self.final_mask)

            self.prediction = tf.argmax(self.pred_softmax, 1,name="prediction")

            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.y), tf.float32))

            #self.train_op = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999).minimize(self.loss)
            # Gradient clipping
            optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999)
            grads = optimizer.compute_gradients(self.loss)
            clipped_grads = [(tf.clip_by_value(grad, -1.0, 1.0), var) if grad != None else (grad, var) for grad, var in grads]
            self.train_op = optimizer.apply_gradients(clipped_grads)

            tf.summary.scalar('accuracy', self.accuracy)
            '''
            tf.summary.histogram('logits', self.logits)
            tf.summary.histogram('prediction', self.prediction)
            for var in tf.trainable_variables():
                tf.summary.histogram(var.name.replace(':', '_'), var)

            grads = tf.gradients(self.loss, tf.trainable_variables())
            for grad in grads: # none issues
                if grad == None:
                    continue
                else:
                    tf.summary.histogram(grad.name.replace(':', '_'), grad)
                pass
            '''
            self.inc_step = self.global_step.assign_add(1)
            self.summary_op = tf.summary.merge_all()

    def setup_loss(self, logits, mask):
        onehot_labels = tf.one_hot(self.y, 2)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=self.y))
        weights = [v for v in tf.trainable_variables() if 'bias' not in v.name]
        l2_loss = tf.add_n([tf.nn.l2_loss(weight) for weight in weights])
        loss = loss + self.config.l2_beta * l2_loss
        tf.summary.scalar('loss', loss)

        return loss

    def decode(self, input, input_mask):
        """
        Couple of bilstm layers stacked
        """
        with tf.variable_scope("decoder_layer1") as scope:
            layer1, _, _ = biLSTM(input, input_mask, self.config.state_size, dropout=self.keep_prob, scope=scope)

        with tf.variable_scope("decoder_layer2") as scope:
            layer2, _, _ = biLSTM(layer1, input_mask, self.config.state_size, dropout=self.keep_prob, scope=scope)
        return layer2

    def encode(self, dist, mask, scope):
        """
        Returns encoded representation of inputs
        """
        #TODO share parmeters, to learn similar representations
        with tf.variable_scope(scope):
            print('-'*5 + scope + '-'*5)
            repr, _, _ = biLSTM(dist,mask, dropout=self.keep_prob, state_size=self.config.state_size, scope=scope)
            print(repr)
        return repr

    def setup_embeddings(self):
        """
        return glove pretrained embeddings for inputs
        """
        with tf.variable_scope("emb"), tf.device("/cpu:0"):
            question = tf.nn.embedding_lookup(self.word_emb_mat, self.q)
            context = tf.nn.embedding_lookup(self.word_emb_mat, self.x)
            answer = tf.nn.embedding_lookup(self.word_emb_mat, self.a)
            question = tf.reshape(question, [-1, self.JQ, self.embedding_size])
            context = tf.reshape(context, [-1, self.JX, self.embedding_size])
            answer = tf.reshape(answer, [-1, self.JA, self.embedding_size])
            return question, context, answer

    def setup_feature_embeddings(self):
        '''
        15 dimensional random embedding
        '''
        fx = tf.nn.embedding_lookup(self.fx_emb_mat,self.fx)
        fx = tf.reshape(fx,[-1,3*self.JX,15])
        return fx

    def create_feed_dict(self, question_batch, question_len_batch, context_batch, context_len_batch,
                         context_features_batch, context_features_len_batch, ans_batch, ans_len_batch, label_batch=None):

        feed_dict = {}
        JQ = np.max(question_len_batch)
        JX = np.max(context_len_batch)
        JA = np.max(ans_len_batch)
        JFX = np.max(context_features_len_batch)

        question, question_mask = padding_batch(question_batch, JQ)
        context, context_mask = padding_batch(context_batch, JX)
        answer, answer_mask = padding_batch(ans_batch, JA)
        context_features, context_features_mask = padding_batch(context_features_batch, 3*JX)

        feed_dict[self.q] =question
        feed_dict[self.q_mask] = question_mask
        feed_dict[self.x] = context
        feed_dict[self.x_mask] = context_mask
        feed_dict[self.fx] = context_features
        feed_dict[self.fx_mask] = context_features_mask
        feed_dict[self.a] = answer
        feed_dict[self.a_mask] = answer_mask
        feed_dict[self.JQ] = JQ
        feed_dict[self.JX] = JX
        feed_dict[self.JA] = JA
        feed_dict[self.JFX] = JFX
        feed_dict[self.learning_rate] = self.config.learning_rate
        feed_dict[self.keep_prob] = self.config.keep_prob
        if label_batch is not None:
            feed_dict[self.y] = label_batch
        return feed_dict

    def get_loss(self):
        return self.loss

    def get_accuracy(self):
        return self.accuracy

    def get_summary(self):
        return self.summary_op

    def get_global_step(self):
        return self.global_step
