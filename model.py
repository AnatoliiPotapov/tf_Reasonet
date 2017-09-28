from __future__ import division
import numpy as np
import tensorflow as tf
import logging
import tqdm
from sklearn.metrics import classification_report,accuracy_score
from data_util import minibatches

logging.basicConfig(level=logging.INFO)

from ptpython.repl import embed

def add_paddings(sentence, max_length):
    mask = [True] * len(sentence)
    pad_len = max_length - len(sentence)
    if pad_len > 0:
        padded_sentence = sentence + [0] * pad_len
        mask += [False] * pad_len
    else:
        padded_sentence = sentence[:max_length]
        mask = mask[:max_length]
    return padded_sentence, mask

def padding_batch(data, max_len):
    padded_data = []
    padded_mask = []
    for sentence in data:
        d, m = add_paddings(sentence, max_len)
        padded_data.append(d)
        padded_mask.append(m)
    return (padded_data, padded_mask)


def extract_axis_1(data, ind):
    """
    Get specified elements along the first axis of tensor.
    :param data: Tensorflow tensor that will be subsetted.
    :param ind: Indices to take (one for each element along axis 0 of data).
    :return: Subsetted tensor.
    """

    batch_range = tf.range(tf.shape(data)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)

    return res

class AttentionCell(tf.contrib.rnn.BasicLSTMCell):
    def __init__(self,state_size,state_is_tuple,encoder_input,encoder_input_size):
        self.d = state_size
        self.Y = encoder_input
        self.enc_size = encoder_input_size
        super(AttentionCell,self).__init__(state_size,state_is_tuple)

    def __call__(self,inputs,state):
        with tf.variable_scope("attention_cell"):
            c,h = state
            W_y,W_h,W_p,W_x,w = self.get_weights(self.d)
            m1 = tf.reshape(tf.matmul(tf.reshape(self.Y, [-1,self.d]),W_y),[-1,self.enc_size,self.d],name="m1")
            m2 = tf.expand_dims(tf.matmul(inputs, W_h) ,1,name="m2")
            M = tf.tanh(m1+m2)
            alpha = tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(M,[-1,self.d]),tf.expand_dims(w,1)),[-1,self.enc_size],name="alpha")) #[se]
            r = tf.reshape(tf.matmul(tf.expand_dims(alpha,1),self.Y),[-1,self.d])
            h_star = tf.tanh( tf.matmul(r,W_p) + tf.matmul(inputs,W_x))
            return (h_star, tf.contrib.rnn.LSTMStateTuple(h_star,h_star))

    @staticmethod
    def get_weights(state_size):
        xavier_init= tf.contrib.layers.xavier_initializer()
        zero_init = tf.constant_initializer(0)
        W_y = tf.get_variable("W_y",shape=[state_size,state_size],dtype=tf.float32,initializer=xavier_init)
        W_h = tf.get_variable("W_h",shape=[state_size,state_size],dtype=tf.float32,initializer=xavier_init)
        W_p = tf.get_variable("W_p",shape=[state_size,state_size],dtype=tf.float32,initializer=xavier_init)
        W_x = tf.get_variable("W_x",shape=[state_size,state_size],dtype=tf.float32,initializer=xavier_init)
        w = tf.get_variable("w",shape=[state_size,],dtype=tf.float32,initializer=xavier_init)
        return W_y,W_h,W_p,W_x,w

class Encoder(object):
    def __init__(self,size):
        self.size = size

    def encode(self, inputs, mask, encoder_state_input):

        cell_fw = tf.contrib.rnn.LSTMCell(num_units=self.size, state_is_tuple=True)
        cell_bw = tf.contrib.rnn.LSTMCell(num_units=self.size, state_is_tuple=True)

        if encoder_state_input is not None:
            state_fw = encoder_state_input[0]
            state_bw = encoder_state_input[1]
        else:
            state_fw = None
            state_bw = None
        seq_len = tf.reduce_sum(tf.cast(mask, 'int32'), axis=1)
        seq_len = tf.reshape(seq_len, [-1,])
        (hidden_state_fw,hidden_state_bw),(final_state_fw,final_state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,inputs,seq_len,state_fw,state_bw,tf.float32)

        concat_hidden_states = tf.concat([hidden_state_fw,hidden_state_bw],2)

        concat_final_state = tf.concat([final_state_fw[1], final_state_bw[1]],1)
        return concat_hidden_states,concat_final_state,(final_state_fw,final_state_bw)

class Decoder(object):
    def __init__(self,size):
        self.size = size

    def decode(self,input_repr,x_mask,q_mask,a_mask):
        seq_mask = tf.concat( [q_mask,x_mask,a_mask],1)
        with tf.variable_scope('decode_layer1'):
            print('-'*5 + "decoding layer 1" + '-'*5)
            m,_,_ = self.decode_LSTM(input_repr,seq_mask,None)
            print("first_layer",m)
        with tf.variable_scope('decode_layer2'):
            print('-'*5 + "decoding layer 2" + '-'*5)
            b,_,_ = self.decode_LSTM(m,seq_mask,None)
            print("Second_layer",b)
        return b

    def decode_LSTM(self,inputs,mask,decoder_state_input):
        cell_fw = tf.contrib.rnn.LSTMCell(num_units=self.size,state_is_tuple=True)
        cell_bw = tf.contrib.rnn.LSTMCell(num_units=self.size,state_is_tuple=True)

        if decoder_state_input is not None:
            state_fw = decoder_state_input[0]
            state_bw = decoder_state_input[1]
        else:
            state_fw = None
            state_bw = None
        seq_len = tf.reduce_sum(tf.cast(mask, 'int32'), axis=1)
        seq_len = tf.reshape(seq_len, [-1,])

        (hidden_state_fw,hidden_state_bw),(final_state_fw,final_state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,inputs,seq_len,state_fw,state_bw,tf.float32)

        concat_hidden_states = tf.concat([hidden_state_fw,hidden_state_bw],2)

        concat_final_state = tf.concat([final_state_fw[1], final_state_bw[1]],1)
        return concat_hidden_states,concat_final_state,(final_state_fw,final_state_bw)

class InferModel(object):
    def __init__(self, *args):
        self.config = args[0]
        self.pretrained_embeddings = args[1]
        self.vocab = args[2]
        self.embedding_size = self.config.embedding_size
        self.state_size = self.config.state_size
        self.encoder = Encoder(self.state_size)
        self.decoder = Decoder(self.state_size)

        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)

        N,V= self.config.batch_size,self.vocab
        self.q = tf.placeholder(tf.int64, [N, None],name='q')
        self.x = tf.placeholder(tf.int64, [N, None],name='x')
        self.q_mask = tf.placeholder(tf.bool,[N, None],name='q_mask')
        self.x_mask = tf.placeholder(tf.bool,[N, None],name='x_mask')
        self.a = tf.placeholder(tf.int32,[N, None],name='a')
        self.a_mask = tf.placeholder(tf.bool,[N, None],name='a_mask')
        self.y = tf.placeholder(tf.int32,[N, self.config.num_classes],name='y')
        self.JX = tf.placeholder(tf.int32,shape=(),name='JX')
        self.JQ = tf.placeholder(tf.int32,shape=(),name='JQ')
        self.JA = tf.placeholder(tf.int32,shape=(),name='JA')

        with tf.variable_scope("infer"):
            question,context,answer = self.setup_embeddings()
            print("question",question)
            print("context",context)
            self.question_repr,self.context_repr,self.answer_repr = self.encode(question,context,answer,self.x_mask,self.q_mask,self.a_mask) #[N,JQ,2d] , [N,JX,2d], [N,JA,2d]

            self.config.entailment_attention = True
            if self.config.entailment_attention:
                with tf.variable_scope("attention_layer"):
                    #attention based on the paper: Reasoning about Entailment with Attention
                    q_len = tf.reshape(tf.reduce_sum(tf.cast(self.q_mask, 'int32'), axis=1),[-1,])
                    x_len = tf.reshape(tf.reduce_sum(tf.cast(self.x_mask, 'int32'), axis=1),[-1,])
                    cell = tf.contrib.rnn.BasicLSTMCell(self.config.state_size,state_is_tuple=True)
                    state_fw,state_bw = tf.split(self.context_repr,2,axis=2)
                    first_cell_fw = AttentionCell(state_size=self.state_size,state_is_tuple=True,
                                                  encoder_input=state_fw,encoder_input_size=self.JX)
                    first_cell_bw = AttentionCell(state_size=self.state_size,state_is_tuple=True,
                                                  encoder_input=state_bw,encoder_input_size=self.JX)
                    (h_fw,h_bw),(f_fw,f_bw) = tf.nn.bidirectional_dynamic_rnn(first_cell_fw,first_cell_bw,
                                                                              question,q_len,
                                                                              dtype=tf.float32)
                    self.decode_repr = tf.concat([h_fw,h_bw],2)
                    print("decode_repr",self.decode_repr)

                    self.preds = extract_axis_1(self.decode_repr,q_len -1)
            else:

                self.input_repr = tf.concat( [self.question_repr,self.context_repr,self.answer_repr],1)
                print("input",self.input_repr) # [N,JX+JQ,2*d]
                self.decode_repr = self.decoder.decode(self.input_repr,self.x_mask,self.q_mask,self.a_mask)
                print("decode_repr",self.decode_repr)
                sequence_length = tf.reduce_sum(tf.cast(self.x_mask, 'int32'), axis=1)
                self.preds = extract_axis_1(self.decode_repr,sequence_length -1)

            print("preds",self.preds)
            print('-'*5 + "SOFTMAX LAYER" + '-'*5)
            with tf.variable_scope('softmax'):
                W = tf.get_variable('W',shape=(2*self.state_size,self.config.num_classes),dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable('b',shape=(1),dtype=tf.float32,initializer=tf.constant_initializer(0))
                self.pred = tf.matmul(self.preds, W) + b

            self.prediction = tf.argmax(self.pred,1)
            self.true_label = tf.argmax(self.y,1)
            correct_prediction = tf.equal(self.prediction,self.true_label)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)
            print("preds",self.pred)
            tf.summary.histogram('logit_label', self.pred)

            self.loss = self.setup_loss(self.pred)
            # TODO gradient clipping
            # TODO batch normalization
            self.max_gradient_norm = 0.2

            opt = tf.train.AdadeltaOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss,global_step=self.global_step)
            self.train_op = opt
            self.summary_op = tf.summary.merge_all()


    def encode(self,question,context,answer,context_mask,question_mask,answer_mask):

        with tf.variable_scope("ques_encode"):
            print('-'*5 + "encoding question" + '-'*5)
            question_repr,_,_ = self.encoder.encode(question,question_mask,encoder_state_input=None)
            print("questionrepr",question_repr)
        with tf.variable_scope("context_encode"):
            print('-'*5 + "encoding context" + '-'*5)
            context_repr,_,_ = self.encoder.encode(context,context_mask,encoder_state_input=None)
            print("context repr",context_repr)
        with tf.variable_scope("answer_encode"):
            print('-'*5 + "encoding answer" + '-'*5)
            answer_repr,_,_ = self.encoder.encode(answer,answer_mask,encoder_state_input=None)
            print("answer repr",answer_repr)

        return question_repr,context_repr,answer_repr

    def setup_embeddings(self):
        with tf.variable_scope("emb"), tf.device("/cpu:0"):
            word_emb_mat = tf.get_variable(dtype=tf.float32 ,initializer=self.pretrained_embeddings,name="word_emb_mat")
            question = tf.nn.embedding_lookup(word_emb_mat,self.q)
            context = tf.nn.embedding_lookup(word_emb_mat,self.x)
            answer = tf.nn.embedding_lookup(word_emb_mat,self.a)
            question = tf.reshape(question,[self.config.batch_size,self.JQ,self.embedding_size])
            context = tf.reshape(context,[self.config.batch_size,self.JX,self.embedding_size])
            answer = tf.reshape(answer,[self.config.batch_size,self.JA,self.embedding_size])
            return question,context,answer

    def setup_loss(self,preds):
        #loss = tf.losses.hinge_loss(logits=preds,labels=self.true_label)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=preds,labels=self.y))
        #TODO Loss mask
        tf.summary.scalar('loss', loss)
        return loss

    def create_feed_dict(self, question_batch,question_len_batch,
                         context_batch, context_len_batch,
                         ans_batch,ans_len_batch,
                         label_batch=None):

        feed_dict = {}
        JQ = np.max(question_len_batch)
        JX = np.max(context_len_batch)
        JA = np.max(ans_len_batch)

        question, question_mask = padding_batch(question_batch, JQ)
        context, context_mask = padding_batch(context_batch, JX)
        answer, answer_mask = padding_batch(ans_batch, JA)

        feed_dict[self.q] = question
        feed_dict[self.q_mask] = question_mask
        feed_dict[self.x] = context
        feed_dict[self.x_mask] = context_mask
        feed_dict[self.a] = answer
        feed_dict[self.a_mask] = answer_mask
        feed_dict[self.JQ] = JQ
        feed_dict[self.JX] = JX
        feed_dict[self.JA] = JA
        if label_batch is not None:
            num_classes = 2 #(0,1)
            temp = np.zeros([len(label_batch),num_classes])
            for i,x in enumerate(label_batch):
                if x == 1:
                    temp[i][0] = 1
                else:
                    temp[i][1] = 1

            feed_dict[self.y] = temp
        return feed_dict

    def train_on_batch(self,sess,*batches):

        q_batch,q_len_batch,c_batch,c_len_batch,a_batch,a_len_batch,infer_label_batch = batches
        feed = self.create_feed_dict(q_batch, q_len_batch, c_batch, c_len_batch,
                                     a_batch, a_len_batch\
                                     ,label_batch=infer_label_batch)

        loss,global_step,summary = sess.run([self.loss,self.global_step,self.summary_op], feed_dict=feed)

        return loss,summary

    def run_epoch(self, sess, train_set, valid_set,train_raw,valid_raw, epoch):
        train_minibatch = minibatches(train_set, self.config.batch_size)
        global_loss = 0
        global_accuracy = 0
        set_num = len(train_set)
        batch_size = self.config.batch_size
        batch_count = int(np.ceil(set_num * 1.0 / batch_size))
        for i, batch in enumerate(train_minibatch):
            loss,summary = self.train_on_batch(sess,*batch)
            self.writer.add_summary(summary, epoch * batch_count + i)
            print("Loss-",loss)
            #logging.info('-' + "EVALUATING ON TRAINING" + '-')
            train_dataset=[train_set,train_raw]
            train_score = self.evaluate_answer(sess,train_dataset)
            #print("training-accuracy",train_score)
            #logging.info('-' + "EVALUATING ON VALIDATION" + '-')
            valid_dataset=[train_set,train_raw]
            score = self.evaluate_answer(sess,valid_dataset)
            #print("validation-accuracy",score)
            global_loss += loss
        return global_loss,summary

    def answer(self,session,test_batch):
        q_batch,q_len_batch,c_batch,c_len_batch,a_batch,a_len_batch,infer_label_answers = test_batch

        feed = self.create_feed_dict(q_batch, q_len_batch, c_batch, c_len_batch,
                                     a_batch, a_len_batch,infer_label_answers)
        output_feed = [self.prediction,self.accuracy] #already argmaxed
        outputs,accuracy = session.run(output_feed,feed)
        return (outputs,infer_label_answers,accuracy)

    def predict_on_batch(self,session,dataset):
        predict_minibatch = minibatches(dataset,self.config.batch_size)
        preds = []
        for i,batch in enumerate(predict_minibatch):
            preds.append(self.answer(session,batch))
        return preds

    def evaluate_answer(self,session,eval_dataset,print_report=False):
        batch_num = int(np.ceil(len(eval_dataset) * 1.0 / self.config.batch_size))
        eval_data = eval_dataset[0]
        eval_raw = eval_dataset[1]
        preds = self.predict_on_batch(session,eval_data)
        accuracy_overall = 0
        for batch in preds:
            pred,true,accuracy = batch
            accuracy_overall +=  accuracy_score(true,pred)
            if print_report:
                print(classification_report(true, pred, target_names=['0','1']))
        accuracy_overall = accuracy
        return accuracy_overall

    def validate(self,session,dataset):
        batch_num = int(np.ceil(len(dataset) * 1.0 / self.config.batch_size))
        valid_minibatch = minibatches(dataset,self.config.batch_size)
        valid_loss = 0
        valid_accuracy = 0
        for i,batch in enumerate(valid_minibatch):
            loss,accuracy,prediction = self.test(session,batch)
            valid_loss += loss
            valid_accuracy += accuracy
        valid_loss = valid_loss/self.config.batch_size
        valid_accuracy = valid_accuracy/self.config.batch_size
        return valid_loss,valid_accuracy

    def test(self,session,test_batch):
        q_batch,q_len_batch,c_batch,c_len_batch,a_batch,a_len_batch,infer_label_answers = test_batch
        input_feed = self.create_feed_dict(q_batch, q_len_batch, c_batch, c_len_batch,
                                           a_batch, a_len_batch,
                                           label_batch=infer_label_answers)

        output_feed = [self.loss,self.prediction,self.accuracy]
        output_loss,output_prediction,accuracy = session.run(output_feed,input_feed)
        return output_loss,accuracy,output_prediction


    def train(self, session, dataset):
        params = tf.trainable_variables()
        if self.config.dataset == 'squad':
            train_set = dataset['training']
            valid_set = dataset['validation']
            train_raw = dataset['training_raw']
            valid_raw = dataset['validation_raw']
        elif self.config.dataset == 'snli':
            train_set = dataset['training']
            valid_set = dataset['validation']
            train_raw = None
            valid_raw = None
        self.writer = tf.summary.FileWriter('./tmp/', graph=tf.get_default_graph())

        for epoch in range(self.config.num_epochs):
            #self.learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, i * FLAGS.batch_size, train_size, FLAGS.decay, staircase=True)
            logging.info('-'*5 + "TRAINING-EPOCH-" + str(epoch)+ '-'*5)
            score,summary = self.run_epoch(session, train_set,valid_set,train_raw,valid_raw,epoch)
            logging.info('-'*5 + "VALIDATION" + '-'*5)
            validation_loss,validation_accuracy = self.validate(session, valid_set)
            print("validation loss",str(validation_loss))
            print("validation accuracy",str(validation_accuracy))
            valid_dataset = [valid_set,valid_raw]
            score = self.evaluate_answer(session, valid_dataset,printreport=True)
            print("Validation score",score)
            #TODO save the model
