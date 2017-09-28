import json
import cmd
from os.path import join
import math
import numpy as np
import tensorflow as tf
import nltk
from os.path import join
from tqdm import tqdm
from data_util import load_glove_embeddings, load_dataset, load_dontknow_dataset
from cell import padding_batch
from infer_model import InferModel
import logging
logging.basicConfig(level=logging.INFO)
from data_util import minibatches
from sklearn.metrics import classification_report, accuracy_score, f1_score
from tensorflow.python.platform import gfile
from tensorflow.contrib.tensorboard.plugins import projector

from ptpython.repl import embed

tf.app.flags.DEFINE_string("data_dir", "./data/squad_features", "data directory")
tf.app.flags.DEFINE_string("data_size", "tiny", "tiny/full")
tf.app.flags.DEFINE_string("dataset", "squad", "squad/dontknow")
tf.app.flags.DEFINE_string("mode", "train", "train/interactive/test")
tf.app.flags.DEFINE_string("logdir", "temp", "log directory")
tf.app.flags.DEFINE_integer("num_classes", 2, "number of classes")

tf.app.flags.DEFINE_float("learning_rate", 0.0015, "Initial learning rate ")
tf.app.flags.DEFINE_integer("num_per_decay", 6, "Epochs before reducing learning rate.")
tf.app.flags.DEFINE_float("decay_factor", 0.01, "Decay factor")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Norm for clipping gradients ")
tf.app.flags.DEFINE_float("keep_prob", 0.8, "keep_prob")
tf.app.flags.DEFINE_float("l2_beta", 0.00001, "L2-beta")
tf.app.flags.DEFINE_integer("num_epochs", 10, "Number of epochs")

tf.app.flags.DEFINE_integer("state_size", 100, "State Size")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Embedding Size")
tf.app.flags.DEFINE_integer("max_question_length", 60, "Maximum Question Length")
tf.app.flags.DEFINE_integer("max_context_length", 300, "Maximum Context Length")
tf.app.flags.DEFINE_integer("batch_size", 10, "Batch size")

tf.app.flags.DEFINE_integer("num_hops", 0, "Number of hops")
tf.app.flags.DEFINE_bool("overlap",False,"Use overlap features or not")
tf.app.flags.DEFINE_bool("ques_aligned_context",False,"Use question aligned embeddings")
tf.app.flags.DEFINE_bool("self_alignment",False,"Use self alignment")
tf.app.flags.DEFINE_bool("use_decoder",True,"Use two decoder layers")

tf.app.flags.DEFINE_integer("gpu_id", 0, "gpu id")
tf.app.flags.DEFINE_float("gpu_fraction", 0.8, " % of GPU memory used.")

FLAGS = tf.app.flags.FLAGS


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab


class Trainer():
    def __init__(self, model, flags):
        self.model = model
        self.config = flags
        self.global_step = model.get_global_step()
        self.loss = model.get_loss()
        self.accuracy = model.get_accuracy()
        self.summary_op = model.get_summary()
        self.train_writer = tf.summary.FileWriter('./{}/{}/train/'.format(self.config.logdir,self.config.dataset), graph=tf.get_default_graph())
        self.valid_writer = tf.summary.FileWriter('./{}/{}/valid/'.format(self.config.logdir,self.config.dataset), graph=tf.get_default_graph())
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = self.model.word_emb_mat.name
        embedding.metadata_path = './temp/embed.csv'
        projector.visualize_embeddings(self.train_writer,config)
        self.saver_embed = tf.train.Saver([self.model.word_emb_mat])

    def run_epoch(self, session, train_set, train_raw,epoch):
        total_batches = int(len(train_set) / self.config.batch_size)
        train_minibatches = minibatches(train_set, self.config.batch_size, self.config.dataset)
        training_loss = 0.0
        training_accuracy = 0.0
        infer_label = []
        prediction_all = []
        for batch in tqdm(train_minibatches, desc="Trainings", total=total_batches):
            if len(batch[0]) != self.config.batch_size:
                continue
            session.run(self.model.inc_step)
            loss, accuracy, summary, global_step, infer_label_batch,prediction = self.train_single_batch(session,*batch)
            _ = [infer_label.append(x) for x in infer_label_batch]
            _= [ prediction_all.append(x) for x in prediction]
            self.train_writer.add_summary(summary, global_step)
            self.saver_embed.save(session, './temp/embedding_test.ckpt', 1)
            training_accuracy += accuracy
            training_loss += loss
        training_loss = training_loss/total_batches
        training_accuracy = training_accuracy/total_batches
        print(classification_report(infer_label, prediction_all, target_names=['can\'t','can']))
        score = f1_score(y_true=infer_label,y_pred=prediction_all)
        print("Loss",training_loss)
        print("F1_score",score)
        return score


    def train_single_batch(self,session,*batch):
        q_batch,q_len_batch,c_batch,c_len_batch,cf_batch,cf_len_batch,a_batch,a_len_batch,infer_label_batch = batch
        input_feed  = self.model.create_feed_dict(q_batch,q_len_batch,c_batch,c_len_batch,
                                                  cf_batch,cf_len_batch,a_batch,a_len_batch,
                                                  label_batch=infer_label_batch)
        output_feed = [self.model.train_op,self.loss,self.global_step,self.accuracy,self.summary_op,self.model.prediction]
        _,loss,global_step,accuracy,summary,prediction = session.run(output_feed,feed_dict=input_feed)
        return loss,accuracy,summary,global_step,infer_label_batch,prediction

    def validate(self,session,validation_set,validation_raw,epoch):
        total_batches = int(len(validation_set)/self.config.batch_size)
        validation_accuracy = 0.0
        validation_loss = 0.0
        infer_label = []
        prediction_all = []
        validate_minibatches = minibatches(validation_set,self.config.batch_size,self.config.dataset)
        for batch in tqdm(validate_minibatches,total=total_batches,desc="Validate"):
            if len(batch[0]) != self.config.batch_size:
                continue
            loss,accuracy,summary,global_step,infer_label_batch,prediction = self.validate_single_batch(session,*batch)
            self.valid_writer.add_summary(summary, global_step)
            validation_accuracy += accuracy
            validation_loss += loss
            _ = [infer_label.append(x) for x in infer_label_batch]
            _= [ prediction_all.append(x) for x in prediction]
        validation_loss = validation_loss/total_batches
        validation_accuracy = validation_accuracy/total_batches
        print(classification_report(infer_label, prediction_all, target_names=['can\'t','can']))
        score = f1_score(y_true=infer_label,y_pred=prediction_all,average='weighted')
        print("Loss",validation_loss)
        print("F1_score",score)
        return score

    def validate_single_batch(self,session,*batch):
        q_batch,q_len_batch,c_batch,c_len_batch,cf_batch,cf_len_batch,a_batch,a_len_batch,infer_label_batch = batch
        input_feed  = self.model.create_feed_dict(q_batch,q_len_batch,c_batch,c_len_batch,
                                                  cf_batch,cf_len_batch,a_batch,a_len_batch,
                                                  label_batch=infer_label_batch)
        output_feed = [self.loss,self.global_step,self.accuracy,self.summary_op,self.model.prediction]
        loss,global_step,accuracy,summary_op,prediction = session.run(output_feed,feed_dict=input_feed)
        #print("prediction",prediction)
        #print("true",infer_label_batch)
        return loss, accuracy, summary_op, global_step, infer_label_batch,prediction



class InteractiveSess(cmd.Cmd):

    def __init__(self, config, session):
        self.context = None
        self.question = None
        self.inference = None
        self.session = session
        self.graph = None
        self.config = config
        super(InteractiveSess, self).__init__()

    def create_feed_dict(self, question_batch, question_len_batch, context_batch, context_len_batch,
                         context_features_batch, context_features_len_batch, ans_batch, ans_len_batch, label_batch=None):

        feed_dict = {}
        q = self.graph.get_tensor_by_name("q:0")
        q_mask = self.graph.get_tensor_by_name("q_mask:0")
        x = self.graph.get_tensor_by_name("x:0")
        x_mask = self.graph.get_tensor_by_name("x_mask:0")
        fx = self.graph.get_tensor_by_name("fx:0")
        fx_mask = self.graph.get_tensor_by_name("fx_mask:0")
        a = self.graph.get_tensor_by_name("a:0")
        a_mask = self.graph.get_tensor_by_name("a_mask:0")
        y = self.graph.get_tensor_by_name("y:0")
        JX = self.graph.get_tensor_by_name("JX:0")
        JQ = self.graph.get_tensor_by_name("JQ:0")
        JFX = self.graph.get_tensor_by_name("JFX:0")
        JA = self.graph.get_tensor_by_name("JA:0")
        keep_prob = self.graph.get_tensor_by_name("keep_prob:0")

        jq = np.max(question_len_batch)
        jx = np.max(context_len_batch)
        ja = np.max(ans_len_batch)
        jfx = np.max(context_features_len_batch)
        question, question_mask = padding_batch(question_batch, jq)
        context, context_mask = padding_batch(context_batch, jx)
        answer, answer_mask = padding_batch(ans_batch, ja)
        context_features, context_features_mask = padding_batch(context_features_batch, 3*jx)

        feed_dict[q] =question
        feed_dict[q_mask] = question_mask
        feed_dict[x] = context
        feed_dict[x_mask] = context_mask
        feed_dict[fx] = context_features
        feed_dict[fx_mask] = context_features_mask
        feed_dict[a] = answer
        feed_dict[a_mask] = answer_mask
        feed_dict[JQ] = jq
        feed_dict[JX] = jx
        feed_dict[JA] = ja
        feed_dict[JFX] = jfx
        feed_dict[keep_prob] = self.config.keep_prob
        if label_batch is not None:
            feed_dict[y] = label_batch
        return feed_dict

    def predict_single(self, session, *batch):
        q_batch,q_len_batch,c_batch,c_len_batch,cf_batch,cf_len_batch,a_batch,a_len_batch = batch[0]
        input_feed  = self.create_feed_dict([q_batch],[q_len_batch],[c_batch],[c_len_batch],
                                                  [cf_batch],[cf_len_batch],[a_batch],[a_len_batch]
                                                  ,label_batch=None)
        prob_softmax = []
        probs = []
        data_size = None
        for _ in range(20):
            output_feed = [self.graph.get_tensor_by_name("infer/pred_softmax:0"),
                            self.graph.get_tensor_by_name("infer/prediction:0"),
                            self.graph.get_tensor_by_name("dataset_size:0")]
            a, b, c  = session.run(output_feed, feed_dict=input_feed)
            if data_size is None:
                data_size = c
            prob_softmax.append(a[0][int(b[0])])
            probs.append(b[0])

        predictive_mean = np.mean(probs, axis=0)
        predictive_variance = np.var(probs, axis=0)
        l = 2
        tau = l**2 * (1 - self.config.learning_rate) / (2 * data_size * self.config.decay_factor)
        predictive_variance += tau**-1
        print("predicting with confidence - {}".format(predictive_variance))
        return b[0]



    def sent_token_ids(self, vocab, sentence):
        '''
        return vocab mapping for each word, and 1 for UNK
        '''
        return [vocab.get(w, 1) for w in sentence]


    def process_input(self):
        single_set, single_raw = [], []
        vocab,_ = initialize_vocab(join('./data/squad_features/vocab.dat'))


        wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()
        q = [] # q,q_len,c,c_len,a,a_len
        c = []
        f = []
        a = []
        tokens1 = nltk.word_tokenize(self.context)
        tokens2 = nltk.word_tokenize(self.question)

        lemmas1 = [wordnet_lemmatizer.lemmatize(x) for x in tokens1]
        lemmas2 = [wordnet_lemmatizer.lemmatize(x) for x in tokens2]
        lower1 = [x.lower() for x in tokens1]
        lower2 = [x.lower() for x in tokens2]

        _ = [c.append(x) for x in self.sent_token_ids(vocab, self.context)]
        _ = [q.append(x) for x in self.sent_token_ids(vocab, self.question)]

        _ = [f.append(1) if x in tokens2 else f.append(0) for x in tokens1]
        _ = [f.append(1) if x in lemmas2 else f.append(0) for x in lemmas1]
        _ = [f.append(1) if x in lower2 else f.append(0) for x in lower1]

        single_set = [q, len(q), c, len(c), f, len(f), a, len(a)]
        return single_set, single_raw

    def run_inference(self):
        self.graph = tf.get_default_graph()
        test_set, test_raw = self.process_input()
        self.inference = self.predict_single(self.session, test_set, test_raw)

    def do_context(self, text):
        """
        context
        """
        if text:
            print("Context")
            print(text)
            self.context = text

    def do_question(self, text):
        """
        question
        """
        if text:
            print("question")
            print(text)
            self.question = text

    def do_infer(self, text):
        if self.context is not None and self.question is not None:
            self.run_inference()
            if self.inference == 1:
                print("I can answer that")
            else:
                print("Can't answer that")
        else:
            print("Enter both question and context")

    def do_EOF(self, line):
        return True

    def postloop(self):
        print


class Tester():
    def __init__(self):
        pass



def train():
    if FLAGS.dataset == 'dontknow':
        dataset = load_dontknow_dataset(FLAGS.data_size,FLAGS.max_question_length,FLAGS.max_context_length)
        embed_path = join(FLAGS.data_dir,"glove.trimmed.100.npz")
        vocab_path = join(FLAGS.data_dir, "vocab.dat")
        vocab, rev_vocab = initialize_vocab(vocab_path)
        embeddings = load_glove_embeddings(embed_path)

    elif FLAGS.dataset == 'squad':
        dataset = load_dataset(FLAGS.data_dir,FLAGS.data_size,FLAGS.max_question_length,FLAGS.max_context_length)
        embed_path = join(FLAGS.data_dir, "glove.trimmed.100.npz")
        vocab_path = join(FLAGS.data_dir, "vocab.dat")
        vocab, rev_vocab = initialize_vocab(vocab_path)
        embeddings = load_glove_embeddings(embed_path)
    else:
        print("enter either squad or dontknow for dataset flag")
        return
    FLAGS.dataset_size = len(dataset['training'])
    model = InferModel(FLAGS, embeddings, vocab)

    trainer = Trainer(model,FLAGS)
    saver = tf.train.Saver()
    validation_scores = []
    with tf.device("/gpu:{}".format(FLAGS.gpu_id)):
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_fraction
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            logging.info("Created a new model")
            sess.run(tf.global_variables_initializer())
            logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
            train_set = dataset['training']
            valid_set = dataset['validation']
            train_raw = dataset['training_raw']
            valid_raw = dataset['validation_raw']
            with open('log.txt', 'w') as e:
                for line in train_raw:
                    question = ' '.join(line[0])
                    context = ' '.join(line[1])
                    answer =  ' '.join(line[2])
                    e.write("-"*5)
                e.write(" -- VALID- - -")
                for line in valid_raw:
                    question = ' '.join(line[0])
                    context = ' '.join(line[1])
                    answer = ' '.join(line[2])
                    e.write(context + "- - - " + question + '\n')
                    e.write("-"*5)

            for epoch in range(FLAGS.num_epochs):
                logging.info('-'*5 + "TRAINING-EPOCH-" + str(epoch)+ '-'*5)
                score = trainer.run_epoch(sess, train_set, train_raw, epoch)
                logging.info('-'*5 + "-VALIDATE-" + str(epoch)+ '-'*5)
                val_score = trainer.validate(sess, valid_set, valid_raw, epoch)
                validation_scores.append(val_score)
                #TODO early stopping
                print("Saving Model")
                save_path = saver.save(sess,"./{}/{}/model.ckpt".format(FLAGS.logdir,FLAGS.dataset))

def test():
    pass

def interactive():
    '''
    Loading dataset to calculate data_size
    '''

    with tf.device("/gpu:{}".format(FLAGS.gpu_id)):
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_fraction
        with tf.Session(config=config) as sess:
            saver = tf.train.import_meta_graph("./{}/{}/model.ckpt.meta".format(FLAGS.logdir,FLAGS.dataset))
            saver.restore(sess, tf.train.latest_checkpoint("./{}/{}".format(FLAGS.logdir,FLAGS.dataset)))
            interactive_sesh = InteractiveSess(FLAGS,sess)
            interactive_sesh.cmdloop()


def main(_):
    if FLAGS.mode == 'train':
        train()
    elif FLAGS.mode == 'test':
        test()
    elif FLAGS.mode == 'interactive':
        interactive()
if __name__ == "__main__":
    tf.app.run()
