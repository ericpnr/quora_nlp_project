from __future__ import print_function
from __future__ import division
from tqdm import tqdm as ProgressBar
from collections import defaultdict, Counter
import numpy as np
import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import scipy.sparse
import datetime as dt
import random
import os

class Neural_CNN(object):
    """
    PURPOSE: To build, and train a deep neural network for the purposes of
             predicting the probability that a given product (for example bananas)
             is included in a given purchased basket of good, given all other
             items already purchased

    ARGS:
    setup                       (dict) containing all some or none of the following
        learning_rate               (float) ml gradient descent learning rate
        max_sequence_length         (int) maximum length of token sequence
        embedding_dimension         (int) truncated length of embedding vectors
        batch_normalization         (bool) indicator for use of batch normalization
        n_filters_per_kernel_by_lyr (list(int)) number of filter per kernel size by layer
        kernel_sizes_by_lyr         (list(list(int)) list of kernel sizes for each layer
        n_hidden                    (list) list of nuerons per hidden layer
        hidden_activation           (str) hidden layer activation function
        dropout_rate                (float) probability of dropout
        n_tokens                    (int) total number of rows in embed matrix
        n_outputs                   (int) number of unique labels
        root_log_dir                (str) directory where tf logs will be kept
        checkpoint_dir              (str) directory where checkpoints are kept
    """

    def __init__(self,setup):
        #Extracting Variables
        self.LRN_RATE = setup.get('learning_rate',0.01)
        self.MAX_SEQ_LEN = setup.get('max_sequence_length',50)
        self.EMBD_DIM = setup.get('embedding_dimension',100)
        self.BATCH_NORM  = setup.get('batch_normalization',True)
        self.n_filters_per_kernel_by_lyr = setup.get('n_filters_per_kernel_by_lyr',[10])
        self.kernel_sizes_by_lyr = setup.get('kernel_sizes_by_lyr',[[1,2,3]])
        self.n_hidden = setup.get('n_hidden',[100])
        self.padding = setup.get('padding','SAME')
        self.hidden_activation = setup.get('hidden_activation','elu')
        self.DROP_RATE = setup.get('dropout_rate',0.5)
        self.N_TOKN = setup.get('n_tokens',100)
        self.N_OUTPUTS = setup.get('n_outputs',2)
        self.root_log_dir = setup.get('root_log_dir',''.join([os.getcwd(),'/neural_cnn_logs']))
        self.check_pt_dir = setup.get('checkpoint_dir',''.join([os.getcwd(),'/neural_cnn_ckpts']))
        self.summary_dir = setup.get('summary_dir',''.join([os.getcwd(),'/neural_cnn_run_summary']))
        self.graph = None


    def _placeholders_(self):
        """
        PURPOSE: Construct graph placeholder variables.

        RETURNS:
        sequences_     (tf.tensor) order inputs
        W_embed_    (tf.tensor) product embedding matrix
        Y_          (tf.tensor) class labels
        training_   (tf.tensor) training indicator
        """
        sequences_ = tf.placeholder(tf.int32,
                                    shape=[None,self.MAX_SEQ_LEN],
                                    name='sequences')
        W_embed_ = tf.placeholder(tf.float32,
                                  shape=[self.N_TOKN+1,self.EMBD_DIM],
                                  name='W_embed')
        Y_ = tf.placeholder(tf.int64,shape=[None],name='Y')
        training_ = tf.placeholder_with_default(False,shape=(),name='training')
        return sequences_,W_embed_,Y_,training_


    def _embedding_lookup_layer_(self,W_embed_,sequences_,reduce='sum'):
        """
        PURPOSE: Constructing the Embedding Look up Layer

        ARGS:
        W_embed_    (tf.tensor) product embedding matrix
        sequences_     (tf.tensor) product numbers of included product in each sequences
        reduce      (str) reduce method 'mean' or 'sum'

        RETURNS:
        embedding_layer_         (tf.tensor) vector representation of an order.
        """
        embedding_lkup_ = tf.nn.embedding_lookup(W_embed_,sequences_,name='Embeddings')
        return embedding_lkup_


    def __activation_lookup__(self,layer,layer_name):
        """
        PURPOSE: Applying an activation function to a layer

        ARGS:
        layer:      (tf.tensor) layer activation function will be applied
        layer_name: (str) name of layer

        RETURNS:
        activation_ (tf.tensor) layer with activation function applied
        """
        if self.hidden_activation == 'elu':
            activation_ = tf.nn.elu(layer,name=layer_name)
        if self.hidden_activation == 'relu':
            activation_ = tf.nn.relu(layer,name=layer_name)
        if self.hidden_activation == 'leaky_relu':
            activation_ = tf.nn.leaky_relu(layer,name=layer_name)
        return activation_


    def _hidden_layers_(self,embedding_layer_,training_):
        """
        PURPOSE: Constructing the sequence of hidden layers

        ARGS:
        embedding_layer_         (tf.tensor) vector representation of an order
        training_                (tf.tensor) indicator for training or testing task

        RETURNS:
        fc_                       (tf.tensor) output of last fully connected layer
        """
        he_init_ = tf.initializers.he_uniform()

        prev_layer_ = embedding_layer_
        for i,n_filters in enumerate(self.n_filters_per_kernel_by_lyr):
            with tf.variable_scope(("ConvLayer_%d"%i)):
                conv_list = []
                for kernel_size in self.kernel_sizes_by_lyr[i]:
                    with tf.variable_scope(("KernelSize_%d"%kernel_size)):
                        conv_ = tf.layers.conv1d(prev_layer_,n_filters,
                                              kernel_size,
                                              activation=self.hidden_activation)
                        conv_ = tf.reduce_max(conv_,axis=1)
                        if self.DROP_RATE > 0:
                            conv_ = tf.layers.dropout(conv_,rate=self.DROP_RATE,training=training_)
                    conv_list.append(conv_)
                prev_layer_ = tf.concat(conv_list,axis=1)

        fc_ = prev_layer_
        for i,hidden_size in enumerate(self.n_hidden):
            with tf.variable_scope(("FCLayer_%d"%i)):
                if self.BATCH_NORM:
                    fc_=tf.layers.dense(fc_,hidden_size,
                                        kernel_initializer=he_init_,
                                        name=("Hidden_%d_b4_bn"%i))
                    fc_=tf.layers.batch_normalization(fc_,training=training_,
                                                      name=("Hidden_%d_bn"%i))
                    fc_=self.__activation_lookup__(fc_,("Hidden_%d_act"%i))
                else:
                    fc_ = tf.layers.dense(fc_,hidden_size,
                                          activation=self.hidden_activation,
                                          kernel_initializer=he_init_,
                                          name=("Hidden_%d"%i))
                if self.DROP_RATE > 0:
                    fc_ = tf.layers.dropout(fc_,rate=self.DROP_RATE,
                                            training=training_)
        return fc_

    def _output_layer_(self,h_):
        """
        PURPOSE: Constructing the logits output layer

        ARGS:
        h_      (tf.tensor) output of last hidden layer

        RETURNS:
        logits_ (tf.tensor) logits output layer
        """
        logits_ = tf.layers.dense(h_,self.N_OUTPUTS,name='Logits_lyr')
        return logits_


    def _loss_function_(self,logits_,Y_):
        """
        PURPOSE:Constructing the cross entropy loss function

        ARGS:
        logits_     (tf.tensor) logits output layer
        Y_          (tf.tensor) order class label

        RETURN:
        xentropy    (tf.tensor) raw cross entropy values
        loss_       (tf.tensor) mean cross cross entropy
        """
        xentropy_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y_,
                                                                   logits=logits_,name="Xentropy")
        loss_ = tf.reduce_mean(xentropy_,name="Loss")

        return xentropy_, loss_


    def _optimizer_(self,loss_):
        """
        PURPOSE: Constructing the optimizer and training method

        ARGS:
        loss_       (tf.tensor) mean cross entropy values

        RETURNS:
        optimizer_      (tf.tensor) optimizer
        training_op_    (tf.tensor) training method
        """
        optimizer_ = tf.train.AdagradOptimizer(learning_rate=self.LRN_RATE)
        training_op_ = optimizer_.minimize(loss_)
        return optimizer_,training_op_


    def _evaluation_(self,logits_,Y_):
        """
        PURPOSE: Constructing the Evaluation Piece

        ARGS:
        logits_     (tf.tensor) output layer
        Y_          (tf.tensor) order class labels

        RETURNS:
        correct_    (tf.tensor) number of correct classifications
        accuracy_    (tf.tensor) accuracy on entire dataset
        """
        correct_ = tf.nn.in_top_k(logits_, Y_, 1)
        accuracy_ = tf.reduce_mean(tf.cast(correct_,tf.float32))
        return correct_,accuracy_


    def _initializer_(self):
        """
        PURPOSE: Initializing all Variables

        RETURNS:
        init_       (tf.tensor) initializer for all graph variables
        saver_      (tf.tensor) saver method
        """
        init_ = tf.global_variables_initializer()
        saver_ = tf.train.Saver()
        return init_, saver_


    def build_graph(self):
        """
        PURPOSE: Building the lazily executed tensor flow graph
        """
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope("PlaceHolders"):
                sequences_,W_embed_,Y_,training_ = self._placeholders_()
            for var in (sequences_,W_embed_,Y_,training_):
                tf.add_to_collection("Input_var",var)

            with tf.name_scope("EmbeddingLyr"):
                embedding_layer_ = self._embedding_lookup_layer_(W_embed_,sequences_)

            with tf.name_scope("HiddenLyrs"):
                h_ = self._hidden_layers_(embedding_layer_,training_)

            with tf.name_scope("OutputLyr"):
                logits_ = self._output_layer_(h_)
            tf.add_to_collection("LogitsLyr", logits_)

            with tf.name_scope("Loss"):
                xentropy_,loss_ = self._loss_function_(logits_,Y_)
            for op in (xentropy_,loss_):
                tf.add_to_collection("Loss_ops",op)

            with tf.name_scope("Optimizer"):
                optimizer_,training_op_ = self._optimizer_(loss_)
            for op in (optimizer_,training_op_):
                tf.add_to_collection("Optimizer_ops",op)

            with tf.name_scope("Evaluation"):
                correct_,accuracy_ = self._evaluation_(logits_,Y_)
            for op in (correct_,accuracy_):
                tf.add_to_collection("Eval_ops",op)

            with tf.name_scope("VariableInit"):
                init_,saver_ = self._initializer_()
            for op in (init_,saver_):
                tf.add_to_collection("Init_Save_ops",op)
        #Generating file names
        self._file_names_()


    def _file_names_(self):
        """
        PURPOSE: Constructing appropriate file names for logging, and checkpointing
        """
        now_time = dt.datetime.now().second
        file_ext = ''.join(['CNN','_',str(self.EMBD_DIM)
                                 ,'_',str(len(self.n_hidden))
                                 ,'x',str(self.n_hidden[0])
                                 ,'-t',str(now_time)])
        self.log_dir = ''.join([self.root_log_dir,'/run-',file_ext,'/'])
        self.temp_ckpt = ''.join([self.check_pt_dir,'/run-',file_ext,'/','temp.ckpt'])
        self.final_ckpt = ''.join([self.check_pt_dir,'/run-',file_ext,'/','final.ckpt'])
        self.summary_file = ''.join([self.summary_dir,'/run-',file_ext,'.txt'])
        self.most_recent_summary_file = ''.join([self.summary_dir,'/most_recent_summary.txt'])



    def write_graph(self):
        """
        PURPOSE: Writing the graph to a log file so the it can be reused and or
                 viewed in tensorboard
        """
        if self.graph is not None:
            with tf.Session(graph=self.graph) as sess:
                init_,_ = tf.get_collection('Init_Save_ops')
                init_.run()
                file_writer = tf.summary.FileWriter(self.log_dir,self.graph)
        else:
            print('No Current Graph')


    def _partition_(self,list_in,n):
        """
        PURPOSE: Generate indexs for minibatchs used in minibatch gradient descent
        """
        random.shuffle(list_in)
        return [list_in[i::n] for i in range(n)]

    def train_graph(self,train_dict):
        """
        PURPOSE: Train a deep neural net classifier for baskets of products

        ARGS:
        train_dict          (dict) dictionary with ALL the following key values
            embeddings       (list(list)) trained product embedding layers
            sequences_train        (list(list)) training order product numbers
            labels_train        (list) training order class labels
            sequences_valid         (list(list)) test order product numbers
            labels_valid         (list) test order class
            batch_size          (int) number of training example per mini batch
            n_stop              (int) early stopping criteria
        """
        embeddings = train_dict.get('embeddings',None)
        sequences_train = train_dict.get('sequences_train',None)
        labels_train = train_dict.get('labels_train',None)
        sequences_valid = train_dict.get('sequences_valid',None)
        labels_valid = train_dict.get('labels_valid',None)
        batch_size = train_dict.get('batch_size',100)
        n_stop = train_dict.get('n_stop',5)

        n_train_ex = len(sequences_train)
        n_batches = n_train_ex // batch_size
        done,epoch,acc_reg = 0,0,[0,1]

        with self.graph.as_default():
            correct_,accuracy_ = tf.get_collection('Eval_ops')
            acc_summary = tf.summary.scalar('Accuracy',accuracy_)
            file_writer = tf.summary.FileWriter(self.log_dir,self.graph)

        with tf.Session(graph=self.graph) as sess:
            init_,saver_ = tf.get_collection('Init_Save_ops')
            correct_,accuracy_ = tf.get_collection('Eval_ops')
            optimizer_,training_op_ = tf.get_collection("Optimizer_ops")
            sequences_,W_embed_,Y_,training_ = tf.get_collection("Input_var")

            sess.run(init_)
            while done != 1:
                epoch += 1
                batches = self._partition_(list(range(n_train_ex)),n_batches)
                #Mini-Batch Training step
                for iteration in ProgressBar(range(n_batches),'Epoch {} Iterations'.format(epoch)):
                    sequences_batch = [sequences_train[indx] for indx in batches[iteration]]
                    labels_batch = [labels_train[indx] for indx in batches[iteration]]
                    sess.run([training_op_], feed_dict={training_:True,
                                                        W_embed_:embeddings,
                                                        sequences_:sequences_batch,
                                                        Y_:labels_batch})
                    #Intermediate Summary Writing
                    if iteration % 10 == 0:
                        summary_str = acc_summary.eval(feed_dict={training_:False,
                                                                  W_embed_:embeddings,
                                                                  sequences_:sequences_valid,
                                                                  Y_:labels_valid})
                        step = epoch*n_batches + iteration
                        file_writer.add_summary(summary_str,step)
                #Early Stopping Regularization
                if epoch % 1 == 0:
                    # Evaluating the Accuracy of Current Model
                    acc_ckpt = accuracy_.eval(feed_dict={training_:False,
                                                         W_embed_:embeddings,
                                                         sequences_:sequences_valid,
                                                         Y_:labels_valid})
                    if acc_ckpt > acc_reg[0]:
                        # Saving the new "best" model
                        save_path = saver_.save(sess,self.temp_ckpt)
                        acc_reg = [acc_ckpt,1]
                    elif acc_ckpt <= acc_reg[0] and acc_reg[1] < n_stop:
                        acc_reg[1] += 1
                    elif acc_ckpt <= acc_reg[0] and acc_reg[1] >= n_stop:
                        #Restoring previous "best" model
                        saver_.restore(sess,self.temp_ckpt)
                        done = 1
                #Calculating Accuracy for Output
                acc_train = accuracy_.eval(feed_dict={training_:False,
                                                      W_embed_:embeddings,
                                                      sequences_:sequences_train,
                                                      Y_:labels_train})
                acc_test = accuracy_.eval(feed_dict={training_:False,
                                                     W_embed_:embeddings,
                                                     sequences_:sequences_valid,
                                                     Y_:labels_valid})
                print('Register:{} Epoch:{:2d} Train Accuracy:{:6.4f} Validation Accuracy: {:6.4f}'.format(acc_reg, epoch, acc_train, acc_test))
                #Final Model Save
                save_path = saver_.save(sess,self.final_ckpt)


    def predict_and_report(self,sequences,labels,W_embed,report=True,file=False):
        """
        PURPOSE: Prediction using best model on provided examples and generating
                 report if indicated and labels are provided.

        ARGS:
        sequences      (list(list)) order of product numbers
        labels      (list) order class labels
        W_embed     (list(list)) trained word embedding Matrix
        report      (bool) indicator for whether a report is generated
        """
        from sklearn.metrics import confusion_matrix,classification_report
        import json

        with tf.Session(graph=self.graph) as sess:
            _,saver_ = tf.get_collection('Init_Save_ops')
            saver_.restore(sess,self.final_ckpt)
            logits_ = self.graph.get_tensor_by_name('OutputLyr/Logits_lyr/BiasAdd:0')
            sequences_,W_embed_,Y_,training_ = tf.get_collection("Input_var")
            self.logits_prediction = logits_.eval(feed_dict={W_embed_:W_embed,
                                                        sequences_:sequences,
                                                        training_:False})
            self.class_prediction = np.argmax(self.logits_prediction,axis=1)

            confusion_mat = confusion_matrix(labels,self.class_prediction)
            true_neg = confusion_mat[0,0]/(confusion_mat[0,0]+confusion_mat[1,0])
            false_neg = confusion_mat[0,1]/(confusion_mat[0,1]+confusion_mat[1,1])
            ratio = true_neg/false_neg

            if report:
                print('-----------{}-----------'.format('Confusion Matrix'))
                print(confusion_mat,'\n')
                print('-----------{}-----------'.format('Classification Report'))
                print(classification_report(labels,self.class_prediction))
                print('True Negative:', true_neg)
                print('False Negative:', false_neg)
                print('Upper Constraint:', ratio)
            if file:
                summary_dict = self.__dict__.copy()
                class_report_dict = classification_report(labels,
                                                          self.class_prediction,
                                                          output_dict=True)
                summary_dict.update(class_report_dict)
                summary_dict.update({'true_negative':true_neg,
                              'false_negative':false_neg,
                              'upper_constraint':ratio})
                summary_dict.pop('graph',None)
                summary_dict.pop('logits_prediction',None)
                summary_dict.pop('class_prediction',None)
                with open(self.summary_file,'w') as file:
                    json.dump(summary_dict,file,indent=2)
                with open(self.most_recent_summary_file,'w') as file:
                    json.dump(summary_dict,file,indent=2)
