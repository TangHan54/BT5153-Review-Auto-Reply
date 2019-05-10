import time
import pickle 
import re
import numpy as np
import os
import pandas as pd
import pickle
from imblearn.over_sampling import RandomOverSampler
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
tf.__version__

class seq2seqAttention():

    def __init__(self):
        f = open('data/review_matrix.pickle','rb')
        review_matrix = pickle.load(f)
        f = open('data/reply_matrix.pickle','rb')
        reply_matrix = pickle.load(f)
        
        train_valid_split = int(len(review_matrix)*0.1)

        # Split the reviews and replies into training and validating data
        self.train_reviews = review_matrix[train_valid_split:]
        self.train_replies = reply_matrix[train_valid_split:]

        self.valid_reviews = review_matrix[:train_valid_split]
        self.valid_replies = reply_matrix[:train_valid_split]

        f = open('data/vocab_reduced.pickle','rb')
        self.vocab_reduced = pickle.load(f)

        # Set the Hyperparameters
        self.epochs = 10000
        self.batch_size = 64
        self.rnn_size = 32
        self.num_layers = 2
        self.encoding_embedding_size = 32
        self.decoding_embedding_size = 32
        self.learning_rate = 0.005
        self.learning_rate_decay = 0.9
        self.min_learning_rate = 0.0001
        self.keep_probability = 0.75
        self.maxlen = 100

    # clean text
    def clean_text(self, text):
        '''Clean text by removing unnecessary characters and altering the format of words.'''

        text = text.lower()
        text = re.sub(r"i'm", "i am", text)
        text = re.sub(r"he's", "he is", text)
        text = re.sub(r"she's", "she is", text)
        text = re.sub(r"it's", "it is", text)
        text = re.sub(r"that's", "that is", text)
        text = re.sub(r"what's", "that is", text)
        text = re.sub(r"where's", "where is", text)
        text = re.sub(r"how's", "how is", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"n'", "ng", text)
        text = re.sub(r"'bout", "about", text)
        text = re.sub(r"'til", "until", text)
        text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
        text = re.sub(r"\r\n", " ", text)
        text = re.sub(r"jennyhealintcom", "<EMAIL>", text)
        
        return text


    def model_inputs(self):
        '''Initialize model inputs'''
        input_data = tf.placeholder(tf.int32, [None, None], name='input')
        targets = tf.placeholder(tf.int32, [None, None], name='targets')
        lr = tf.placeholder(tf.float32, name='learning_rate')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        return input_data, targets, lr, keep_prob

    def process_encoding_input(self,target_data, vocab_to_int, batch_size):
        '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''
        ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
        dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

        return dec_input

    def encoding_layer(self, rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
        '''Create the encoding layer'''
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        enc_cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)
        _, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = enc_cell,
                                                    cell_bw = enc_cell,
                                                    sequence_length = sequence_length,
                                                    inputs = rnn_inputs, 
                                                    dtype=tf.float32)
        return enc_state

    def decoding_layer_train(self, encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope,
                            output_fn, keep_prob, batch_size):
        '''Decode the training data'''
        
        attention_states = tf.zeros([batch_size, 1, dec_cell.output_size])
        
        att_keys, att_vals, att_score_fn, att_construct_fn = \
                tf.contrib.seq2seq.prepare_attention(attention_states,
                                                    attention_option="bahdanau",
                                                    num_units=dec_cell.output_size)
        
        train_decoder_fn = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                        att_keys,
                                                                        att_vals,
                                                                        att_score_fn,
                                                                        att_construct_fn,
                                                                        name = "attn_dec_train")
        train_pred, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell, 
                                                                train_decoder_fn, 
                                                                dec_embed_input, 
                                                                sequence_length, 
                                                                scope=decoding_scope)
        train_pred_drop = tf.nn.dropout(train_pred, keep_prob)
        return output_fn(train_pred_drop)

    def decoding_layer_infer(self, encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id,
                            maximum_length, vocab_size, decoding_scope, output_fn, keep_prob, batch_size):
        '''Decode the prediction data'''
        
        attention_states = tf.zeros([batch_size, 1, dec_cell.output_size])
        
        att_keys, att_vals, att_score_fn, att_construct_fn = \
                tf.contrib.seq2seq.prepare_attention(attention_states,
                                                    attention_option="bahdanau",
                                                    num_units=dec_cell.output_size)
        
        infer_decoder_fn = tf.contrib.seq2seq.attention_decoder_fn_inference(output_fn, 
                                                                            encoder_state[0], 
                                                                            att_keys, 
                                                                            att_vals, 
                                                                            att_score_fn, 
                                                                            att_construct_fn, 
                                                                            dec_embeddings,
                                                                            start_of_sequence_id, 
                                                                            end_of_sequence_id, 
                                                                            maximum_length, 
                                                                            vocab_size, 
                                                                            name = "attn_dec_inf")
        infer_logits, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell, 
                                                                    infer_decoder_fn, 
                                                                    scope=decoding_scope)
        
        return infer_logits

    def decoding_layer(self, dec_embed_input, dec_embeddings, encoder_state, vocab_size, sequence_length, rnn_size,
                    num_layers, vocab_to_int, keep_prob, batch_size):
        '''Create the decoding cell and input the parameters for the training and inference decoding layers'''
        
        with tf.variable_scope("decoding") as decoding_scope:
            lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
            drop = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
            dec_cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)
            
            weights = tf.truncated_normal_initializer(stddev=0.1)
            biases = tf.zeros_initializer()
            output_fn = lambda x: tf.contrib.layers.fully_connected(x, 
                                                                    vocab_size, 
                                                                    None, 
                                                                    scope=decoding_scope,
                                                                    weights_initializer = weights,
                                                                    biases_initializer = biases)

            train_logits = self.decoding_layer_train(encoder_state, 
                                                dec_cell, 
                                                dec_embed_input, 
                                                sequence_length, 
                                                decoding_scope, 
                                                output_fn, 
                                                keep_prob, 
                                                batch_size)
            decoding_scope.reuse_variables()
            infer_logits = self.decoding_layer_infer(encoder_state, 
                                                dec_cell, 
                                                dec_embeddings, 
                                                vocab_to_int['<GO>'],
                                                vocab_to_int['<EOS>'], 
                                                sequence_length - 1, 
                                                vocab_size,
                                                decoding_scope, 
                                                output_fn, keep_prob, 
                                                batch_size)

        return train_logits, infer_logits

    def seq2seq_model(self, input_data, target_data, keep_prob, batch_size, sequence_length, answers_vocab_size, 
                    questions_vocab_size, enc_embedding_size, dec_embedding_size, rnn_size, num_layers, 
                    questions_vocab_to_int):
        
        '''Use the previous functions to create the training and inference logits'''
        
        enc_embed_input = tf.contrib.layers.embed_sequence(input_data, 
                                                        answers_vocab_size+1, 
                                                        enc_embedding_size,
                                                        initializer = tf.random_uniform_initializer(0,1))
        enc_state = self.encoding_layer(enc_embed_input, rnn_size, num_layers, keep_prob, sequence_length)

        dec_input = self.process_encoding_input(target_data, questions_vocab_to_int, batch_size)
        dec_embeddings = tf.Variable(tf.random_uniform([questions_vocab_size+1, dec_embedding_size], 0, 1))
        dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
        
        train_logits, infer_logits = self.decoding_layer(dec_embed_input, 
                                                    dec_embeddings, 
                                                    enc_state, 
                                                    questions_vocab_size, 
                                                    sequence_length, 
                                                    rnn_size, 
                                                    num_layers, 
                                                    questions_vocab_to_int, 
                                                    keep_prob, 
                                                    batch_size)
        return train_logits, infer_logits

    def pad_sentence_batch(self, sentence_batch, vocab_to_int):
        """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
        max_sentence = max([len(sentence) for sentence in sentence_batch])
        return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]

    def batch_data(self, questions, answers, batch_size):
        """Batch questions and answers together"""
        for batch_i in range(0, len(questions)//batch_size):
            start_i = batch_i * batch_size
            questions_batch = questions[start_i:start_i + batch_size]
            answers_batch = answers[start_i:start_i + batch_size]
            pad_questions_batch = np.array(self.pad_sentence_batch(questions_batch, self.vocab_reduced))
            pad_answers_batch = np.array(self.pad_sentence_batch(answers_batch, self.vocab_reduced))
            yield pad_questions_batch, pad_answers_batch
    
    def train(self):
        # Reset the graph to ensure that it is ready for training
        tf.reset_default_graph()
        # Start the session
        sess = tf.InteractiveSession()

        # Load the model inputs    
        input_data, targets, lr, keep_prob = self.model_inputs()
        # Sequence length will be the max line length for each batch
        sequence_length = tf.placeholder_with_default(self.maxlen, None, name='sequence_length')
        # Find the shape of the input data for sequence_loss
        input_shape = tf.shape(input_data)

        # Create the training and inference logits
        train_logits, inference_logits = self.seq2seq_model(
            tf.reverse(input_data, [-1]), targets, keep_prob, self.batch_size, sequence_length, len(self.vocab_reduced), 
            len(self.vocab_reduced), self.encoding_embedding_size, self.decoding_embedding_size, self.rnn_size, self.num_layers, 
            self.vocab_reduced)

        # Create a tensor for the inference logits, needed if loading a checkpoint version of the model
        tf.identity(inference_logits, 'logits')

        with tf.name_scope("optimization"):
            # Loss function
            cost = tf.contrib.seq2seq.sequence_loss(
                train_logits,
                targets,
                tf.ones([input_shape[0], sequence_length]))

        # Optimizer
        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)

        display_step = 100 # Check training loss after every 100 batches
        stop_early = 0 
        stop = 5 # If the validation loss does decrease in 5 consecutive checks, stop training
        total_train_loss = 0 # Record the training loss for each display step
        total_epoch_train_loss = []
        total_epoch_valid_loss = [] # Record the validation loss for saving improvements in the model
        checkpoint = "model/best_model.ckpt" 
        saver = tf.train.Saver() 
        sess.run(tf.global_variables_initializer())


        for epoch_i in range(1, self.epochs+1):
            epoch_train_loss = 0
            for batch_i, (questions_batch, answers_batch) in enumerate(
                    self.batch_data(self.train_reviews, self.train_replies, self.batch_size)):
                start_time = time.time()
                _, loss = sess.run(
                    [train_op, cost],
                    {input_data: questions_batch,
                    targets: answers_batch,
                    lr: self.learning_rate,
                    sequence_length: answers_batch.shape[1],
                    keep_prob: self.keep_probability})

                epoch_train_loss += loss
                total_train_loss += loss
                end_time = time.time()
                batch_time = end_time - start_time

                if batch_i % display_step == 0:
                    print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                        .format(epoch_i,
                                self.epochs, 
                                batch_i, 
                                len(self.train_reviews) // self.batch_size, 
                                total_train_loss / display_step, 
                                batch_time*display_step))
                    total_train_loss = 0
                    
                    # Reduce learning rate, but not below its minimum value
                    self.learning_rate *= self.learning_rate_decay
                    if self.learning_rate < self.min_learning_rate:
                        self.learning_rate = self.min_learning_rate
            
            total_valid_loss = 0
            start_time = time.time()
            for batch_ii, (questions_batch, answers_batch) in \
                    enumerate(self.batch_data(self.valid_reviews, self.valid_replies, self.batch_size)):
                valid_loss = sess.run(
                cost, {input_data: questions_batch,
                        targets: answers_batch,
                        lr: self.learning_rate,
                        sequence_length: answers_batch.shape[1],
                        keep_prob: 1})
                total_valid_loss += valid_loss
            end_time = time.time()
            batch_time = end_time - start_time
            avg_valid_loss = total_valid_loss / (len(self.valid_reviews) / self.batch_size)
            print('Valid Loss: {:>6.3f}, Seconds: {:>5.2f}'.format(avg_valid_loss, batch_time))

            avg_train_loss = epoch_train_loss/ (len(self.train_reviews) / self.batch_size)
            total_epoch_valid_loss.append(avg_valid_loss)
            total_epoch_train_loss.append(avg_train_loss)
            if avg_valid_loss <= min(total_epoch_valid_loss):
                print('New Record!') 
                stop_early = 0
                saver.save(sess, checkpoint)

            else:
                print("No Improvement.")
                stop_early += 1
                if stop_early == stop:
                    break

            if stop_early == stop:
                print("Stopping Training.")
                break

        with open('result/train_loss.pickle', 'wb') as f:
            pickle.dump(total_epoch_train_loss,f)

        with open('result/valid_loss.pickle', 'wb') as f:
            pickle.dump(total_epoch_valid_loss,f)

        def question_to_seq(question, vocab_to_int):
            '''Prepare the question for the model'''
        
            question = self.clean_text(question)
            return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in question.split()]


        
        def reply(input_question):
            answer_ref = {self.vocab_reduced[i]:i for i in self.vocab_reduced.keys()}

            # Prepare the question
            input_question = question_to_seq(input_question, self.vocab_reduced)
            # Pad the questions until it equals the max_line_length
            input_question = input_question + [self.vocab_reduced["<PAD>"]] * (self.maxlen - len(input_question))
            # Add empty questions so the the input_data is the correct shape
            batch_shell = np.zeros((self.batch_size, self.maxlen))
            # Set the first question to be out input question
            batch_shell[0] = input_question    
            # Run the model with the input question
            answer_logits = sess.run(inference_logits, {input_data: batch_shell, 
                                                        keep_prob: 1.0})[0]

            # Remove the padding from the Question and Answer
            pad_q = self.vocab_reduced["<PAD>"]
            pad_a = self.vocab_reduced["<PAD>"]

            print('Question')
            print('  Input Words: {}'.format([answer_ref[i] for i in input_question if i != pad_q]))

            print('\nAnswer')
            print('  Response Words: {}'.format([answer_ref[i] for i in np.argmax(answer_logits, 1) if i != pad_a]))

        # Generate result
        # Create your own input question
        input_question1 = 'It keeps crashing.'
        input_question2 = "Helpful in keeping track of attacks and possibly triggers."
        reply(input_question1)
        reply(input_question2)