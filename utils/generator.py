import time
import pickle 
import re
import numpy as np
import os
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
tf.__version__

def model_inputs():
    '''Initialize model inputs'''
    input_data = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    return input_data, targets, lr, keep_prob

def process_encoding_input(target_data, vocab_to_int, batch_size):
    '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    return dec_input

def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
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

def decoding_layer_train(encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope,
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

def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id,
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

def decoding_layer(dec_embed_input, dec_embeddings, encoder_state, vocab_size, sequence_length, rnn_size,
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

        train_logits = decoding_layer_train(encoder_state, 
                                            dec_cell, 
                                            dec_embed_input, 
                                            sequence_length, 
                                            decoding_scope, 
                                            output_fn, 
                                            keep_prob, 
                                            batch_size)
        decoding_scope.reuse_variables()
        infer_logits = decoding_layer_infer(encoder_state, 
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

def seq2seq_model(input_data, target_data, keep_prob, batch_size, sequence_length, answers_vocab_size, 
                  questions_vocab_size, enc_embedding_size, dec_embedding_size, rnn_size, num_layers, 
                  questions_vocab_to_int):
    
    '''Use the previous functions to create the training and inference logits'''
    
    enc_embed_input = tf.contrib.layers.embed_sequence(input_data, 
                                                       answers_vocab_size+1, 
                                                       enc_embedding_size,
                                                       initializer = tf.random_uniform_initializer(0,1))
    enc_state = encoding_layer(enc_embed_input, rnn_size, num_layers, keep_prob, sequence_length)

    dec_input = process_encoding_input(target_data, questions_vocab_to_int, batch_size)
    dec_embeddings = tf.Variable(tf.random_uniform([questions_vocab_size+1, dec_embedding_size], 0, 1))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    
    train_logits, infer_logits = decoding_layer(dec_embed_input, 
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

# Set the Hyperparameters
epochs = 100
batch_size = 128
rnn_size = 512
num_layers = 2
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.005
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.75


# load data
fpath = 'data/google_play_review.pickle'
with open(fpath, 'rb') as f:
    df = pickle.load(f)

# clean text
def clean_text(text):
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

reviews = [clean_text(i) for i in df.reviews if isinstance(i, str)]
replies = [clean_text(j) for (i,j) in zip(df.reviews, df.replies) if isinstance(i, str)]

minlen = 2
maxlen = 100

# Filter out the reviews/replies that are too short/long
review_reply = [(i,j) for (i,j) in zip(reviews, replies) if (minlen<=len(i.split())<=maxlen) & (minlen<=len(j.split())<=maxlen)]

all_vob = []
for (i,j) in review_reply:
    all_vob = all_vob + i.split() + j.split()

# Remove rare words from the vocabulary.
threshold = 5
vocab = {i:all_vob.count(i) for i in set(all_vob)}
vocab = {word:vocab[word] for word in vocab.keys() if vocab[word] > threshold}

# The ratio of remaining words
print(len(vocab)/len(set(all_vob)))

adds = ['<PAD>','<EOS>','<UNK>','<GO>']
vocab_reduced = adds + list(vocab.keys()) 
vocab_reduced = {word: idx for (idx, word) in enumerate(vocab_reduced)}
with open('data/vocab_reduced.pickle','wb') as f: 
    pickle.dump(vocab_reduced,f) 

# replace rare words with <UNK>
# add <EOS> to the end of every reply.
# transform reviews/replies to a list of list of index
review_matrix = []
reply_matrix = []
for review, reply in review_reply:
    review_int = [vocab_reduced[word] if word in vocab_reduced.keys() else vocab_reduced['<UNK>'] for word in review.split()]
    reply_int = [vocab_reduced[word] if word in vocab_reduced.keys() else vocab_reduced['<UNK>'] for word in reply.split()]
    reply_int.append(vocab_reduced['<EOS>'])
    review_matrix.append(review_int)        
    reply_matrix.append(reply_int)

# Reset the graph to ensure that it is ready for training
tf.reset_default_graph()
# Start the session
sess = tf.InteractiveSession()
    
# Load the model inputs    
input_data, targets, lr, keep_prob = model_inputs()
# Sequence length will be the max line length for each batch
sequence_length = tf.placeholder_with_default(maxlen, None, name='sequence_length')
# Find the shape of the input data for sequence_loss
input_shape = tf.shape(input_data)

# Create the training and inference logits
train_logits, inference_logits = seq2seq_model(
    tf.reverse(input_data, [-1]), targets, keep_prob, batch_size, sequence_length, len(vocab_reduced), 
    len(vocab_reduced), encoding_embedding_size, decoding_embedding_size, rnn_size, num_layers, 
    vocab_reduced)

# Create a tensor for the inference logits, needed if loading a checkpoint version of the model
tf.identity(inference_logits, 'logits')

with tf.name_scope("optimization"):
    # Loss function
    cost = tf.contrib.seq2seq.sequence_loss(
        train_logits,
        targets,
        tf.ones([input_shape[0], sequence_length]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)

def pad_sentence_batch(sentence_batch, vocab_to_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]

def batch_data(questions, answers, batch_size):
    """Batch questions and answers together"""
    for batch_i in range(0, len(questions)//batch_size):
        start_i = batch_i * batch_size
        questions_batch = questions[start_i:start_i + batch_size]
        answers_batch = answers[start_i:start_i + batch_size]
        pad_questions_batch = np.array(pad_sentence_batch(questions_batch, vocab_reduced))
        pad_answers_batch = np.array(pad_sentence_batch(answers_batch, vocab_reduced))
        yield pad_questions_batch, pad_answers_batch

# train_valid split
train_valid_split = int(len(review_matrix)*0.1)

# Split the reviews and replies into training and validating data
train_reviews = review_matrix[train_valid_split:]
train_replies = reply_matrix[train_valid_split:]

valid_reviews = review_matrix[:train_valid_split]
valid_replies = reply_matrix[:train_valid_split]

print(len(train_reviews))
print(len(valid_reviews))

display_step = 100 # Check training loss after every 100 batches
stop_early = 0 
stop = 5 # If the validation loss does decrease in 5 consecutive checks, stop training
validation_check = ((len(train_reviews))//batch_size//2)-1 # Modulus for checking validation loss
total_train_loss = 0 # Record the training loss for each display step
summary_valid_loss = [] # Record the validation loss for saving improvements in the model

checkpoint = "model/best_model.ckpt" 

sess.run(tf.global_variables_initializer())

import ipdb; ipdb.set_trace()
for epoch_i in range(1, epochs+1):
    for batch_i, (questions_batch, answers_batch) in enumerate(
            batch_data(train_reviews, train_replies, batch_size)):
        start_time = time.time()
        _, loss = sess.run(
            [train_op, cost],
            {input_data: questions_batch,
             targets: answers_batch,
             lr: learning_rate,
             sequence_length: answers_batch.shape[1],
             keep_prob: keep_probability})

        total_train_loss += loss
        end_time = time.time()
        batch_time = end_time - start_time

        if batch_i % display_step == 0:
            print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                  .format(epoch_i,
                          epochs, 
                          batch_i, 
                          len(train_reviews) // batch_size, 
                          total_train_loss / display_step, 
                          batch_time*display_step))
            total_train_loss = 0

        if batch_i % validation_check == 0 and batch_i > 0:
            total_valid_loss = 0
            start_time = time.time()
            for batch_ii, (questions_batch, answers_batch) in \
                    enumerate(batch_data(valid_reviews, valid_replies, batch_size)):
                valid_loss = sess.run(
                cost, {input_data: questions_batch,
                       targets: answers_batch,
                       lr: learning_rate,
                       sequence_length: answers_batch.shape[1],
                       keep_prob: 1})
                total_valid_loss += valid_loss
            end_time = time.time()
            batch_time = end_time - start_time
            avg_valid_loss = total_valid_loss / (len(valid_reviews) / batch_size)
            print('Valid Loss: {:>6.3f}, Seconds: {:>5.2f}'.format(avg_valid_loss, batch_time))
            
            # Reduce learning rate, but not below its minimum value
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate

            summary_valid_loss.append(avg_valid_loss)
            if avg_valid_loss <= min(summary_valid_loss):
                print('New Record!') 
                stop_early = 0
                saver = tf.train.Saver() 
                saver.save(sess, checkpoint)

            else:
                print("No Improvement.")
                stop_early += 1
                if stop_early == stop:
                    break
    
    if stop_early == stop:
        print("Stopping Training.")
        break

# Generate result
# Create your own input question
answer_ref = {vocab_reduced[i]:i for i in vocab_reduced.keys()}
input_question = 'Bad Application hate it annoying do not want use it again'

def question_to_seq(question, vocab_to_int):
    '''Prepare the question for the model'''
     
    question = clean_text(question)
    return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in question.split()]

# Prepare the question
input_question = question_to_seq(input_question, vocab_reduced)

# Pad the questions until it equals the max_line_length
input_question = input_question + [vocab_reduced["<PAD>"]] * (maxlen - len(input_question))
# Add empty questions so the the input_data is the correct shape
batch_shell = np.zeros((batch_size, maxlen))
# Set the first question to be out input question
batch_shell[0] = input_question    
    
# Run the model with the input question
answer_logits = sess.run(inference_logits, {input_data: batch_shell, 
                                            keep_prob: 1.0})[0]

# Remove the padding from the Question and Answer
pad_q = vocab_reduced["<PAD>"]
pad_a = vocab_reduced["<PAD>"]

print('Question')
print('  Word Ids:      {}'.format([answer_ref[i] for i in input_question if i != pad_q]))
print('  Input Words: {}'.format([answer_ref[i] for i in input_question if i != pad_q]))

print('\nAnswer')
print('  Word Ids:      {}'.format([answer_ref[i] for i in np.argmax(answer_logits, 1) if i != pad_a]))
print('  Response Words: {}'.format([answer_ref[i] for i in np.argmax(answer_logits, 1) if i != pad_a]))