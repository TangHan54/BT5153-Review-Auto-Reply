import pickle 
import re
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
    
    return text

reviews = [clean_text(i) for i in df.reviews if isinstance(i, str)]
replies = [clean_text(j) for (i,j) in zip(df.reviews, df.replies) if isinstance(i, str)]

min_line_length = 8
max_line_length = 100

# Filter out the reviews/replies that are too short/long
review_reply = [(i,j) for (i,j) in zip(reviews, replies) if (min_line_length<=len(i.split())<=max_line_length) & (min_line_length<=len(j.split())<=max_line_length)]

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
vocab_list = vocab.keys() + adds

# We will aim to replace fewer than 5% of words with <UNK>

