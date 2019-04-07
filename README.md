# BT5153-Review-Auto-Reply
Contributors: Derek Li Lingling, Veronica Hu He, Zheng Yiran, Sophia Yue Qi Hui, Jason Chew Guo Jie, Tang Han

This project aims to generate replies to users' reviews automatically with NLP techniques.

## Plan
1. Scrap training data from Google Play Store.
2. Text Processing.
    - language identification
3. Modelling.
    - [Word Embedding](https://www.tensorflow.org/alpha/tutorials/sequences/word_embeddings)

## Data Source 
[Migraine Buddy](https://play.google.com/store/apps/details?id=com.healint.migraineapp)

## Initialization
1. Create the virtual environment
> mkvirtualenv auto-reply\
> workon auto-reply\
> pip install -r requirements.txt
2. [install chrome driver](https://sites.google.com/a/chromium.org/chromedriver/downloads)\
Place it in the PATH /usr/bin or /usr/local/bin.
If the PATH is unknown, do the following:
> echo $PATH

## To generate the 'generator model'
> python utils/generator.py

## tockenize 
> subword-nmt learn-joint-bpe-and-vocab --input data/train.txt -s 10000 -o data/code.bpe --write-vocabulary data/vocab.bpe

> subword-nmt apply-bpe -c data/code.bpe --vocabulary data/vocab.bpe --vocabulary-threshold 5 < data/train.reviews.txt > data/train.reviews.bpe

> subword-nmt apply-bpe -c data/code.bpe --vocabulary data/vocab.bpe --vocabulary-threshold 5 < data/train.replies.txt > data/train.replies.bpe

> subword-nmt apply-bpe -c data/code.bpe --vocabulary data/vocab.bpe --vocabulary-threshold 5 < data/test.reviews.txt > data/test.reviews.bpe

> subword-nmt apply-bpe -c data/code.bpe --vocabulary data/vocab.bpe --vocabulary-threshold 5 < data/test.replies.txt > data/test.replies.bpe

```
export DATA_PATH=$(pwd)/data

export VOCAB_SOURCE=${DATA_PATH}/vocab.bpe
export VOCAB_TARGET=${DATA_PATH}/vocab.bpe
export TRAIN_SOURCES=${DATA_PATH}/train.reviews.bpe
export TRAIN_TARGETS=${DATA_PATH}/train.replies.bpe
export DEV_SOURCES=${DATA_PATH}/test.reviews.bpe
export DEV_TARGETS=${DATA_PATH}/test.replies.bpe

export DEV_TARGETS_REF=${DATA_PATH}/test.replies.pred.txt
```

## Train model 
```
export MODEL_DIR=${TMPDIR:-/tmp}/nmt_tutorial
export TRAIN_STEPS=10000
mkdir -p $MODEL_DIR

python -m bin.train \
  --config_paths="
      ./example_configs/nmt_small.yml,
      ./example_configs/train_seq2seq.yml,
      ./example_configs/text_metrics_bpe.yml" \
  --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET" \
  --input_pipeline_train "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $TRAIN_SOURCES
      target_files:
        - $TRAIN_TARGETS" \
  --input_pipeline_dev "
    class: ParallelTextInputPipeline
    params:
       source_files:
        - $DEV_SOURCES
       target_files:
        - $DEV_TARGETS" \
  --batch_size 32 \
  --train_steps $TRAIN_STEPS \
  --output_dir $MODEL_DIR

```