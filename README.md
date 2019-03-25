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

## To do
requirements.txt