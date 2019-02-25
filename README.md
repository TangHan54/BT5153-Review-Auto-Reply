# BT5153-Review-Auto-Reply

This project aims to generate replies to users' reviews automatically with NLP techniques.

## Plan
1. Scrap training data from Google Play Store.
2. Text Processing.
    - language identification
3. Modelling.

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


## To do
requirements.txt