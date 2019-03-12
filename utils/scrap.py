from selenium import webdriver
import time
from bs4 import BeautifulSoup, Comment
import pandas as pd
import re
import pickle

def scrap(link="https://play.google.com/store/apps/details?id=com.healint.migraineapp&hl=en&showAllReviews=true"):
    SCROLL_PAUSE_TIME = 10

    #Setting up Chrome webdriver Options
    chrome_options = webdriver.ChromeOptions()

    #creating Chrome webdriver instance with the set chrome_options
    driver = webdriver.Chrome(chrome_options=chrome_options)
    driver.get(link)

    #Initialization
    last_height = 0
    new_height = 1
    counter = 0

    while True:
        if last_height < new_height:
            # Get new scroll height
            last_height = driver.execute_script("return document.body.scrollHeight")

            # Scroll down to bottom
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            # Wait to load page
            time.sleep(SCROLL_PAUSE_TIME)

            # Get new scroll height
            new_height = driver.execute_script("return document.body.scrollHeight")
            
        else:
            try:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight - 1000)")
                time.sleep(SCROLL_PAUSE_TIME)
                driver.find_element_by_css_selector('.RveJvd.snByac').click()
                # Wait to load page
                time.sleep(SCROLL_PAUSE_TIME)
                # Get new scroll height
                new_height = driver.execute_script("return document.body.scrollHeight")
                
            except:
                print("end of task. Scroll number is " + str(counter))
                break
                
        counter = counter + 1

                    
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    customers = []
    dates = []
    ratings = []
    helpfuls = []
    reviews = []
    replies = []

    for record in soup.find_all('div', class_="xKpxId"):
    
        customer_name = record.find("span" , class_="X43Kjb").text
        date = record.find("span" , class_="p2TkOb").text
        rating = record.find('div', {"aria-label" : re.compile('Rate *')})['aria-label']
        helpful = record.find('div', {"aria-label" : 'Number of times this review was rated helpful'}).text
        review_short = record.find_next('span', {'jsname' : "bN97Pc"}).text
        review_long = record.find_next('span', {'jsname' : "fbQN7e"}).text
        if review_long != "":
            review = review_long
        else:
            review = review_short

        reply = record.find_next('div', class_="LVQB0b").contents[2]
        
        if reply == "":
            reply = " "
    
    
    customers.append(customer_name)
    dates.append(date)
    ratings.append(rating)
    helpfuls.append(helpful)
    reviews.append(review)
    replies.append(reply)

    output = pd.DataFrame(customers, columns=['customer_name'])
    output['ratings'] = pd.DataFrame(ratings)
    output['helpfuls'] = pd.DataFrame(helpfuls)
    output['reviews'] = pd.DataFrame(reviews)
    output['replies'] = pd.DataFrame(replies)
    
    fpath = 'data/google_play_review.pickle' 
    with open(fpath, 'wb') as f:
        pickle.dump(output,f)
