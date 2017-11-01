from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
import time, sys

def wait_until_chatting(driver):
    try:
        element_present = EC.presence_of_element_located((By.CLASS_NAME, 'message_wrapper'))
        WebDriverWait(driver, 30).until(element_present)
        print("[CONNECT] Successfully connected")
        return driver
    except TimeoutException:
        print('[TIMEOUT] Timeout Occurs. Try wait until chatting.')
        resdriver = wait_until_chatting(driver)
        return resdriver

final_str = ""

driver = webdriver.PhantomJS('/Users/outtoin/Documents/lib/bin/phantomjs')
driver.get('https://coinone.co.kr/chart/')

wait_until_chatting(driver)

time.sleep(10)

while (True):
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    chats = soup.select('#canvas > ul > li')

    for chat in chats:
        time_text = chat.select('div.time')[0].get_text()
        id = chat.select('div.message_wrapper > span.nickname')[0]['title']
        text = chat.select('div.message_wrapper > span.chat-message')[0].get_text()
        print(time_text, '|', id, ':', text)

    print('-----------------------------------')
    time.sleep(60)
