from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

import numpy as np
import time

def wait(d: int) -> None:
    start = time.time()
    while time.time() - start < d:
        continue

def download(wanted):

    if wanted is None or len(wanted) == 0:
        return

    target = 'https://donneespubliques.meteofrance.fr/?fond=donnee_libre&prefixe=Txt%2FMarine%2FArchive%2Fmarine&extension=csv.gz&date='

    opt = Options()
    opt.add_argument('--window-size=2560,1600')
    prefs = {
        'download.default_directory': '/Users/david/Desktop/Sophia/EBCWF/Data/Buoy'
    }
    opt.add_experimental_option('prefs', prefs)

    s = Service(executable_path='/usr/local/chromedriver')

    driver = webdriver.Chrome(service=s, options=opt)

    for date in wanted:
        driver.get(target + date)
        try: # close pop-up window if it appears
            buttonSelector = '#fenetremodale > p > input[type=submit]:nth-child(4)'
            acceptTrackers = driver.find_element(By.CSS_SELECTOR, buttonSelector)
            acceptTrackers.click()
        except NoSuchElementException:
            pass
        wait(5)

    driver.close()

def main():

    request = input(
        'Enter the years whose data you want to extract\n' + 
        'Either a single year (with all of the four digits) or following the format YYYY-YYYY\n>>> '
    )
    if '-' in request:
        try:
            beg, end = int(request[:4]), int(request[5:])
        except ValueError:
            print('Invalid request')
            return
        assert 1996 < beg <= end < 2022
    else:
        try:
            beg = int(request)
        except ValueError:
            print('Invalid request')
            return
        assert 1996 < beg < 2022
        end = beg

    download([f'{i}0{j}' for i in range(beg, end + 1) for j in range(5, 10)])

if __name__ == '__main__':
    main()
