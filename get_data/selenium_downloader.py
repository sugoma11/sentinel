from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import Select
import time

# put in this directory chrome driver

def download():
    global driver
    # click on "download image"
    button = driver.find_element(by=By.XPATH, value='/html/body/div[1]/div[2]/div[1]/div[11]/div[4]/div/img')
    button.click()
    time.sleep(5)
    # select dropdown with 'png' or 'jpg' formats
    dropdown = Select(driver.find_element(by=By.XPATH, value='/html/body/div[1]/div[2]/div[4]/div[2]/div/div[2]/div[1]/div[7]/div/select'))
    # choose png
    dropdown.select_by_value('png')
    time.sleep(3)
    button = driver.find_element(by=By.XPATH, value='/html/body/div[1]/div[2]/div[4]/div[2]/div/div[2]/a')
    button.click()

    time.sleep(3)
    button = driver.find_element(by=By.XPATH, value='//*[@id="app"]/div[4]/div[2]/span')
    button.click()
    time.sleep(3)
    prev_url = driver.current_url
    button = driver.find_element(by=By.XPATH, value= '//*[@id="visualization-tab"]/div/div/div[1]/div[2]/div[1]/div[1]'
                                                     '/div/div/i[3]')
    button.click()
    time.sleep(5)

    if driver.current_url == prev_url:
        exit(0)


def setup():
    global chrome_options
    chrome_options = Options()
    global driver
    driver = webdriver.Chrome(options=chrome_options)
    url = 'https://apps.sentinel-hub.com/eo-browser/?zoom=14&lat=44.7117&lng=37.80988&themeId=DEFAULT-THEME&visualizationUrl=https%3A%2F%2Fservices.sentinel-hub.com%2Fogc%2Fwms%2Fbd86bcc0-f318-402b-a145-015f85b9427e&datasetId=S2L2A&fromTime=2021-05-18T00%3A00%3A00.000Z&toTime=2021-05-18T23%3A59%3A59.999Z&layerId=1_TRUE_COLOR&demSource3D=%22MAPZEN%22'
    driver.get(url)
    # acception of use-terms
    button = driver.find_element(by=By.XPATH, value='/html/body/div[1]/div[3]/div[2]/div/div[2]/div[1]')
    button.click()
    time.sleep(3)
    # closing welcome-banner
    button = driver.find_element(by=By.XPATH, value='/html/body/div[5]/div/div/div/div/button/span')
    button.click()
    time.sleep(3)
    # removing labels
    # activate checkbox
    checkbox = driver.find_element(by=By.XPATH, value='/html/body/div[1]/div[2]/div[1]/div[2]/div[2]/div')
    checkbox.click()
    time.sleep(3)
    # uncheck labels box
    button = driver.find_element(by=By.XPATH, value='/html/body/div[1]/div[2]/div[1]/div[2]/div[2]/div/section/div[3]/label[3]/div/span')
    button.click()
    time.sleep(3)
    button = driver.find_element(by=By.XPATH, value='//*[@id="visualization-tab"]/div/div/div[1]/div[2]/div[1]/div[1]/div/div/i[2]')
    button.click()
    time.sleep(3)
    button = driver.find_element(by=By.XPATH, value='//*[@id="visualization-tab"]/div/div/div[1]/div[2]/div[2]/div/div[1]/div/div[1]/div/div[3]')
    button.click()
    time.sleep(3)
    actions = ActionChains(driver)
    actions.move_to_element(button)
    time.sleep(4)
    actions.move_by_offset(-65, 0)
    time.sleep(4)
    actions.click().perform()
    time.sleep(2)


setup()
for i in range(0, 2500):
    download()
