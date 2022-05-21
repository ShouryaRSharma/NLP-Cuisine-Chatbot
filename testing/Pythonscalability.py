import requests
import concurrent
import time
from concurrent.futures import ThreadPoolExecutor

sequence = [1, 2, 5, 10, 20]


#load test

def sendReq(url):
    return requests.post(url, json={"text": "Hi I would like to order some burgers"})

def req(num):
    urls = ["http://localhost:5000/ask"]*num
    with ThreadPoolExecutor(max_workers = num) as pool:
         responseL = list(pool.map(sendReq, urls))
         return responseL
def clear():
    a = requests.post("http://localhost:5000/ask", json={"text": "please cancel order"})
    a = requests.post("http://localhost:5000/ask", json={"text": "yes cancel order"})


for n in sequence:
    clear()
    print("\n Number of requests: " + str(n))
    start_time = time.perf_counter()
    responses = req(n)
    for response in responses:
        print(response)
    print(time.perf_counter() - start_time, "seconds")

clear()
