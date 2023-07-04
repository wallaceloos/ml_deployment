import json
import requests
import cv2

url = 'http://127.0.0.1:8001/model'

filename = "mlops.PNG"
input_image = cv2.imread(filename)
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
input_image = cv2.resize(input_image, (200,200))
request_data = json.dumps({'img':input_image.tolist()})

headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
param = {"img":input_image.tolist()}
response = requests.post(url, json = param, headers=headers)

print(response.text)