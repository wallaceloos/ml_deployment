from flask import Flask, request
import numpy as np
import torch
import requests
from PIL import Image
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

app = Flask(__name__)

@app.route('/model', methods=['POST'])
def predict():
    request_data = request.get_json(force=True)
    input_image = request_data['img']
    input_image = np.asarray(input_image)
    input_image = input_image.astype(np.uint8)

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    model.eval()
    output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    list_pred = ""
    for i in range(5):
        list_pred += "Top " + str(i+1)+ ": " + categories[top5_catid[i]] + "({:.2f}".format(top5_prob[i])+"%) \n"
    
    return list_pred
    
if __name__ == "__main__":
    app.run(port=8001, debug=True)