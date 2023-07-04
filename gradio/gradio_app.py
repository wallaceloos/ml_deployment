import gradio as gr
import numpy as np
import torch
from torchvision import transforms
import requests

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

def predict_image(input_image):

    img = np.asarray(input_image)
    img = preprocess(img)
    input_batch = img.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    model.eval()
    output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    top5_prob, top5_catid = torch.topk(probabilities, 5)
    confidences = {labels[top5_catid[i]]: float(top5_prob[i]) for i in range(5)}   
    
    return confidences

if __name__ == "__main__":
    gr.Interface(fn=predict_image, 
             inputs=gr.Image(type="pil"),
             outputs=gr.Label(num_top_classes=5)).launch(share=False)