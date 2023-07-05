<p align="justify"> MLOps stands for <strong> Machine Learning Operations</strong>; operations that come from the development/operations (DevOps) term, which is a methodology in software development and operation.  In this short tutorial we will focus only in the deployment phase! We will deploy a simple model using: Gradio, Streamlit, Docker, and Flask. The model is a Resnet-152 that predicts a class given an image. Pytorch was used to implement this model, but this tutorial applies to a different library/framework as well.   
  
 ### Web Server

<p align="justify"> Just a quick review about web server. This is not necessary for the deployment itself, but it helps to understand some of the vocabulary that will be used later on, but feel free to skip it. When you type a web address on your computer browser you are making a Hypertext Transfer Protocol (HTTP) request to that address, which is converted to an IP address by the Domain Name System (DNS). One quick reminder, your computer usually has a private IP address that is translated by the Network Address Translation (NAT) to a public IP address, however with IPv6 this will not be more necessary. Then your request is sent to a server. We can have different types of servers: file server, web server, email server, and so on. A web server, simply saying, stores and delivers the content for a website using HTTP. The HTTP is the communication protocol between the client (your browser) and the server. One popular web server is NGINX. It can be used as a reverse proxy that I will talk more about in the Flask section. I think this is enough to understand the next steps.   
  
### Gradio and Streamlit  
  
<p align="justify">Gradio and Streamlit provide a very fast way for prototyping your model, and if you do not have much experience with front-end, it is okay, because you can build one with a few lines of code in Python. 

<p align="justify"><a href=https://gradio.app/>Gradio</a> is  a Python library that helps to create friendly web interfaces for your machine learning application. The disadvantage is that you do not have much flexibility regarding the UI components. The installation of Gradio can be performed using pip install gradio. For our web application the user will be able to upload an image and the model will return the top 5 most likely classes that the image belongs to, and their probabilities. This interface  can be done with the only 3 lines shown below.

```python
if __name__ == "__main__":
    gr.Interface(fn=predict_image, 
                 inputs=gr.Image(type="pil"), 
                 outputs=gr.Label(num_top_classes=5)).launch(share=False)
```   
<p align="center">
<img src="https://github.com/wallaceloos/ml_deployment/blob/main/imgs/gradio_ops.PNG" width="90%" height="90%">
</p>  
<p align="justify">The input is an image and the output are the top 5 classes. The predict_image function has like argument the image data, and after applying the preprocessing function on it, the model performs the prediction and returns the labels. 
  
```python
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
```  
<p align="center">
<img src="https://github.com/wallaceloos/ml_deployment/blob/main/imgs/gradio_ops_result.PNG" width="90%" height="90%">
</p>

<p align="justify"><a href=https://streamlit.io/>Streamlit</a> gives you a little bit more flexibility than Gradio, but with great power comes great responsibility (and more lines of codes). You have more autonomy about how to organize your UI components. Similar to Gradio, Streamlit can be installed using pip install streamlit. I am going to create a similar interface that I created using Gradio. Streamlit already has a function to upload files, that we can simply use by calling the function st.file_uploader.  

```python
if __name__ == "__main__":

    st.title("Resnet For Image Classification :ship: ")

    with st.sidebar:
        file_obj = st.file_uploader('Choose an image:', type=['jpg', 'png', 'jpeg'])
```  
<p align="center">
<img src="https://github.com/wallaceloos/ml_deployment/blob/main/imgs/streamlit_ops.PNG" width="90%" height="90%">
</p>
<p align="justify">Next step I define the layout of the page with two columns, where the first column will depict the image and the second column will show the predictions. When the image is uploaded, it is preprocessed and the model is called in order to make the predictions. The predict function is the same function that was used in the Gradio example.

```python
    col1,col2 = st.columns(2)
    with col1:
        if file_obj is not None:
            img = Image.open(file_obj)
            img = np.array(img)
            st.image(img)

            if st.button("Submit"):
                ypred = predict_image(img)
    with col2:    
        if len(ypred) > 0:
            for k, v in ypred.items():
                st.progress(int(100*v), text=k+ " (" + str(int(np.round(v*100))) + "\%)")
```  

<p align="center">
<img src="https://github.com/wallaceloos/ml_deployment/blob/main/imgs/streamlit_ops_result.PNG" width="90%" height="90%">
</p>    

<p align="justify">You can host these apps on Hugging Face Space. If you don’t want to use a GPU, it’s free, but if you want to use one, you can select the many options that Hugging Face provides. For production purposes, you might prefer a different way for deployment.   

### Docker    

<p align="justify"> <a href=https://www.docker.com/>Docker</a> is a platform to build and run container images. A container is a runnable instance of an image that is portable and isolated from all other processes on the host machine. An image contains an executable application that is created using a dockerfile. Dockerfile is a text file where you will describe what you need in order to run your application. Thus, after having your application done, you can add everything that your app needs inside of a container and ship it away! Virtual Machine (VM) and containers are both virtualization tools, and the main difference between them is that VMs run on their own operating system, while containers run on the host kernel. They don’t need a hypervisor (which is a guest operating system).  

<p align="justify"> In this example I’ll take our gradio app and make a container of it. Since our app is already done, we can already write our dockerfile. Actually we need to make only subtle changes. In the moment that we launch the app, we can set the server name to “0.0.0.0” to make the app accessible on the local network, and also we can specify a port, in this case I choose 7000. 

```python
gr.Interface(fn=predict_image,
            inputs=gr.Image(type="pil"),
            outputs=gr.Label(num_top_classes=5)).launch(server_name="0.0.0.0", server_port=7000)
```
<p align="justify">The first instruction is FROM and it will specify which parent image we want. In this case we need one that supports cuda. The ARG instruction defines a variable that will be used during the docker build time, and ENV instruction sets the environment variables. These instructions were used just to avoid “interaction” while the container was being built. The instruction RUN executes any commands and WORKDIR sets the working directory for any RUN, CMD, ENTRYPOINT, COPY and ADD instructions that follow it in the dockerfile. The instruction COPY copies new files or directories from a source into the container path. The ENTRYPOINT specifies the command that will be executed when the container starts. 
  
```dockerfile
FROM nvidia/cuda:11.4.3-runtime-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America

RUN apt-get update && apt install -y software-properties-common; \
    add-apt-repository ppa:deadsnakes/ppa; \
    apt-get update; \
    apt install -y python3.9; \
    apt install -y python3-pip; \
    pip3 install torch>=1.13.1+cu117;

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

ENTRYPOINT ["python3", "main.py"]
```
<p align="justify">The first lines of our dockerfile consist in downloading our parent image and then installing the dependencies necessary to run our app. The file requirements.txt contains all the python libraries necessary for our app. Then, we copy all files to our container and define our ENTRYPOINT.  

To build the image use:  

`docker build –-tag gradio_docker .`  

To run the container use:  

`docker run --rm -t -p 7000:7000 --gpus device=0 gradio_docker`   

`–rm`: automatically remove the container when it exits   
`-t`: allocate a pseudo-TTY. Enables interaction with OS by handling input and output   
`-p`: publish a container’s port(s) to the host (hostPort:containerPort)   
`-gpus`: allows you to access NVIDIA GPU resources, and `device` specifies the GPUs    

<p align="justify">After creating your container you can upload it to the docker hub. Docker hub is a repository provided by docker where you can share your container image with anyone that you want. To push your image to docker hub you can do it

`docker login --username username`  
`docker tag my-image username/my-repo`  
`docker push username/my-repo`  

### Flask  

<p align="justify"> <a href=https://flask.palletsprojects.com/>Flask</a> is a web framework written in Python that helps to develop web applications. It is based on Web Server Gateway Interface (WSGI) and has a built-in HTTP server. However, it is not suitable for production, only for development and testing. It is recommended to use a dedicated web server for production, like NGINX. Adding a web server in front of the WSGI server is called a reverse proxy. It forwards requests from the web server to a backend Python application. In order to make our web application communicate with our server we can use REST API. API stands for application programming interface, and it is used to communicate your application with other applications. REST stands for Representational State Transfer and it is a specification to help standardize information exchange, and when web APIs follow this specification are called RESTful API. 

<p align="justify">The following example shows how to send an image from the client to our server in order to make a prediction. The front-end can be created using HTML, Javascript, etc, but I will use Python as well. After reading the image I create a JSON (Javascript Object Notation) file and using a HTTP method, POST, I send a request to the specified url. It will send the data to our server. 

```python
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
param = {"img":input_image.tolist()}
response = requests.post(url, json = param, headers=headers)
```

<p align="justify">On the server side, the route function from Flask, maps the url to a specific Python function through an decorator. The endpoint name for the route is “/model“ and it is defined by us. It could have any other name. The function returns a response text that is displayed on the client side. The preprocessing and prediction of the image is made in a very similar way to the previous examples shown before.   

```python
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
``` 
