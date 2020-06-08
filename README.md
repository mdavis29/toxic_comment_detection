
## README

This project is a demonstration of how to use a deploy a deep learning model as a service. The
model uses natural language processing and tensorflow neural network to estimate whether the probablilty that a text string is 'toxic' or otherwise hate speech or obesene.

This project can be run three different ways,
+ as a local app (using local python interpreter)
+ as a local docker container, with the app exposed through the container
+ as a kubernetes micros service, loading the image from docker hu

#### Inputs
json payload with
```sh
"{\"text\":\"this is a test\"}"
```

#### Outputs
json payload with
```
{"non -toxic": [0.9540797472000122], "severe_toxic": [0.011814841069281101], "obscene": [0.09778174012899399], "threat": [0.0021917244885116816], "insult": [0.31358540058135986], "identity_hate": [0.2735470235347748]}
```

## Python project to predict Toxic Comments
Data is from kaggle Toxic Comment Challenge

#### Combining Docker and Flask to Deploy Rest API and Basic Web service
This repo is an example of how to take a pre trained tensorflow model that estimates
probabilties that speech.  It uses a docker container and a flask app to expose the models for inference on new data.

#### License
see attached license file

#### Warning!!!
This data set contains racists and offensive words for the purposes of training algorithms to detect such speech and
is not appropriate for minors,  and is not a reflection of the authors views or opinions.

### Running Locally:
#### To run app locally
```sh
python app.py
```

#### To test
```sh
 curl -H "Content-Type: application/json" -X POST -d "{\"text\":\"this is a test\"}" http://127.0.0.1:5000/score
```

### Running Docker:
#### Building with Docker
```sh
sudo docker build -t toxiccomment:latest .
```

#### Running with Docker in Background
```sh
sudo docker run -d -p 5000:5000 toxiccomment:latest
```

#### Running with Docker in terminal
```sh
sudo docker run -it -p 5000:5000 toxiccomment:latest
```

### Deployming with Minikube

#### Sending image to dockerhub for kubernetes deployment
to deploy on Kubernetes, a container image needs to be availible on docker hub or
a private repository. This image contains the app, and is pushed from the latest built container, after tagging the container

 + logs in
 + adds tag
 + pushes to docker hub
 For kubernetes deployments, a docker image containing the app my be pushed to a container
 registry.
```sh
sudo docker login
sudo docker tag toxiccomment:latest mdavis29/datascience_example:toxiccomment
sudo docker push mdavis29/datascience_example:toxiccomment
```

#### Run on Minikube with a yaml file

Start minikube (if testing on a local kubernetes instance)
```sh
 minikube start
```

This uses a yaml file to run configure the kubernetes services, and example using
a load-balancer is attached.

```sh
kubectl apply -f load-balancer-deployment.yaml
kubectl expose deployment toxiccomment --type="LoadBalancer" --name=toxiccomment --target-port=5000 --port=5000
minikube service toxiccomment --url
 ```

##### Clean Up Kubernetes
remove all the services and deployments

```sh
kubectl delete services toxiccomment
kubectl delete deployment toxiccomment
```
