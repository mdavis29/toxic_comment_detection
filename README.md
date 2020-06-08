
## README

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

#### Run on Minikube
creates a Minikube deployment using the image from docker hub
```sh
sudo kubectl create deployment toxiccomment --image=mdavis29/datascience_example:toxiccomment
```

exposes the service
```sh
sudo kubectl expose deployment toxiccomment --type=LoadBalancer --port=5000
```
Shows the port mapping, where the servicce is exposed
```sh
kubectl get services
```
```sh
minikube service toxiccomment
```
