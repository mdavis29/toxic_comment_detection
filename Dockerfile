FROM tensorflow/tensorflow:2.2.0
EXPOSE 5000
COPY . /app
WORKDIR /app
RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["app.py"]
