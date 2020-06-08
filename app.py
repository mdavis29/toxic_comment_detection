from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
import pickle
import tensorflow
from flask import Flask, request, url_for, jsonify
from flask_restful import Resource, Api
import json
import h5py
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
import pickle
from flask import Flask, url_for, render_template, redirect


host = 'http://127.0.0.1'
port = 5000

app = Flask(__name__)
api = Api(app)


def load_model():
        # set file paths
    model_file_path = 'models/toxic_comment_DNN.h5'
    weights_file_path = "models/weights.best.hdf5"
    tokenizer_file_path = 'models/tokenizer.pkl'

    # loading the tokenizer
    with open(tokenizer_file_path, 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load the model json
    with open(model_file_path, 'rb') as f:
        model_json = pickle.load(f)

    # create the model from json and load the weights
    model = model_from_json(model_json)
    model.load_weights(weights_file_path, by_name=False)
    print('model loaded')
    return model, tokenizer


def predict( data):
    '''
    Core scoring function for the model
    :param json_string: with a key of 'text' where the text features are extracted
    :return: json_string of predictions with
    keys: 'non_toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate

    '''
    if type(data) is str:
        data = [data]
    features = tokenizer.texts_to_matrix(data)
    preds = model.predict(features)
    keys = ['non -toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    output = dict(zip(keys, preds.transpose().tolist()))
    return output


@app.route('/')
def index():
    text=request.args.get('text', '')
    text = text.strip()
    output_dict = predict(text)

    if len(text) > 0:
        return render_template('results.html', text=output_dict)
    return render_template('index.html')

@app.route('/echo', methods=['POST'])
def echo():
    json_data = request.get_json()
    text = json_data.get('text')
    return 'Rest API Pass through : {}'.format(text)


@app.route('/score', methods=['POST'])
def score():
    json_data = request.get_json()
    text = json_data.get('text')
    print(text)
    output_dict = predict(text)
    return json.dumps(output_dict)


if __name__ == '__main__':
    model, tokenizer = load_model()
    app.run(host='0.0.0.0', threaded=False)

# test api from command line
# curl -H "Content-Type: application/json" -X POST -d "{\"text\":\"this is a test\"}" http://127.0.0.1:5000/score
