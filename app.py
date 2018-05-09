from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
import pickle
import tensorflow
from flask import Flask, request, url_for, jsonify
from flask_restful import Resource, Api
import json
import werkzeug

host = 'http://127.0.0.1'
port = 5000

app = Flask(__name__)
api = Api(app)


class ScoreModel(Resource):
    def __init__(self):
        # set file paths
        self.model_file_path = 'models/toxic_comment_DNN.h5'
        self.weights_file_path = "models/weights.best.hdf5"
        self.tokenizer_file_path = 'models/tokenizer.pkl'

        # loading the tokenizer
        with open(self.tokenizer_file_path, 'rb') as handle:
            tokenizer = pickle.load(handle)

        # load the model json
        with open(self.model_file_path, 'rb') as f:
            model_json = pickle.load(f)

        # create the model from json and load the weights
        model_loaded = model_from_json(model_json)
        model_loaded.load_weights(self.weights_file_path, by_name=False)
        self.tokenizer = tokenizer
        self.model = model_loaded

    def predict(self, data):
        '''
        Core scoring function for the model
        :param json_string: with a key of 'text' where the text features are extracted
        :return: json_string of predictions with
        keys: 'non_toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate

        '''
        if type(data) is str:
            data = [data]
        features = self.tokenizer.texts_to_matrix(data)
        preds = self.model.predict(features)
        keys = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        output = dict(zip(keys, preds.transpose().tolist()))
        return output


@app.route('/')
def api_root():
    return 'RestAPI for predictive model, to test'


@app.route('/echo', methods=['POST'])
def echo():
    json_data = request.get_json()
    text = json_data.get('text')
    return 'Rest API Pass through :' + text


@app.route('/score', methods=['POST'])
def score_model():
    from keras.models import model_from_json
    from keras.preprocessing.text import Tokenizer
    import pickle
    import tensorflow
    import tensorflow as tf
    from keras import backend as K

    config = tf.ConfigProto(allow_soft_placement=True, device_count={'CPU': 1, 'GPU': 0})
    session = tf.Session(config=config)
    K.set_session(session)
    json_data = request.get_json()
    text = json_data.get('text')
    if type(text) is str:
        input_string = [text]
    if type(input_string) is list:
        s = ScoreModel()
        output_dict = s.predict(input_string)
        return json.dumps(output_dict)
    else:
        return json.dumps('mode scoring scipped, because of input type :' + str(type(input_string)))



if __name__ == '__main__':
    app.run()

# test api from command line
# curl -H "Content-Type: application/json" -X POST -d "{\"text\":\"this is a test\"}" http://127.0.0.1:5000/score

