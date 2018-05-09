from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
import pickle
import tensorflow
import json

# This script generates the scoring and schema files
# Creates the schema, and holds the init and run functions needed to
# operationalize the Iris Classification sample

# Import data collection library. Only supported for docker mode.
# Functionality will be ignored when package isn't found
try:
    from azureml.datacollector import ModelDataCollector
except ImportError:
    print("Data collection is currently only supported in docker mode. May be disabled for local mode.")
    # Mocking out model data collector functionality
    class ModelDataCollector(object):
        def nop(*args, **kw): pass
        def __getattr__(self, _): return self.nop
        def __init__(self, *args, **kw): return None
    pass

import os



class ScoreModel():
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

def score():
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


def main():
    from azureml.api.schema.dataTypes import DataTypes
    from azureml.api.schema.sampleDefinition import SampleDefinition
    from azureml.api.realtime.services import generate_schema
    import pandas

    df = pandas.DataFrame(data=[[3.0, 3.6, 1.3, 0.25]],
                          columns=['sepal length', 'sepal width', 'petal length', 'petal width'])

    # Turn on data collection debug mode to view output in stdout
    os.environ["AML_MODEL_DC_DEBUG"] = 'true'

    # Test the output of the functions
    init()
    input1 = pandas.DataFrame([[3.0, 3.6, 1.3, 0.25]])
    print("Result: " + run(input1))

    inputs = {"input_df": SampleDefinition(DataTypes.PANDAS, df)}

    # Genereate the schema
    generate_schema(run_func=run, inputs=inputs, filepath='./outputs/service_schema.json')
    print("Schema generated")


import os

if __name__ == "__main__":
    main()
# test api from command line
# curl -H "Content-Type: application/json" -X POST -d "{\"text\":\"this is a test\"}" http://127.0.0.1:5000/score

