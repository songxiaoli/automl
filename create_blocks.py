from keras.models import model_from_json
from keras.models import Model,load_model
from keras.layers import Embedding
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import Conv1D
from keras.layers import MaxPool1D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Concatenate
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import pandas as pd
import json

from keras.models import Model,load_model

#create a model from scratch

embeddingLayer = Embedding(1000,128, 
                           input_length=20,trainable=False)
sequenceInput = Input(shape=(20,),dtype='int32')
embeddedSequences = embeddingLayer(sequenceInput)
gru = Bidirectional(LSTM(128,return_sequences=True))(embeddedSequences)
filterSizes = [2,4,5]
filters = 100
convs=[]
avgs =[]
for fsize in filterSizes:
    conv = Conv1D(filters=filters,kernel_size=fsize,activation='relu')(gru)
    pool = MaxPool1D(pool_size=2)(conv)
    pool = Dropout(0.2)(pool)
    pool = Flatten()(pool)
    convs.append(pool)

convout = Concatenate(axis=-1)(convs)
out = Dense((5),activation='softmax')(convout)
mod = Model(sequenceInput,[out])
mod.summary()


model_json = json.loads(mod.to_json())
layers = model_json['config']['layers']
mapping_layers = {}
for i in [0, 1,2,3,6,9,12,15,16]:
    l = layers[i]
    del l['name']
    del l['inbound_nodes']
    class_ = l['class_name']
    mapping_layers[class_] = l
mapping_layers['InputLayer']['inbound_nodes'] = []
mapping_layers['Bidirectional']['config']['layer']['config']['return_sequences'] = False

input_layers = model_json['config']['input_layers']
input_layers[0][0] = 'input'
output_layers = model_json['config']['output_layers']

mapping_layers['input_layers'] = input_layers
mapping_layers['output_layers'] = output_layers

with open('mapping_layers.json', 'w') as outfile:
    json.dump(mapping_layers, outfile)

