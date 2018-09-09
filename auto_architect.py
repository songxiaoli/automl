from keras.optimizers import RMSprop
import json
import pandas as pd
from keras.models import model_from_json
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

with open('mapping_layers.json') as f:
    mapping_layers = json.load(f)

def create_layer(cur_layer_class, pre_layer_name, layer_names):
    cur_layer_dict = mapping_layers[cur_layer_class]
    
    if cur_layer_class in layer_names:
        cur_layer_name = cur_layer_class.lower() + '_' + str(layer_names[cur_layer_class])
        layer_names[cur_layer_class] = layer_names[cur_layer_class]+1
    else:
        cur_layer_name = cur_layer_class.lower() + '_0'
        layer_names[cur_layer_class] = 1 
    cur_layer_dict['name'] = cur_layer_name
    cur_layer_dict['config']['name'] = cur_layer_name
    
    cur_layer_dict['inbound_nodes'] = [[[pre_layer_name, 0, 0, {}]]]
    
    return cur_layer_name, cur_layer_dict, layer_names

def create_arc(input_, output_, layers):
    arch = {}
    arch['class_name'] = 'Model'    
    arch['keras_version'] = '2.2.0'
    arch['backend'] = 'tensorflow'
    
    arch['config'] = {}
    arch['config']['name'] = 'model'
    arch['config']['input_layers'] = mapping_layers['input_layers']
    
    
    arch['config']['layers'] = []
    input_layer = mapping_layers['InputLayer']
    input_layer['name'] = 'input'
    input_layer['config']['name'] = 'input'
    input_layer['config']['batch_input_shape'] = [None, input_.shape[1]]
    arch['config']['layers'].append(input_layer)
    
    layer_names = {}
    pre_layer_name = 'input'
    for cur_layer_class in layers:
        cur_layer_name, cur_layer_dict, layer_names = create_layer(cur_layer_class, pre_layer_name, layer_names)
        arch['config']['layers'].append(cur_layer_dict)
        pre_layer_name = cur_layer_name
        
    output_layers = mapping_layers['output_layers']
    output_layers[0][0] = pre_layer_name
    arch['config']['output_layers'] = output_layers
    return arch


da = pd.read_csv('amazon_reviews_us_Mobile_Electronics_v1_00.tsv', sep='\t', error_bad_lines=False)
da = da[['review_body', 'star_rating']]
da = da.dropna()
X = da.review_body
Y = da.star_rating
le = LabelEncoder()
Y = le.fit_transform(Y)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)
max_words = 1000
max_len = 20
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

a = create_arc(sequences_matrix, Y_train, ['Embedding', 'Bidirectional', 'Dense'])
mod = model_from_json(json.dumps(a))
mod.compile(loss='sparse_categorical_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
mod.fit(sequences_matrix,Y_train,batch_size=128,epochs=1,validation_split=0.2)
