# Keras 2 Functional API code

from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import Adam
from keras.layers import Dense, Activation, Dropout, Flatten, Input, LSTM, Embedding, Merge
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, LearningRateScheduler
from time import time
import math
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras import backend as K
batch_size = 16



def word2vec(embed_matrix, num_words, embed_dim, seq_length, drop_rate):
	inputs = Input(shape=seq_length)
	embed = Embedding(number_words, embed_dim, trainable=False)(inputs)
	lstm_1 = LSTM(512, return_sequences=True)(embed)
	drop_1 = Dropout(drop_rate)(lstm_1)
	lstm_2 = LSTM(512, return_sequences=False)(drop_1)
	drop_2 = Dropout(drop_rate)

	dense = Dense(1024, activation='tanh')
	model = Model(inputs=inputs, outputs=dense)


	return model


def build_vgg(img_shape=(416, 416, 3), n_classes=16, n_layers=16, l2_reg=0.,
                load_pretrained=False, freeze_layers_from='base_model'):
    # Decide if load pretrained weights from imagenet
    if load_pretrained:
        weights = 'imagenet'
    else:
        weights = None

    # Get base model
    if n_layers==16:
        base_model = VGG16(include_top=False, weights=weights,
                           input_tensor=None, input_shape=img_shape)
    elif n_layers==19:
        base_model = VGG19(include_top=False, weights=weights,
                           input_tensor=None, input_shape=img_shape)
    else:
        raise ValueError('Number of layers should be 16 or 19')

    # Add final layers
    x = base_model.output
    x = Flatten(name="flatten")(x)
    x = Dense(1024, activation='relu', name='dense_1')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', name='dense_2')(x)
    x = Dropout(0.5)(x)
    x = Dense(n_classes, name='dense_3_{}'.format(n_classes))(x)
    predictions = Activation("softmax", name="softmax")(x)

    # This is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze some layers
    if freeze_layers_from is not None:
        if freeze_layers_from == 'base_model':
            print ('   Freezing base model layers')
            for layer in base_model.layers:
                layer.trainable = False
        else:
            for i, layer in enumerate(model.layers):
                print(i, layer.name)
            print ('   Freezing from layer 0 to ' + str(freeze_layers_from))
            for layer in model.layers[:freeze_layers_from]:
               layer.trainable = False
            for layer in model.layers[freeze_layers_from:]:
               layer.trainable = True
    # adam = Adam(0.0001)
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[f1, 'accuracy'])

    return model 


def vqa_model(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate, num_classes):
    vgg_model = build_vgg()
    lstm_model = Word2VecModel(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate)
    print "Merging final model..."

    merged = Merge([vgg_model, lstm_model], mode='mul')
    
    drop_1 = Dropout(dropout_rate)(merged)
    dense_1 = Dense(1000, activation='tanh')(drop_1)
    drop_2 = Dropout(dropout_rate)(dense_1)
    dense_2 = Dense(num_classes, activation='softmax')(drop_2)

    fc_model = Model(inputs=merged, outputs=dense_2)
    
    fc_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
        metrics=['accuracy'])
    return fc_model
















# Training



import numpy as np
from keras.models import model_from_json#load_model
from keras.callbacks import ModelCheckpoint
import os
import argparse
from models import *
from prepare_data import *
from constants import *

def get_model(dropout_rate, model_weights_filename):
    print "Creating Model..."
    metadata = get_metadata()
    num_classes = len(metadata['ix_to_ans'].keys())
    num_words = len(metadata['ix_to_word'].keys())

    embedding_matrix = prepare_embeddings(num_words, embedding_dim, metadata)
    model = vqa_model(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate, num_classes)
    if os.path.exists(model_weights_filename):
        print "Loading Weights..."
        model.load_weights(model_weights_filename)

    return model

def train(args):
    dropout_rate = 0.5
    train_X, train_y = read_data(args.data_limit)    
    model = get_model(dropout_rate, model_weights_filename)
    checkpointer = ModelCheckpoint(filepath=ckpt_model_weights_filename,verbose=1)
    model.fit(train_X, train_y, nb_epoch=args.epoch, batch_size=args.batch_size, callbacks=[checkpointer], shuffle="batch")
    model.save_weights(model_weights_filename, overwrite=True)

def val():
    val_X, val_y, multi_val_y = get_val_data() 
    model = get_model(0.0, model_weights_filename)
    print "Evaluating Accuracy on validation set:"
    metric_vals = model.evaluate(val_X, val_y)
    print ""
    for metric_name, metric_val in zip(model.metrics_names, metric_vals):
        print metric_name, " is ", metric_val

    # Comparing prediction against multiple choice answers
    true_positive = 0
    preds = model.predict(val_X)
    pred_classes = [np.argmax(_) for _ in preds]
    for i, _ in enumerate(pred_classes):
        if _ in multi_val_y[i]:
            true_positive += 1
    print "true positive rate: ", np.float(true_positive)/len(pred_classes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--data_limit', type=int, default=215359, help='Number of data points to fed for training')
    args = parser.parse_args()

    if args.type == 'train':
        train(args)
    elif args.type == 'val':
        val()
