import gen_vector
import networks
import argparse
import numpy as np
from datetime import datetime
from keras.callbacks import EarlyStopping, ModelCheckpoint
import subprocess
import shlex
import utils

num_lstm_layer = 2
num_hidden_node = 64
dropout = 0.5
batch_size = 50
patience = 10
max_epochs = 5

startTime = datetime.now()

print('Loading data...')
input_train, output_train, input_test, output_test, alphabet_tag = gen_vector.create_data()
print('Building model...')

time_step, input_length = np.shape(input_train)[1:]
output_length = np.shape(output_train)[2]

ner_model = networks.building_ner(num_lstm_layer, num_hidden_node, dropout, time_step, input_length, output_length)
print('Model summary...')
print(ner_model.summary())
print('Training model...')



exit()

early_stopping = EarlyStopping(patience=patience)
model_save = ModelCheckpoint('weights.{epoch:02d}-{loss:.2f}.hdf5', save_best_only=True, monitor='loss', mode='min', period=1)
history = ner_model.fit(input_train, output_train, batch_size=batch_size, epochs=max_epochs,
                         callbacks=[model_save])
print('Testing model...')

answer = ner_model.predict_classes(input_test, batch_size=batch_size)
utils.predict_to_file(answer, output_test, alphabet_tag, 'out.txt')
input = open('out.txt')
p1 = subprocess.Popen(shlex.split("perl conlleval.pl"), stdin=input)
p1.wait()

endTime = datetime.now()
print("Running time: ")
print (endTime - startTime)
