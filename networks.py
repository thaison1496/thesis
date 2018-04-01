from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, TimeDistributed, Activation, \
                        Bidirectional, Masking, Embedding, Input, Concatenate


def building_ner(num_lstm_layer, num_hidden_node, dropout, \
    time_step, vector_length, output_lenght, embedd_matrix):
    model = Sequential()
    # 439169 - number of embedded words, 300 - embedding size
    model.add(Embedding(embedd_matrix.shape[0], embedd_matrix.shape[1], weights=[embedd_matrix], input_length=time_step, trainable=False))
    model.add(Masking(mask_value=0., input_shape=(time_step, vector_length)))
    for i in range(num_lstm_layer-1):
        model.add(Bidirectional(LSTM(units=num_hidden_node, return_sequences=True, dropout=dropout,
                                     recurrent_dropout=dropout)))
    model.add(Bidirectional(LSTM(units=num_hidden_node, return_sequences=True, dropout=dropout,
                                 recurrent_dropout=dropout), merge_mode='concat'))
    model.add(TimeDistributed(Dense(output_lenght)))
    model.add(Activation('softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def building_ner2(num_lstm_layer, num_hidden_node, dropout, \
    time_step, vector_length, output_lenght, embedd_matrix, additional_feature_size):
    # model = Sequential()
    inp = Input(batch_shape=(None, time_step), name="word_index")
    inp_add = Input(batch_shape=(None, time_step, additional_feature_size), name="additional_feature")
    embedd = Embedding(embedd_matrix.shape[0], embedd_matrix.shape[1], weights=[embedd_matrix], input_length=time_step, trainable=False)(inp)
    masking = Masking(mask_value=0., input_shape=(time_step, vector_length))(embedd)
    concat = Concatenate()([masking, inp_add])
    # for i in range(num_lstm_layer-1):
    #     model.add(Bidirectional(LSTM(units=num_hidden_node, return_sequences=True, dropout=dropout,
    #                                  recurrent_dropout=dropout)))
    lstm = Bidirectional(LSTM(units=num_hidden_node, return_sequences=True, dropout=dropout,
                                 recurrent_dropout=dropout))(concat)
    dense = TimeDistributed(Dense(output_lenght))(lstm)
    activation = Activation('softmax')(dense)
    model = Model(input=[inp, inp_add], output=activation)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

