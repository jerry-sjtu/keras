import numpy as np
from scipy.sparse import csc_matrix,csr_matrix

row = np.array([0, 2, 2, 0, 1, 2])
col = np.array([0, 0, 1, 2, 2, 2])
data = np.array([1, 2, 3, 4, 5, 6])
sparse_matrix = csc_matrix((data, (row, col)), shape=(300, 300))
sparse_matrix.toarray()

indptr = np.array([0, 2, 3, 6])
indices = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
sparse_matrix = csr_matrix((data, indices, indptr), shape=(3, 3)).toarray()

from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer

max_words = 1000
batch_size = 32
nb_epoch = 5

print('Loading data...')
(X_train, y_train), (X_test, y_test) = reuters.load_data(nb_words=max_words, test_split=0.2)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

nb_classes = np.max(y_train)+1
print(nb_classes, 'classes')

print('Vectorizing sequence data...')
tokenizer = Tokenizer(nb_words=max_words)
X_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
X_test = tokenizer.sequences_to_matrix(X_test, mode='binary')
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Convert class vector to binary class matrix (for use with categorical_crossentropy)')
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)


print('Building model...')
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                nb_epoch=nb_epoch, batch_size=batch_size,
                verbose=1)#, validation_split=0.1)
score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])


X_train_sparse = csr_matrix(X_train)

def batch_generator(X, y, batch_size):
    n_batches_for_epoch = X.shape[0]//batch_size
    for i in range(n_batches_for_epoch):
        index_batch = range(X.shape[0])[batch_size*i:batch_size*(i+1)]
        X_batch = X[index_batch,:].todense()
        y_batch = y[index_batch,:]
        yield(np.array(X_batch),y_batch)


def batch_generator(X, y, batch_size):
    number_of_batches = 10000 / batch_size
    counter=0
    shuffle_index = np.arange(np.shape(y)[0])
    np.random.shuffle(shuffle_index)
    X = X[shuffle_index, :]
    y = y[shuffle_index]
    while 1:
        index_batch = shuffle_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[index_batch,:].todense()
        y_batch = y[index_batch]
        counter += 1
        yield(np.array(X_batch),y_batch)
        if (counter < number_of_batches):
            np.random.shuffle(shuffle_index)
            counter=0

model.fit_generator(generator=batch_generator(X_train_sparse, Y_train, batch_size),
                    nb_epoch=nb_epoch,
                    samples_per_epoch=X_train_sparse.shape[0])