import os
import time
import gc
import random
import pandas as pd
import numpy as np
import keras
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from keras.datasets import mnist, cifar10, cifar100, fashion_mnist


# A class that cycles through a categorical data set and serves out a fixed number of samples from each class
class CategoricalDataGenerator:
    # data_y is a list of integer class labels
    def __init__(self, data_x, data_y, num_classes):
        self.feature_shape = data_x.shape[1:]
        self.num_classes = num_classes
        self.data_x = []
        self.cur_indices = [0 for i in range(self.num_classes)]

        # Save data for each class
        for i in range(self.num_classes):
            indices = np.reshape(data_y, data_y.shape[0]) == i
            self.data_x.append(data_x[indices])

    # Get num_samples many points from each class in classes
    def next_block(self, classes, num_samples, shuffle_data=True):
        total_num_samples = len(classes) * num_samples
        out_x = np.zeros(((total_num_samples,) + self.feature_shape))
        out_y = np.zeros((total_num_samples, self.num_classes))

        for i in range(len(classes)):
            c = classes[i]
            cur_index = self.cur_indices[c]

            num_available = self.data_x[c].shape[0] - cur_index
            num_left = num_samples
            out_begin = i * num_samples
            out_end = i * num_samples + min(num_left, num_available)
            in_begin = cur_index
            in_end = cur_index + min(num_left, num_available)  # = self.data_x[c].shape[0]
            while num_left > 0:
                out_x[out_begin:out_end] = self.data_x[c][in_begin:in_end]

                cur_index = in_end
                num_left -= out_end - out_begin
                num_available = self.data_x[c].shape[0] - cur_index
                if num_available == 0: num_available = self.data_x[c].shape[0]

                out_begin = out_end
                out_end += min(num_available, num_left)
                in_begin = in_end if in_end != self.data_x[c].shape[0] else 0
                in_end = in_begin + min(num_available, num_left)

            self.cur_indices[c] = cur_index

            out_y[i * num_samples:(i + 1) * num_samples, c] = 1

        # Shuffle data
        if shuffle_data:
            shuffle_ind = list(range(out_x.shape[0]))
            random.shuffle(shuffle_ind)
            out_x = out_x[shuffle_ind, :]
            out_y = out_y[shuffle_ind]

        return out_x, out_y


# Load CIFAR10/100 images as ndarrays
def load_cifar_from_keras(proportion_of_data=1.0, shuffle=True, num=10):
    if num == 10:
        (train_x, train_y), (test_x, test_y) = cifar10.load_data()
    else:
        (train_x, train_y), (test_x, test_y) = cifar100.load_data(label_mode='fine')

    if shuffle:
        train_shuffle_ind = list(range(train_x.shape[0]))
        test_shuffle_ind = list(range(test_x.shape[0]))
        random.shuffle(train_shuffle_ind)
        random.shuffle(test_shuffle_ind)

        train_x = train_x[train_shuffle_ind, :, :, :]
        train_y = train_y[train_shuffle_ind]
        test_x = test_x[test_shuffle_ind, :, :, :]
        test_y = test_y[test_shuffle_ind]

        train_x = train_x / 255.0
        test_x = test_x / 255.0

    if proportion_of_data != 1.0:
        num_train = int(proportion_of_data * train_x.shape[0])
        num_test = int(proportion_of_data * test_x.shape[0])
        train_x = train_x[:num_train, :, :, :]
        train_y = train_y[:num_train]
        test_x = test_x[:num_test, :, :, :]
        test_y = test_y[:num_test]

    return train_x, train_y, test_x, test_y


# Load MNIST or fashion MNIST images
def load_mnist_from_keras(proportion_of_data=1.0, fashion=False, shuffle=True):
    loader = fashion_mnist if fashion else mnist
    (train_x, train_y), (test_x, test_y) = loader.load_data()

    # Flatten images into 1D array
    train_x = np.reshape(train_x, (train_x.shape[0], -1))
    test_x = np.reshape(test_x, (test_x.shape[0], -1))

    # Shuffle before taking subsets
    if shuffle:
        train_shuffle_ind = list(range(train_x.shape[0]))
        test_shuffle_ind = list(range(test_x.shape[0]))
        random.shuffle(train_shuffle_ind)
        random.shuffle(test_shuffle_ind)

        train_x = train_x[train_shuffle_ind, :]
        train_y = train_y[train_shuffle_ind]
        test_x = test_x[test_shuffle_ind]
        test_y = test_y[test_shuffle_ind]

    train_x = train_x / 255.0
    test_x = test_x / 255.0

    if proportion_of_data != 1.0:
        num_train = int(proportion_of_data * train_x.shape[0])
        num_test = int(proportion_of_data * test_x.shape[0])
        train_x = train_x[:num_train, :]
        train_y = train_y[:num_train]
        test_x = test_x[:num_test, :]
        test_y = test_y[:num_test]

    return train_x, train_y, test_x, test_y


# Return a short dense sequential network description of the form, e.g., 2-10-10-5
# Dropout is indicated as, e.g. 2-10d-10d-5
def get_model_description(model):
    layers = model.get_config()['layers']
    desc = str(layers[0]['config']['batch_input_shape'][1])
    for layer in layers:
        if layer['class_name'] == 'Dropout':
            desc += 'd'
        else:
            desc += '-' + str(layer['config']['units'])
    return desc


# Create a curriculum from a list of categories
#  - lessons: dict {str name:integer list} of lists of categories included in each lesson
def categorical_curriculum(data_train_x, data_train_y, data_test_x, data_test_y, lessons):
    lesson_train_x = []
    lesson_train_y = []
    lesson_test_x = []
    lesson_test_y = []
    for name, lesson in lessons:
        train_indices = [y in lesson for y in data_train_y]
        test_indices = [y in lesson for y in data_test_y]

        train_x = data_train_x[train_indices]
        train_y = data_train_y[train_indices]
        test_x = data_test_x[test_indices]
        test_y = data_test_y[test_indices]

        train_y = to_categorical(train_y, num_classes=num_classes)
        test_y = to_categorical(test_y, num_classes=num_classes)

        lesson_train_x.append(train_x)
        lesson_train_y.append(train_y)
        lesson_test_x.append(test_x)
        lesson_test_y.append(test_y)

    return lesson_train_x, lesson_train_y, lesson_test_x, lesson_test_y


# Create a curriculum from an increasing list of percentages of data to include
def percent_curriculum(data_train_x, data_train_y, data_test_x, data_test_y, data_percents):
    new_curric = [(str(int(100 * p)) + '%' if p != 1.0 else 'all', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) for p in data_percents]

    data_train_y = to_categorical(data_train_y, num_classes=num_classes)
    data_test_y = to_categorical(data_test_y, num_classes=num_classes)

    lesson_train_x = []
    lesson_train_y = []
    lesson_test_x = []
    lesson_test_y = []
    for p in data_percents:
        end = int(data_train_x.shape[0] * p)
        lesson_train_x.append(data_train_x[:end])
        lesson_train_y.append(data_train_y[:end])
        lesson_test_x.append(data_test_x)
        lesson_test_y.append(data_test_y)

    return lesson_train_x, lesson_train_y, lesson_test_x, lesson_test_y, new_curric


# Create a strictly increasing curriculum
def fixed_subset_curriculum(subset_size, num_lessons, forced_classes=None):
    new_curric = []
    for i in range(num_lessons + 1):
        if i == num_lessons:
            new_curric.append(('all', None))
        else:
            new_curric.append((str(i + 1) + ' out of ' + str(num_lessons) + ' lessons', None))

    return new_curric


# Append a history returned by fit() to the suit's history
def append_history(suit, history, epochs, samples_per_epoch, update_best_accs):
    if 'hist' not in suit:
        suit['hist'] = history.history
        suit['hist']['training_samples_seen'] = [(i + 1) * samples_per_epoch for i in range(epochs)]
    else:
        suit['hist']['val_acc'] += history.history['val_acc']
        suit['hist']['acc'] += history.history['acc']
        suit['hist']['training_samples_seen'][len(suit['hist']['training_samples_seen']):] = \
            [suit['hist']['training_samples_seen'][-1] + (i + 1) * samples_per_epoch for i in range(epochs)]

    if update_best_accs:
        max_acc_cur_step = np.max(suit['hist']['acc'][len(suit['hist']['acc']) - epochs:])
        max_val_acc_cur_step = np.max(suit['hist']['val_acc'][len(suit['hist']['val_acc']) - epochs:])
        if ('best_acc' in suit['hist'] and max_acc_cur_step > suit['hist']['best_acc']) or 'best_acc' not in suit['hist']:
            suit['hist']['best_acc'] = max_acc_cur_step
        if ('best_val_acc' in suit['hist'] and max_val_acc_cur_step > suit['hist']['best_val_acc']) or 'best_val_acc' not in suit['hist']:
            suit['hist']['best_val_acc'] = max_val_acc_cur_step

        for i in [5, 10, 15]:
            suit['hist']['val acc %de' % (i)] = np.max(history.history['val_acc'][:i])


# Zeros weights/biases into the units of the output layer that correspond to classes outside 'lesson'
# Returns a copy of the weights/biases that are being set to zero
def freeze_params_outside_lesson(model, num_classes, lesson):
    w = model.get_weights()
    weights_into = w[-2].copy()
    biases = w[-1].copy()

    for i in range(num_classes):
        if i in lesson:
            continue

        w[-2][:, i] = np.full(weights_into.shape[0], -10000000)
        w[-1][i] = -10000000

    model.set_weights(w)
    return weights_into, biases


# Set weights/biases into the units of the output layer that correspond to classes outside 'lesson'
def set_params_outside_lesson(model, num_classes, lesson, weights_into, biases):
    w = model.get_weights()

    for i in range(num_classes):
        if i in lesson:
            continue

        w[-2][:, i] = weights_into[:, i]
        w[-1][i] = biases[i]

    model.set_weights(w)


# Load data set: MNIST, fashion MNIST, CIFAR10, or CIFAR100
num_classes = 100
# orig_train_x, orig_train_y, orig_test_x, orig_test_y = load_mnist_from_keras(proportion_of_data=1.0, fashion=True)
orig_train_x, orig_train_y, orig_test_x, orig_test_y = load_cifar_from_keras(proportion_of_data=1.0, num=100)

# Create base curriculum containing all training data presented at once
curriculum0 = [('all', list(range(num_classes)))]

# Define test suits
'''
Format:

----- parameters -----
'name': description of curriculum, must be unique in test_suits
'curric': the lesson list, may be unspecified if 'data_percent' is specified
'data_percent': list of fractions of data to use on the ith epoch iteration. 'curric' is ignored if this is specified
'subset size': creates len(epochs)-1 many random lessons with 'subset size' many elements, then one lesson
               with all categories. 'curric' is ignored if this is specified
'epochs': list of epochs to perform, aligns with curriculum lessons and unit additions
'base model': list of hidden layer unit counts
'add_units': list of pairs (l, n), after 1st iteration for each pair n units are added to layer l.
             Not included in dict means no units added
'weight_init': keyword specifying how to initialize weights for new hidden units, see add_hidden_units()
               May be left out of dict if 'add_units' is left out
'repeat': integer number of times to repeat the suit, must be included and >= 1
'freeze': prevent training on subsets of classes from altering weights at output layer outside that subset

----- internal -----
'curric_train_x', 'curric_train_y', 'curric_test_x', 'curric_test_y': lists storing train/test data for each lesson in curriculum specified by 'curric'
'hist': used to store training and test accuracies
'''

test_suits = [
    {'name': '500-500',
     'curric': curriculum0,
     'epochs': [80],
     'batch size': 1024,
     'freeze': False,
     'base model': [500, 200],
     'dropout': 0.1,
     'repeat': 5},
    {'num subsets presented': 100,
     'epochs per subset': 1,
     'subset size': 8,
     'batches per class per epoch': 8,
     'batch size': 32,
     'class pool update size': 8,
     'num full epochs at end': 15,
     'freeze': True,
     'repeat': 1},
    {'num subsets presented': 100,
     'epochs per subset': 1,
     'subset size': 8,
     'batches per class per epoch': 16,
     'batch size': 32,
     'class pool update size': 8,
     'num full epochs at end': 15,
     'freeze': True,
     'repeat': 1},
]

# Copy unlisted parameters from first suit
for i in range(1, len(test_suits)):
    for k, v in test_suits[0].items():
        if k not in test_suits[i]:
            test_suits[i][k] = v

# Prepare suit names and epoch schedule
for suit in test_suits:
    if 'subset size' in suit:
        # Set suit name
        s = '%dx%dx%dx%d, %d-subsets' % (suit['num subsets presented'], suit['epochs per subset'], suit['batches per class per epoch'], suit['batch size'], suit['subset size'])
        s += ', freeze' if suit['freeze'] else ', no freeze'
        if 'name' in suit:
            suit['name'] += ', ' + s
        else:
            suit['name'] = s

        suit['epochs'] = []
        for i in range(suit['num subsets presented']):
            suit['epochs'].append(suit['epochs per subset'])
        suit['epochs'].append(suit['num full epochs at end'])

# Prepare suit curriculum data
for suit in test_suits:
    if 'data_percent' in suit:
        suit['curric_train_x'], suit['curric_train_y'], suit['curric_test_x'], suit['curric_test_y'], suit['curric'] = \
            percent_curriculum(orig_train_x, orig_train_y, orig_test_x, orig_test_y, suit['data_percent'])
    elif 'subset size' in suit:
        suit['curric'] = fixed_subset_curriculum(suit['subset size'], len(suit['epochs']) - 1, suit.get('forced_classes'))
    else:
        suit['curric_train_x'], suit['curric_train_y'], suit['curric_test_x'], suit['curric_test_y'] = \
            categorical_curriculum(orig_train_x, orig_train_y, orig_test_x, orig_test_y, suit['curric'])

# Prepare repeated suits
for i in range(len(test_suits)):
    for j in range(1, test_suits[i]['repeat']):
        new_suit = test_suits[i].copy()
        new_suit['name'] = new_suit['name'] + ' (repeat ' + str(j + 1) + ')'
        new_suit['repeat'] = None
        test_suits.append(new_suit)


for suit in test_suits:
    start_time = time.time()
    class_accuracies_avg = 0
    prev_accuracies = []
    class_pool = list(range(num_classes))
    suit['class freq'] = [[] for i in range(num_classes)]
    suit['distinct classes'] = []
    suit['train data gen'] = CategoricalDataGenerator(orig_train_x, orig_train_y, num_classes)
    suit['test data gen'] = CategoricalDataGenerator(orig_test_x, orig_test_y, num_classes)

    # Define model: either dense feedforward as specified by the suit or a preset convolutional network
    model = Sequential()
    if 'base model' in suit:
        if len(suit['base model']) == 0:
            model.add(Dense(units=num_classes, activation='softmax', input_dim=orig_train_x.shape[1]))
        else:
            if len(suit['base model']) == 1:
                model.add(Dense(units=suit['base model'][0], activation='relu', input_dim=orig_train_x.shape[1]))
                if 'dropout' in suit: model.add(Dropout(suit['dropout']))
            else:
                model.add(Dense(units=suit['base model'][0], activation='relu', input_dim=orig_train_x.shape[1]))
                if 'dropout' in suit: model.add(Dropout(suit['dropout']))
                for n in suit['base model'][1:]:
                    model.add(Dense(units=n, activation='relu'))
                    if 'dropout' in suit: model.add(Dropout(suit['dropout']))
            model.add(Dense(units=num_classes, activation='softmax'))
    else:
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=orig_train_x.shape[1:]))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

    optimizer = 'adam' #keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    # Run model training
    training_iterations = len(suit['epochs'])
    for iter in range(training_iterations):
        lesson_num = iter if iter < len(suit['curric']) else len(suit['curric']) - 1
        print('-------------------------------------------------------------------------------')

        # Create random subset lesson data on the fly
        lesson = None
        test_data = None
        if 'subset size' in suit:
            if iter == training_iterations - 1:
                lesson = list(range(num_classes))
                train_x, train_y, test_x, test_y = orig_train_x, orig_train_y, orig_test_x, orig_test_y
                train_y = to_categorical(train_y, num_classes=num_classes)
                test_y = to_categorical(test_y, num_classes=num_classes)
                test_data = (test_x, test_y)
            else:
                lesson = random.sample(class_pool, suit['subset size'])

                suit['distinct classes'].append(len(set(lesson)))

                train_x, train_y = suit['train data gen'].next_block(lesson, suit['batch size'] * suit['batches per class per epoch'])
                test_data = suit['test data gen'].next_block(lesson, suit['batch size'])

            print('Lesson: ' + suit['name'] + ', ' + suit['curric'][lesson_num][0] + ', ' + str(lesson))
        else:
            lesson = suit['curric'][lesson_num][1]
            train_x = suit['curric_train_x'][lesson_num]
            train_y = suit['curric_train_y'][lesson_num]
            test_x = suit['curric_test_x'][lesson_num]
            test_y = suit['curric_test_y'][lesson_num]
            test_data = (test_x, test_y)

            print('Lesson: ' + suit['name'] + ', ' + suit['curric'][lesson_num][0])

        # Freeze parameters assoicated to output units not being trained in the current lesson
        weights_into = None; biases = None
        if 'freeze' in suit and suit['freeze']:
            weights_into, biases = freeze_params_outside_lesson(model, num_classes, lesson)

        # Fit the model
        epochs = suit['epochs'][iter]
        history = model.fit(train_x, train_y, validation_data=test_data, epochs=epochs, batch_size=suit['batch size'], verbose=2, shuffle=False)

        # Unfreeze parameters
        if 'freeze' in suit and suit['freeze']:
            set_params_outside_lesson(model, num_classes, lesson, weights_into, biases)

        # Update class pool
        if len(lesson) < num_classes:
            max_test_acc = np.max(history.history['val_acc'])

            prev_accuracies.append(max_test_acc)
            if (len(prev_accuracies) > 5):
                prev_accuracies.pop(0)

            class_accuracies_avg = np.mean(prev_accuracies)

            print('class avg acc %.2f' % (100 * class_accuracies_avg))

            if max_test_acc < class_accuracies_avg:
                class_pool += lesson[:suit['class pool update size']]

            class_freq = np.zeros(num_classes)
            for i in range(num_classes):
                class_freq[i] = sum([x == i for x in class_pool]) / len(class_pool)
                suit['class freq'][i].append(class_freq[i])

        # Print accuracy on entire test set
        if iter % 10 == 0:
            pred = model.predict(orig_test_x)
            pred = np.argmax(pred, axis=1)
            print('Test acc on entire test set: ' + str(sum(pred.flatten() == orig_test_y.flatten()) / orig_test_y.shape[0]))

        append_history(suit, history, epochs, train_x.shape[0], lesson == list(range(num_classes)))

    # Help keras garbage collect
    del model
    gc.collect()
    keras.backend.clear_session()

    suit['runtime'] = time.time() - start_time
    print('Time: %.2f' % (suit['runtime']))

# Print accuracies for each suit
for suit in test_suits:
   print('suit: ' + suit['name'] + ', best training acc: ' + str(suit['hist']['best_acc']))
   print('suit: ' + suit['name'] + ',  best testing acc: ' + str(suit['hist']['best_val_acc']))

# Print average accuracies for each suit
print('\n--------------- training averages ---------------')
seen_curricula = []
for suit in test_suits:
    if suit['name'] in seen_curricula or suit['repeat'] is None: continue
    seen_curricula.append(suit['name'])
    print('%-50s: %5.2f' % (suit['name'], 100 * np.average([s['hist']['best_acc'] for s in test_suits if s['name'].startswith(suit['name'])])))
print('\n--------------- testing averages ---------------')
seen_curricula = []
for suit in test_suits:
    if suit['name'] in seen_curricula or suit['repeat'] is None: continue
    seen_curricula.append(suit['name'])
    print('%-50s: %5.2f' % (suit['name'], 100 * np.average([s['hist']['best_val_acc'] for s in test_suits if s['name'].startswith(suit['name'])])))

print('\n------------------------------')
seen_curricula = []
for suit in test_suits:
    if suit['name'] in seen_curricula or suit['repeat'] is None: continue
    seen_curricula.append(suit['name'])
    for i in [5, 10, 15]:
        l = [s['hist']['val acc %de' % (i)] for s in test_suits if s['name'].startswith(suit['name'])]
        if len(l) > 1:
            print('%-50s, avg  %2de: %5.2f' % (suit['name'], i, 100 * np.average(l)))
            print('%-50s, best %2de: %5.2f' % (suit['name'], i, 100 * np.max(l)))
            print()

# Plot accuracies for each suit
for a in ['acc', 'val_acc']:
    fig = plt.gcf()
    fig.set_size_inches(9.5, 6)
    for suit in test_suits:
        plt.plot(suit['hist'][a], label=suit['name'])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    # plt.ylim(bottom=0.7)
    if a == 'acc':
        plt.title('Training accuracy')
    else:
        plt.title('Validation accuracy')

    plt.legend()
    plt.grid()
    plt.show()

# Plot class frequencies
for suit in test_suits:
    fig = plt.gcf()
    fig.set_size_inches(9.5, 15)
    for i in range(num_classes):
        plt.plot(suit['class freq'][i])  # , label=str(i))
    plt.title(suit['name'])

    # plt.legend()
    plt.grid()
    plt.show()

    fig = plt.gcf()
    fig.set_size_inches(9.5, 3)
    plt.plot(suit['distinct classes'])
    plt.title(suit['name'])

    plt.grid()
    plt.show()
