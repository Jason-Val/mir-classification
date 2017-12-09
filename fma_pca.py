import sys
sys.path.insert(0, './data/fma-data')
import load_data
import numpy as np
from sklearn import decomposition
from sklearn import preprocessing
import random

TRACKS = None

X_DATA = None
Y_DATA = None
X_TEST = None
Y_TEST = None
X_VAL = None
Y_VAL = None

X_BATCH = None
Y_BATCH = None

FEATURE_SIZE = None


def init_debug(n_genres=163, subset='small', pca_on=True, n_components=3):
    global TRACKS
    global X_DATA
    global Y_DATA
    global X_TEST
    global Y_TEST
    global X_VAL
    global Y_VAL
    
    pca = decomposition.PCA(n_components=n_components)
    
    # these are private helper functions
    
    def do_pca(matrix):
        nonlocal pca
        return pca.fit_transform(preprocessing.normalize(np.transpose(matrix)))
    
    def add_features(x, features):
        nonlocal pca_on
        if (pca_on):
            if (len(x) == 0):
                x = np.hstack(do_pca(to_matrix(features)))
            else:
                x = np.vstack([x, np.hstack(do_pca(to_matrix(features)))])
        else:
            if (len(x) == 0):
                x = np.hstack(to_matrix(features))
            else:
                x = np.vstack([x, np.hstack(to_matrix(features))])
        return x
    
    genres = load_data.get_n_genres(n_genres-1)
    
    def add_genres(y, y_genres):
        nonlocal n_genres
        nonlocal genres

        labels = [0 for x in range(n_genres)]
        
        if len(y_genres) == 0:
            labels[-1] = 1.

        for genre in y_genres:
            if genre in genres:         # the genre is the top n genres
                labels[genres.index(genre)] = 1.
        if len(y) == 0:
            y = labels
        else:
            y = np.vstack([y, labels])
        return y
    
    TRACKS = load_data.get_tracks(subset)

    x = []
    y = []
    
    x_t = []
    y_t = []

    x_v = []
    y_v = []
    
    # the difference in # of tracks within genre(s), and # of tracks unlabeled
    # this keeps the number of classified and unclassified tracks roughly equal
    diff_data = 0
    diff_test = 0
    diff_val = 0

    for track in TRACKS:
        if track.split == 'training':
            if set(track.genres).intersection(set(genres)) != set():
                x = add_features(x, track.features)
                y = add_genres(y, track.genres)
                diff_data += 1
            elif diff_data > -40:
                x = add_features(x, track.features)
                y = add_genres(y, [])
                diff_data -= 1
        elif track.split == 'test':
            if set(track.genres).intersection(set(genres)) != set():
                x_t = add_features(x_t, track.features)
                y_t = add_genres(y_t, track.features)
                diff_test += 1
            elif diff_test > -40:
                x_t = add_features(x_t, track.features)
                y_t = add_genres(y_t, [])
                diff_data -= 1
        else:                                           # validation
            if set(track.genres).intersection(set(genres)) != set():
                x_v = add_features(x_v, track.features)
                y_v = add_genres(y_v, track.genres)
                diff_val += 1
            elif diff_val > -40:
                x_v = add_features(x_v, track.features)
                y_v = add_genres(y_v, [])
                diff_val -= 1
    
    Y_DATA = y
    X_DATA = x

    Y_TEST = y_t
    X_TEST = x_t

    Y_VAL = y_v
    X_VAL = x_v

def init():
    global TRACKS
    global X_DATA
    global Y_DATA
    global X_TEST
    global Y_TEST
    global next_pca_batch

    TRACKS = load_data.get_small()
    pca = decomposition.PCA(n_components=1)
    
    x = []
    y = []
    
    x_t = []
    y_t = []
    
    n_genres = num_genres()
    
    # applies pca to a matrix. Then flattens it
    def pca_flatten (matrix):
        nonlocal pca
        return np.hstack(pca.fit_transform(matrix))

    # these are private helper functions
    def add_features(x, features):
        if (len(x) == 0):
                x = np.array(pca_flatten(to_matrix(features)))
        else:
            x = np.vstack([x, pca_flatten(to_matrix(features))])
        return x
        
    def add_genres(y, genres):
        nonlocal n_genres
        y_curr = [0. for x in range(n_genres)]
        for genre in genres:
            y_curr[genre] = 1.
        if (len(y) == 0):
            y = np.array([y_curr])
        else:
            y = np.concatenate((y, np.array([y_curr])))
        return y

    # fill the global variables X_..., Y_...
    for track in TRACKS:
        if track.split == 'training':
            x = add_features(x, list(track.features))
            y = add_genres(y, track.genres)
        else:
            x_t = add_features(x_t, list(track.features))
            y_t = add_genres(y_t, track.genres)
    
    Y_DATA = y
    X_DATA = x

    Y_TEST = y_t
    X_TEST = x_t
    
    # return the next batch of size (size)
    # designed to conform to the mnist standard -- it just works, no generator involved
    # enclosed to protect essential nonlocal helper variables
    i = 0
    prev = 0
    def get_pca_batch(size):
        nonlocal i
        nonlocal prev
        prev = i
        i += size
        return (X_DATA[prev:i], Y_DATA[prev:i])
    # adds our enclosed function to the global namespace
    #next_pca_batch = get_pca_batch

# given a list of features, returns the length of the biggest feature
def max_length(features):
    length = 0
    for feature in features:
        curr = len(feature.data[0])
        if length < curr:
            length = curr
    return length

def feature_size():
    global FEATURE_SIZE
    if FEATURE_SIZE == None:
        global TRACKS
        return max_length(TRACKS[0].features)
    else:
        return FEATURE_SIZE

def fill_zeros(data, length):
    l1 = len(data)
    data += [0.0 for x in range (0, length-l1)]
    return data

# given a list of features
def to_matrix(features):
    matrix = []
    global FEATURE_SIZE
    if FEATURE_SIZE != None:
        size = FEATURE_SIZE
    else:
        size = max_length(features)
    for feature in features:
        for stat in feature.data:
            matrix.append(fill_zeros(stat, size))
    matrix = np.array(matrix)
    return matrix

def get_mapping():
    global TRACKS
    pca = decomposition.PCA(num_components = 1)
    mapping = {}
    for track in TRACKS:
        data = pca.fit_transform(to_matrix(track.features))
        mapping[track.id] = (data, track.genres)
    return mapping

def num_genres():
    genres = load_data.load_genres()[1]
    l = list(genres.keys())
    return len(l)

def init_batch():
    global X_BATCH
    global Y_BATCH
    X_BATCH = np.copy(X_DATA)
    Y_BATCH = np.copy(Y_DATA)

def get_batch(size):
    global X_BATCH
    global Y_BATCH

    if len(X_BATCH) == 0:
        return ([], [])

    batch_x = []
    batch_y = []
    
    for i in range(size):

        if len(X_BATCH) == 0:
            break
        
        r=random.randint(0, len(X_BATCH)-1)
        
        if len(batch_x) == 0:
            batch_x = X_BATCH[r]
            batch_y = Y_BATCH[r]
        else:
            batch_x = np.vstack([batch_x, X_BATCH[r]])
            batch_y = np.vstack([batch_y, Y_BATCH[r]])
        
        X_BATCH = np.delete(X_BATCH, r, 0)
        Y_BATCH = np.delete(Y_BATCH, r, 0)

    return (X_BATCH, Y_BATCH)

def x_train():
    return X_DATA

def y_train():
    return Y_DATA

def x_test():
    return X_TEST

def y_test():
    return Y_TEST

def x_val():
    return X_VAL

def y_val():
    return Y_VAL

load_data.init_loader('./data/fma-data')


if __name__ == 'main':
    
    import matplotlib.pyplot as plt

    #init_debug(n_components=20)

        


    import ply_export
    
    n_genres = 3
    
    pca = decomposition.PCA(n_components=3)
    
    # these are private helper functions
    
    def do_pca(matrix):
        nonlocal pca
        return pca.fit_transform(preprocessing.normalize(np.transpose(matrix)))
    
    def add_features(x, features):
        nonlocal pca_on
        if (pca_on):
            if (len(x) == 0):
                x = np.hstack(do_pca(to_matrix(features)))
            else:
                x = np.vstack([x, np.hstack(do_pca(to_matrix(features)))])
        else:
            if (len(x) == 0):
                x = np.hstack(to_matrix(features))
            else:
                x = np.vstack([x, np.hstack(to_matrix(features))])
        return x
    
    genres = load_data.get_n_genres(n_genres-1)
    
    def add_genres(y, y_genres):
        nonlocal n_genres
        nonlocal genres

        labels = [0 for x in range(n_genres)]
        
        if len(y_genres) == 0:
            labels[-1] = 1.

        for genre in y_genres:
            if genre in genres:         # the genre is the top n genres
                labels[genres.index(genre)] = 1.
        if len(y) == 0:
            y = labels
        else:
            y = np.vstack([y, labels])
        return y
    

    TRACKS = load_data.get_tracks('small')

    x = []
    y = []
    
    # the difference in # of tracks within genre(s), and # of tracks unlabeled
    # this keeps the number of classified and unclassified tracks roughly equal
    diff_data = 0
    diff_test = 0
    diff_val = 0

    num_samples = 100

    genres = [ [] for x in range(n_genres)]

    

    for track in TRACKS:
        #...
        for genre in track.genres:
            
        if track.split == 'training':
            if set(track.genres).intersection(set(genres)) != set():
                x = add_features(x, track.features)
                y = add_genres(y, track.genres)
                diff_data += 1
            elif diff_data > -40:
                x = add_features(x, track.features)
                y = add_genres(y, [])
                diff_data -= 1
        elif track.split == 'test':
            if set(track.genres).intersection(set(genres)) != set():
                x_t = add_features(x_t, track.features)
                y_t = add_genres(y_t, track.features)
                diff_test += 1
            elif diff_test > -40:
                x_t = add_features(x_t, track.features)
                y_t = add_genres(y_t, [])
                diff_data -= 1
        else:                                           # validation
            if set(track.genres).intersection(set(genres)) != set():
                x_v = add_features(x_v, track.features)
                y_v = add_genres(y_v, track.genres)
                diff_val += 1
            elif diff_val > -40:
                x_v = add_features(x_v, track.features)
                y_v = add_genres(y_v, [])
                diff_val -= 1
    
    Y_DATA = y
    X_DATA = x

    Y_TEST = y_t
    X_TEST = x_t

    Y_VAL = y_v
    X_VAL = x_v






