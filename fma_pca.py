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

X_BATCH = []
Y_BATCH = []

FEATURE_SIZE = None

get_batch = None


def init(n_genres=163, subset='small', pca_on=True, n_components=3, reuse=False):
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
                x = np.array([np.hstack(do_pca(to_matrix(features)))])
            else:
                x = np.vstack([x, np.hstack(do_pca(to_matrix(features)))])
        else:
            if (len(x) == 0):
                x = np.array([np.hstack(to_matrix(features))])
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
    
    if not reuse or TRACKS == None:
        TRACKS = load_data.get_tracks(subset)

    x = []
    y = []
    
    x_t = []
    y_t = []

    x_v = []
    y_v = []
    
    # count of how many tracks are added not within specified genres
    others = 0
    others_t = 0
    others_v = 0

    for track in TRACKS:
        if track.split == 'training':
            if set(track.genres).intersection(set(genres)) != set():
                x = add_features(x, track.features)
                y = add_genres(y, track.genres)
            elif len(x)/n_genres > others:
                x = add_features(x, track.features)
                y = add_genres(y, [])
                others += 1
        elif track.split == 'test':
            if set(track.genres).intersection(set(genres)) != set():
                x_t = add_features(x_t, track.features)
                y_t = add_genres(y_t, track.genres)
            elif len(x_t)/n_genres > others_t:
                x_t = add_features(x_t, track.features)
                y_t = add_genres(y_t, [])
                others_t += 1
        else:                                           # validation
            if set(track.genres).intersection(set(genres)) != set():
                x_v = add_features(x_v, track.features)
                y_v = add_genres(y_v, track.genres)
            elif len(x_t)/n_genres > others_v:
                x_v = add_features(x_v, track.features)
                y_v = add_genres(y_v, [])
                others_v += 1
    
    Y_DATA = y
    X_DATA = x

    Y_TEST = y_t
    X_TEST = x_t

    Y_VAL = y_v
    X_VAL = x_v

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

# initialize batch creation; run this every epoch
def init_batch():
    global X_BATCH
    global Y_BATCH
    global get_batch
    
    if len(X_BATCH) == 0:     
        X_BATCH = np.copy(X_DATA)
        Y_BATCH = np.copy(Y_DATA)
    rng_state = np.random.get_state()
    np.random.shuffle(X_BATCH)
    np.random.set_state(rng_state)
    np.random.shuffle(Y_BATCH)
    
    # define get_batch. enclosed to protect helper variable bookmark
    bookmark = 0
    def batch(size):
        nonlocal bookmark
        result = (X_BATCH[bookmark:bookmark+size], Y_BATCH[bookmark:bookmark+size])
        bookmark += size
        return result

    get_batch = batch

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

def n_dim():
    return len(X_DATA[0])

load_data.init_loader('./data/fma-data')

def show_pca_histogram():
    import matplotlib.pyplot as plt
    global TRACKS
    pca = decomposition.PCA()
    
    total = []

    i = 0

    for track in TRACKS:
        matrix = to_matrix(track.features)
        pca.fit(preprocessing.normalize(np.transpose(matrix)))
        ratio = pca.explained_variance_ratio_
        temp = 0
        for i in range(len(ratio)):
            ratio[i] += temp
            temp = ratio[i]
        if len(total) == 0:
            total = ratio
        else:
            total = total + ratio
        i += 1
    result = total/len(TRACKS)
    print(result)

    plt.plot(result)
    plt.show()

def get_random_average(samples, vertices):
    import random

    avg = []
    for sample in range(samples):
        r = random.randint(0,len(vertices)-1)
        
        if len(avg) == 0:
            avg = [vertices[r]]
        else:
            avg = np.vstack([avg, [vertices[r]]])
    
    return np.mean(avg, axis=0)

if __name__ == '__main__':
    
    if len(sys.argv) > 1 and sys.argv[1] == 'debug':
        pass
    else:
        TRACKS = load_data.get_tracks('small')

        import ply_export
        
        n_genres = 3
        
        genres = load_data.get_n_genres(n_genres)

        pca = decomposition.PCA(n_genres)
        
        # these are private helper functions
        
        def do_pca(matrix, pca):
            return pca.fit_transform(preprocessing.normalize(np.transpose(matrix)))

        #vertices = do_pca(to_matrix(TRACKS[0].features))

        #ply_export.write_ply(vertices, 'test1')
        
        vertices = []
        genre_vertices = [[] for x in range(n_genres)]
        for track in TRACKS:
            pca_matrix = [do_pca(to_matrix(track.features), pca)]
            for genre in track.genres:
                if genre in genres:
                    index = genres.index(genre)
                    if len(genre_vertices[index]) == 0:
                        genre_vertices[index] = pca_matrix
                    else:
                        genre_vertices[index] = np.vstack([genre_vertices[index], pca_matrix])
            if len(vertices) == 0:
                vertices = pca_matrix
            else:
                vertices = np.vstack([vertices, pca_matrix])
        
        total_average = np.mean(vertices, axis=0)
        
        #ply_export.write_ply(total_average, 'total')

        genre_avg = [0 for x in range(len(genre_vertices))]

        for i in range(len(genre_vertices)):
            genre_avg[i] = np.mean(genre_vertices[i], axis=0)
        
        for i in range(len(genre_avg)):
            ply_export.write_ply(genre_avg[i], 'genre{}'.format(genres[i]))

        #for i in range(10):
            #ply_export.write_ply(get_random_average(100, vertices), 'test{}'.format(i))




