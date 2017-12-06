import sys
sys.path.insert(0, './data/fma-data')
import load_data
import numpy as np
from sklearn import decomposition

TRACKS = None
X_PCA_DATA = None
Y_PCA_DATA = None
next_pca_batch = None

def init():
    global TRACKS
    global X_PCA_DATA
    global Y_PCA_DATA
    global next_pca_batch
    TRACKS = load_data.get_small_training()
    pca = decomposition.PCA(n_components=1)
    x = []
    y = []
    
    n_genres = num_genres()

    # define X_PCA_DATA
    for track in TRACKS:
        if (len(x) == 0):
            x = np.array([list(map(lambda e: e[0], pca.fit_transform(to_matrix(track.features))))])
        else:
            next = np.array([list(map(lambda e: e[0], pca.fit_transform(to_matrix(track.features))))])
            x = np.concatenate((x, next))
        
        y_curr = [0. for x in range(n_genres)]
        for genre in track.genres:
            y_curr[genre] = 1./len(track.genres)
        if (len(y) == 0):
            y = np.array([y_curr])
        else:
            y = np.concatenate((y, np.array([y_curr])))
    Y_PCA_DATA = y
    X_PCA_DATA = x
    
    i = 0
    prev = 0
    def get_pca_batch(size):
        nonlocal i
        nonlocal prev
        prev = i
        i += size
        return (X_PCA_DATA[prev:i], Y_PCA_DATA[prev:i])
        
    next_pca_batch = get_batch
    

# given a list of features, returns the length of the biggest feature
def max_length(features):
    length = 0
    for feature in features:
        curr = len(feature.data[0])
        if length < curr:
            length = curr
    return length

def fill_zeros(data, length):
    l1 = len(data)
    data += [0.0 for x in range (0, length-l1)]
    return data

# given a list of features
def to_matrix(features):
    matrix = []
    size = max_length(features)
    for feature in features:
        data = feature.data
        for stat in data:
            matrix.append(fill_zeros(stat, size))
    matrix = np.transpose(np.array(matrix))
    return matrix

def relative_lengths(matrix):
    matrixt = np.transpose(matrix)
    lengths = []
    for row in matrixt:
        total = 0
        for num in row:
            total += num*num
        lengths.append(total)
    total = sum(lengths)
    for i in range(0, len(lengths)):
        lengths[i] = lengths[i]/total
    return lengths

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
    

load_data.init_loader('./data/fma-data')

if __name__ == 'main':
    #load_data.init_loader('./data/fma-data')

    tracks = load_data.get_small_training()

    pca = decomposition.PCA()

    average_ratio = np.array([0.0 for x in range (0, 20)])
    i = 0

    print('test!!!')

    for track in tracks:
        l = relative_lengths(pca.fit_transform(to_matrix(track.features)))
        average_ratio += l
        #print(l[0])
        #print(average_ratio[0])
        #if i > 7:
            #print('breaking!')
            #break
        #i += 1
    average_ratio = average_ratio / len(tracks)
    print(average_ratio)






