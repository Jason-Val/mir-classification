
import csv
import numpy as np

DEBUG = False
PATH = '.'

GENRES = None
TRACKS = None
FEATURES = None
genre_to_index = None
index_to_genre = None

class genre:
    def __init__(self, id, count, parent, title, top_level):
        self.id = id
        self.count = count
        self.parent = parent
        self.title = title
        self.top_level = top_level
        self.children = []
    def add_child(self, child):
        self.children.append(child)
    def __str__(self):
        return "{}: {} || {}".format(self.id, self.title, self.parent)

class track:
    def __init__(self, id, genres, title, bitrate, split, subset):
        self.id = id
        self.genres = genres
        self.title = title
        self.bitrate = bitrate
        self.split = split          # training or testing
        self.subset = subset        # small or ...
        self.features = None        # [f1, f2, ...]
    def __str__(self):
        return "{}: {}  ||  {}".format(self.id, self.title, self.genres)

# data  = [1, 2, 3, ...] [...] [...]
# stats = ['mean'      ] [' '] [' ']
class Feature:
    def __init__(self, feature, num_stats):
        self.title = feature
        self.stats = []
        self.data = []
    def add_stat(self, stat):
        self.data.append([])
        self.stats.append(stat)

def init_loader(path):
    global PATH
    PATH = path
    global genre_to_index
    global index_to_genre
    genre_to_index, index_to_genre = init_genre_index()

# returns a list of all tracks in the small dataset
# each track has associated features
# each track has its genre list altered, storing only the shifted ids of each genre
# they are shifted so that the genre ids will be consecutive; this makes the final softmax
# layer of the neural net smaller (versus having a spot for genres 23-25, which don't exist)
def get_tracks(subset):
    result = []
    tracks = load_tracks()
    features = load_features()
    for track_id in features:
        track = tracks[track_id]
        feature = features[track_id]
        if track.subset == subset:
            track.features = feature
            for i in range(0, len(track.genres)):
                track.genres[i] = genre_to_index(track.genres[i])
            result.append(track)
    return result

def get_n_genres(n_genres):
    global GENRES
    if GENRES == None:
        GENRES = load_genres()[1]
    
    genres = list(GENRES.values())
    genres = sorted(genres, reverse=True, key=(lambda x: x.count))

    global genre_to_index

    for i in range(n_genres):
        genres[i] = genre_to_index(genres[i].id)

    return genres[:n_genres]

def init_genre_index():
    global GENRES
    if GENRES == None:
        GENRES = load_genres()[1]

    # [1, 2, 4, 7] ==> [0, 1, 2, 3]
    # {1: 0, 2: 1,}    {0: 1, 1: 2,}    
    map_to_genre = sorted(list(GENRES.keys()))       # this can just be a list
    map_to_index = {}

    for i in range (0, len(map_to_genre)):
        map_to_genre[i] = map_to_genre[i]
        map_to_index[map_to_genre[i]] = i

    def genre_to_index(genre):
        return map_to_index[genre]
    def index_to_genre(index):
        return map_to_genre[index]

    return (genre_to_index, index_to_genre)


def load_tracks():
    mapping = {}                                    # id (track) -> track (class)
    with open('{}/fma_metadata/tracks.csv'.format(PATH)) as f:
        line = f.readline().strip().split(',')
        track_start = line.index('track')           # where the track info begins in the list
        track_info = f.readline().strip().split(',')[track_start:]      #all of the track categores
        split_i = track_start-2                     # the index of split (training/testing)
        subset_i = track_start-1                    # the index of subset (small, big, ...)
        id_i = 0                                    # the track id is always the first element
        genres_i = track_info.index('genres') + track_start      # add track_start to get the index in the total list, not just the track_info list
        title_i = track_info.index('title') + track_start
        bitrate_i = track_info.index('bit_rate') + track_start

        f.readline()

        l = csv.reader(f)

        for info in l:
            genres = eval(info[genres_i])
            mapping[int(info[id_i])] = track(int(info[id_i]), genres, info[title_i], int(info[bitrate_i]), info[split_i], info[subset_i])

    return mapping

def load_genres():
    top_genres = {}
    mapping = {}                                        # id -> genre (class)
    global PATH
    with open('{}/fma_metadata/genres.csv'.format(PATH)) as f:
        top_level_count = 16
        f.readline()
        for line in f:
            genre_data = line.strip().split(',')
            new_genre = genre(int(genre_data[0]), int(genre_data[1]), int(genre_data[2]), genre_data[3], int(genre_data[4]))
            if genre_data[0] == genre_data[4]:
                top_genres[genre_data[0]] = new_genre
            mapping[int(genre_data[0])] = new_genre
        
        for k,node in mapping.items():
            if node.parent == 0:
                node.parent == None
            else:
                node.parent = mapping[node.parent]
                node.parent.add_child(node)
            node.top_level = mapping[node.top_level]
    global GENRES
    GENRES = mapping
    return (top_genres, mapping)

def load_features():
    mapping = {}                # track id -> [features (class)]
    with open('{}/fma_metadata/features.csv'.format(PATH)) as f:
        features = f.readline().strip().split(',')
        statistics = f.readline().strip().split(',')
        numbers = f.readline().strip().split(',')
        f.readline()
        num_stats = len(set(statistics)) -1
        
        lines = csv.reader(f)
        i = 0
        curr_feature = ''
        curr_stat = ''
        for line in lines:
            id = -1
            stat_i = -1         # the index of curr_stat in our Feature's data list
            feature_i = -1       # the index of curr_feature in our mapping
            for feature, stat, index, data in zip(features, statistics, numbers, line):
                if feature == 'feature':    #this is the first column, and contains only the track id
                    mapping[int(data)] = []
                    id = int(data)
                else:
                    if curr_feature != feature:
                        curr_feature = feature
                        stat_i = -1
                        feature_i += 1
                        mapping[id].append(Feature(feature, num_stats))
                    if curr_stat != stat:
                        curr_stat = stat
                        stat_i += 1
                        mapping[id][feature_i].add_stat(stat)
                    mapping[id][feature_i].data[stat_i].append(float(data))
            #if i > 500:
                #break;
            #i += 1
    return mapping

