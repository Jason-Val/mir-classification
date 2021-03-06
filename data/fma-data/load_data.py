import zipfile as zf
import csv
import numpy as np

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
        self.features = []        # [f1, f2, ...]
        self.mel = []
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

"""
Initializes the data loader with the correct path
Also initializes genre_to_index and index_to_genre
"""
def init_loader(path):
    global PATH
    PATH = path
    global genre_to_index
    global index_to_genre
    genre_to_index, index_to_genre = init_genre_index()

"""
The genre ids are non-consecutive
This gives each genre a consecutive id, and translates between the two
"""
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

"""
Returns a list of all tracks in a given dataset
each track has either (or both) of the melspectrogram or features.csv data
each track has its genre list altered, storing only the consecutive ids of each genre
"""
def get_tracks(subset, mel=True, pca=False):
    result = []
    tracks = load_tracks()
    features = None
    mel_map = None
    if pca:
        features = load_features()
    if mel:
        mel_map = load_mel()
    for track_id in tracks.keys():
        track = tracks[track_id]
        if track.subset == subset:
            if pca:
                track.features = features[track_id]
            if mel:
                if track_id in mel_map:         # detect "holes" in the data, and skip these tracks
                    track.mel = mel_map[track_id]
                else:
                    continue
            for i in range(len(track.genres)):
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


def load_tracks():
    print("Reading tracks from file...")
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
    print("Loading genres...")
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


"""
Loads the feautres from features.csv
"""
def load_features():
    print("Loading fma features...")
    mapping = {}                # track id -> [features (class)]
    with open('{}/fma_metadata/features.csv'.format(PATH)) as f:
        features = f.readline().strip().split(',')
        statistics = f.readline().strip().split(',')
        numbers = f.readline().strip().split(',')
        f.readline()
        num_stats = len(set(statistics)) -1
        
        lines = csv.reader(f)
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
    return mapping

# returns a mapping from track id to melspectrogram, where the melspectrogram
# has shape [128, 128, 1].
# The third dimension is because tensorflow expects a color channel; we use greyscale
def load_mel():
    print("Loading melspectrograms...")
    mapping = {}                # track id
    i = 1
    curr_track = 0
    freq_i = 0
    count = 0
    with zf.ZipFile('{}/fma_metadata/melspectrogram_medium.zip'.format(PATH)) as mel_zip:
        with mel_zip.open(mel_zip.namelist()[0]) as mel_file:
            for line in mel_file.readlines():
                line = line.decode('utf-8').split(',')
                if line[1] == '':
                    count += 1
                    curr_track = int(line[0])
                    print('Loading melspectrogram #{} of 24981'.format(count), sep=' ', end='\r', flush=True)
                    mapping[curr_track] = np.array([[ [1.] for sample in range(128)] for freq in range(128)])
                    freq_i = 0
                else:
                    index = 0
                    for sample in line:
                        mapping[curr_track][freq_i][index][0] = sample
                        index += 1
                    freq_i += 1
                i += 1
    print('')
    return mapping
