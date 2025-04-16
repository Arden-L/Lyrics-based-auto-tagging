import os
import sys
import re
import string
import pandas as pd
import numpy as np
import nltk
import matplotlib
matplotlib.use('Agg')  # Prevents GUI from popping up
import matplotlib.pyplot as plt
import seaborn as sns
import pronouncing
import textstat
import torch
import ast
import joblib
import json

from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, hamming_loss, f1_score, jaccard_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler, StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from gensim.models import Word2Vec, FastText, KeyedVectors
from transformers import BertTokenizer, BertModel
from textblob import TextBlob
from tqdm import tqdm


### CONFIGURATION ###

# --- Argument Validation and Error Handling ---
if len(sys.argv) < 2:
    print("Error: Please provide the path to the Music4All dataset folder.")
    print("Usage: python lyric_based_classification.py <path_to_music4all>")
    sys.exit(1)
DATA_DIR = sys.argv[1]
if not os.path.isdir(DATA_DIR):
    print(f"Error: The specified path '{DATA_DIR}' is not a valid directory.")
    sys.exit(1)

# --- Constants ---
TEST_SIZE = 0.2         # TESTING SPLIT - change if desired but check for cache mismatches with saved data files!
SUBSET_SIZE = None     # SIZE OF DATA FOR ANALYSIS - change if desired but check for cache mismatches with saved data files!

LYRICS_SUBDIR = "lyrics"
GENRES_FILE = "id_genres.csv"
METADATA_FILE = "id_metadata.csv"
LANG_FILE = "id_lang.csv"
TAGS_FILE = "id_tags.csv"
LYRICS_ID_COLUMN = "id"
TEXT_COLUMN = "lyrics"
GENRE_COLUMN = "genres"
TAG_COLUMN = 'tags'
RANDOM_STATE = 42
SUBGENRE_TO_BROAD = {
    'russian pop': 'pop',
    'progressive trance': 'electronic',
    'harpsichord': 'classical',
    'dance-punk': 'rock',
    'synthpop': 'pop',
    'mariachi': 'latin',
    'tejano': 'latin',
    'uk drill': 'hip hop',
    'brutal death metal': 'metal',
    'kizomba': 'world',
    'chillstep': 'electronic',
    'minimal': 'electronic',
    'gypsy jazz': 'jazz',
    'pub rock': 'rock',
    'tropical house': 'electronic',
    'modern rock': 'rock',
    'jungle': 'electronic',
    'didgeridoo': 'world',
    'christian hard rock': 'rock',
    'white noise': 'experimental',
    'thrash metal': 'metal',
    'drone': 'experimental',
    'album rock': 'rock',
    'post-metal': 'metal',
    'finnish black metal': 'metal',
    'experimental': 'experimental',
    'uk house': 'electronic',
    'uk garage': 'electronic',
    'cyberpunk': 'electronic',
    'easy listening': 'pop',
    'hip house': 'electronic',
    'electroclash': 'electronic',
    'trap music': 'electronic',
    'cartoon': 'soundtrack',
    'french hip hop': 'hip hop',
    'riot grrrl': 'punk',
    'industrial techno': 'electronic',
    'piedmont blues': 'blues',
    'music box': 'classical',
    'alternative pop rock': 'rock',
    'bounce': 'hip hop',
    'contemporary folk': 'folk',
    'anime': 'soundtrack',
    'swedish metal': 'metal',
    'disco': 'pop',
    'dc hardcore': 'punk',
    'cabaret': 'theatrical',
    'reggae': 'reggae',
    'gospel blues': 'blues',
    'post-punk': 'rock',
    'italo house': 'electronic',
    'industrial metal': 'metal',
    'electro swing': 'electronic',
    'post-rock': 'rock',
    'easycore': 'punk',
    'highlife': 'world',
    'bass house': 'electronic',
    'country': 'country',
    'celtic metal': 'metal',
    'boy band': 'pop',
    'latin jazz': 'latin',
    'metal': 'metal',
    'mod revival': 'rock',
    'surf music': 'rock',
    'gothic rock': 'rock',
    'skate punk': 'punk',
    'asmr': 'experimental',
    'melodic hard rock': 'rock',
    'chanson': 'folk',
    'new weird america': 'folk',
    'bachata': 'latin',
    'bubblegum pop': 'pop',
    'j-rock': 'rock',
    'rawstyle': 'electronic',
    'trance': 'electronic',
    'deep house': 'electronic',
    'steampunk': 'experimental',
    'calypso': 'world',
    'metalcore': 'metal',
    'sleep': 'ambient',
    'filthstep': 'electronic',
    'emo': 'punk',
    'choral': 'classical',
    'accordion': 'folk',
    'noise punk': 'punk',
    'liquid funk': 'funk',
    'brutal deathcore': 'metal',
    'yacht rock': 'rock',
    'opera': 'classical',
    'disco house': 'electronic',
    'indonesian indie': 'world',
    'neurofunk': 'funk',
    'go-go': 'funk',
    'medieval folk': 'folk',
    'techno': 'electronic',
    'discofox': 'electronic',
    'metallic hardcore': 'metal',
    'rap rock': 'rock',
    'tribal house': 'electronic',
    'preverb': 'experimental',
    'sleaze rock': 'rock',
    'garage punk': 'punk',
    'atmosphere': 'ambient',
    'edm': 'electronic',
    'talent show': 'pop',
    'neue deutsche welle': 'electronic',
    'jumpstyle': 'electronic',
    'industrial black metal': 'metal',
    'avant-garde black metal': 'metal',
    'acoustic pop': 'pop',
    'queercore': 'punk',
    'pixie': 'pop',
    'complextro': 'electronic',
    'chamber pop': 'pop',
    'jangle rock': 'rock',
    'funk metal': 'metal',
    'dark cabaret': 'theatrical',
    'southern hip hop': 'hip hop',
    'visual kei': 'rock',
    'deep funk': 'funk',
    'electric blues': 'blues',
    'grindcore': 'metal',
    'rhythm and blues': 'blues',
    'dub metal': 'metal',
    'jazz blues': 'blues',
    'aggrotech': 'electronic',
    'technical deathcore': 'metal',
    'post-black metal': 'metal',
    'electro house': 'electronic',
    'alternative country': 'country',
    'gothenburg metal': 'metal',
    'blues': 'blues',
    'acid jazz': 'jazz',
    'lounge': 'pop',
    'madchester': 'rock',
    'future house': 'electronic',
    'jazz piano': 'jazz',
    'deep chill': 'electronic',
    'anarcho-punk': 'punk',
    'fidget house': 'electronic',
    'christian metal': 'metal',
    'violin': 'classical',
    'vogue': 'electronic',
    'native american': 'world',
    'belgian rock': 'rock',
    'rap': 'hip hop',
    'breakcore': 'electronic',
    'indie rock': 'rock',
    'new wave': 'rock',
    'grunge pop': 'rock',
    'progressive psytrance': 'electronic',
    'baroque': 'classical',
    'scandipop': 'pop',
    'retro soul': 'soul',
    'country pop': 'country',
    'melodic power metal': 'metal',
    'melodic black metal': 'metal',
    'emo rap': 'hip hop',
    'cosmic black metal': 'metal',
    'hip pop': 'hip hop',
    'swedish black metal': 'metal',
    'free folk': 'folk',
    'symphonic rock': 'rock',
    'eurovision': 'pop',
    'swedish soul': 'soul',
    'psychedelic folk': 'folk',
    'new beat': 'electronic',
    'doo-wop': 'pop',
    'electro': 'electronic',
    'harp': 'classical',
    'oi': 'punk',
    'smooth soul': 'soul',
    'funk rock': 'rock',
    'power metal': 'metal',
    'freak folk': 'folk',
    'philly soul': 'soul',
    'singer-songwriter': 'folk',
    'industrial rock': 'rock',
    'bebop': 'jazz',
    'post-hardcore': 'punk',
    'uk pop': 'pop',
    'hard rock': 'rock',
    'art pop': 'pop',
    'christian pop': 'pop',
    'spanish pop': 'pop',
    'k-rock': 'rock',
    'celtic punk': 'punk',
    'west coast rap': 'hip hop',
    'post-screamo': 'punk',
    'sunshine pop': 'pop',
    'garage rock': 'rock',
    'orgcore': 'punk',
    'folk-pop': 'folk',
    'abstract hip hop': 'hip hop',
    'teen pop': 'pop',
    'shibuya-kei': 'world',
    'gypsy punk': 'punk',
    'greek pop': 'pop',
    'eurodance': 'electronic',
    'hardcore hip hop': 'hip hop',
    'melodic deathcore': 'metal',
    'contemporary gospel': 'soul',
    'scottish folk': 'folk',
    'french pop': 'pop',
    'canterbury scene': 'rock',
    'bassline': 'electronic',
    'hardcore': 'punk',
    'dansband': 'pop',
    'nu jazz': 'jazz',
    'crack rock steady': 'punk',
    'italian pop': 'pop',
    'theme': 'soundtrack',
    'symphonic black metal': 'metal',
    'gothic doom': 'metal',
    'grunge': 'rock',
    'gospel': 'soul',
    'afrobeat': 'world',
    'groove metal': 'metal',
    'latin pop': 'latin',
    'chaotic hardcore': 'punk',
    'dubstep': 'electronic',
    'soul': 'soul',
    'space ambient': 'ambient',
    'post-disco': 'pop',
    'dancehall': 'rock',
    'dark ambient': 'ambient',
    'fado': 'world',
    'wonky': 'electronic',
    'motown': 'soul',
    'melancholia': 'ambient',
    'traditional country': 'country',
    'country rock': 'country',
    'trap soul': 'electronic',
    'roots reggae': 'reggae',
    'big beat': 'electronic',
    'orchestra': 'classical',
    'anthem': 'pop',
    'pop rock': 'rock',
    'nigerian hip hop': 'hip hop',
    'cello': 'classical',
    'mpb': 'latin',
    'renaissance': 'classical',
    'emo punk': 'punk',
    'slayer': 'metal',
    'greek black metal': 'metal',
    'ragtime': 'jazz',
    'dutch house': 'electronic',
    'britpop': 'rock',
    'tango': 'latin',
    'blackened hardcore': 'metal',
    'reggae rock': 'reggae',
    'europop': 'pop',
    'electronic': 'electronic',
    'country blues': 'blues',
    'alternative dance': 'electronic',
    'breakbeat': 'electronic',
    'hi-nrg': 'electronic',
    'oriental metal': 'metal',
    'dance pop': 'pop',
    'glam rock': 'rock',
    'black death': 'metal',
    'lithuanian pop': 'pop',
    'noise': 'experimental',
    'nu metal': 'metal',
    'brazilian metal': 'metal',
    'industrial': 'electronic',
    'brazilian thrash metal': 'metal',
    'lovers rock': 'rock',
    'psychobilly': 'punk',
    'beatdown': 'punk',
    'merseybeat': 'rock',
    'k-indie': 'rock',
    'tribute': 'pop',
    'spoken word': 'theatrical',
    'heartland rock': 'rock',
    'afro-funk': 'funk',
    'girl group': 'pop',
    'math rock': 'rock',
    'electro-industrial': 'electronic',
    'atmospheric doom': 'metal',
    'christian hardcore': 'punk',
    'comedy': 'theatrical',
    'lo-fi': 'electronic',
    'motivation': 'ambient',
    'funk carioca': 'latin',
    'sludge metal': 'metal',
    'vocal house': 'electronic',
    'dream pop': 'pop',
    'praise': 'soul',
    'oud': 'world',
    'depressive black metal': 'metal',
    'afropop': 'world',
    '8-bit': 'electronic',
    'darkstep': 'electronic',
    'jam band': 'rock',
    'freestyle': 'electronic',
    'dance rock': 'rock',
    'psychedelic rock': 'rock',
    'brazilian rock': 'rock',
    'skiffle': 'folk',
    'progressive post-hardcore': 'punk',
    'j-metal': 'metal',
    'hip hop': 'hip hop',
    'latin alternative': 'latin',
    'axe': 'latin',
    'brostep': 'electronic',
    'detroit techno': 'electronic',
    'norwegian black metal': 'metal',
    'contemporary country': 'country',
    'emocore': 'punk',
    'pop folk': 'folk',
    'swedish pop': 'pop',
    'baroque pop': 'pop',
    'flamenco': 'latin',
    'neoclassical': 'classical',
    'dub techno': 'electronic',
    'protopunk': 'punk',
    'anti-folk': 'folk',
    'christian rock': 'rock',
    'a cappella': 'classical',
    'death metal': 'metal',
    'alternative hip hop': 'hip hop',
    'british folk': 'folk',
    'electronic rock': 'rock',
    'jump blues': 'blues',
    'vocal jazz': 'jazz',
    'k-pop': 'pop',
    'idol': 'pop',
    'jazz metal': 'metal',
    'new jack swing': 'soul',
    'beach house': 'electronic',
    'memphis blues': 'blues',
    'soundtrack': 'soundtrack',
    'raw black metal': 'metal',
    'bedroom pop': 'pop',
    'jump up': 'electronic',
    'healing': 'ambient',
    'space age pop': 'pop',
    'contemporary classical': 'classical',
    'vocaloid': 'ambient',
    'moombahton': 'latin',
    'ebm': 'electronic',
    'avant-garde jazz': 'jazz',
    'avant-garde': 'experimental',
    'dark wave': 'electronic',
    'irish hip hop': 'hip hop',
    'stoner rock': 'rock',
    'rock en espanol': 'latin',
    'jazz trumpet': 'jazz',
    'progressive house': 'electronic',
    'nordic folk': 'folk',
    'chillwave': 'electronic',
    'ska punk': 'punk',
    'technical death metal': 'metal',
    'uk hip hop': 'hip hop',
    'jangle pop': 'pop',
    'salsa': 'latin',
    'twee pop': 'pop',
    'blues-rock': 'rock',
    'british blues': 'blues',
    'midwest emo': 'punk',
    'banda': 'latin',
    'crust punk': 'punk',
    'piano blues': 'blues',
    'gangster rap': 'hip hop',
    'banjo': 'folk',
    'bluegrass': 'folk',
    'spanish indie pop': 'pop',
    'chillhop': 'electronic',
    'uplifting trance': 'electronic',
    'texas country': 'country',
    'grime': 'hip hop',
    'krautrock': 'rock',
    'opm': 'pop',
    'poetry': 'theatrical',
    'stoner metal': 'metal',
    'smooth jazz': 'jazz',
    'hyperpop': 'pop',
    'funeral doom': 'theatrical',
    'skinhead reggae': 'reggae',
    'acid techno': 'electronic',
    'german thrash metal': 'metal',
    'jazz': 'jazz',
    'c-pop': 'pop',
    'dreamo': 'rock',
    'symphonic metal': 'metal',
    'experimental hip hop': 'hip hop',
    'drum and bass': 'electronic',
    'j-pop': 'pop',
    'medieval': 'classical',
    'goa trance': 'electronic',
    'funk': 'funk',
    'operatic pop': 'pop',
    'kuduro': 'world',
    'lilith': 'ambient',
    'ambient techno': 'electronic',
    'trip hop': 'electronic',
    'schlager': 'world',
    'beats': 'electronic',
    'pop': 'pop',
    'ghettotech': 'electronic',
    'soft rock': 'rock',
    'psychedelic trance': 'electronic',
    'alternative metal': 'metal',
    'witch house': 'electronic',
    'futurepop': 'electronic',
    'football': 'soundtrack',
    'acid house': 'electronic',
    'deathcore': 'metal',
    'meme rap': 'hip hop',
    'bass music': 'electronic',
    'happy hardcore': 'electronic',
    'northern soul': 'soul',
    'screamo': 'punk',
    'hardcore punk': 'punk',
    'big band': 'jazz',
    'future garage': 'electronic',
    'usbm': 'metal',
    'cyber metal': 'metal',
    'downtempo': 'electronic',
    'space rock': 'rock',
    'chicago house': 'electronic',
    'boogaloo': 'funk',
    'straight edge': 'punk',
    'nerdcore': 'hip hop',
    'forro': 'latin',
    'beach music': 'pop',
    'piano rock': 'rock',
    'folk punk': 'punk',
    'soul jazz': 'jazz',
    'french black metal': 'metal',
    'melodic hardcore': 'punk',
    'modern hard rock': 'rock',
    'tech house': 'electronic',
    'trancecore': 'electronic',
    'scottish indie': 'rock',
    'math pop': 'pop',
    'latin': 'latin',
    'uk funky': 'funk',
    'dub reggae': 'reggae',
    'jersey club': 'electronic',
    'mandopop': 'pop',
    'dark techno': 'electronic',
    'hard bop': 'jazz',
    'ambient folk': 'folk',
    'technical black metal': 'metal',
    'minimal techno': 'electronic',
    'chill groove': 'electronic',
    'noise pop': 'pop',
    'progressive deathcore': 'metal',
    'symphonic death metal': 'metal',
    'indie punk': 'punk',
    'modern blues': 'blues',
    'drill': 'hip hop',
    'neofolk': 'folk',
    'neo-classical': 'classical',
    'alternative pop': 'pop',
    'east coast hip hop': 'hip hop',
    'ballroom': 'theatrical',
    'turkish pop': 'pop',
    'worship': 'soul',
    'reggae fusion': 'reggae',
    'southern soul': 'soul',
    'romanian pop': 'pop',
    'martial industrial': 'experimental',
    'alternative rock': 'rock',
    'progressive metal': 'metal',
    'cowpunk': 'punk',
    'sitar': 'world',
    'ukulele': 'folk',
    'canadian country': 'country',
    'ccm': 'soul',
    'microtonal': 'experimental',
    'glam metal': 'metal',
    'wrestling': 'soundtrack',
    'southern rock': 'rock',
    'black thrash': 'metal',
    'eurobeat': 'electronic',
    'indie emo': 'punk',
    'horror punk': 'punk',
    'drone metal': 'metal',
    'comedy rock': 'rock',
    'nwothm': 'metal',
    'video game music': 'soundtrack',
    'pop punk': 'punk',
    'atmospheric sludge': 'metal',
    'wave': 'electronic',
    'gothic americana': 'folk',
    'chiptune': 'electronic',
    'harmonica blues': 'blues',
    'digital hardcore': 'punk',
    'nu disco': 'electronic',
    'power electronics': 'electronic',
    'psychill': 'electronic',
    'mambo': 'latin',
    'progressive bluegrass': 'folk',
    'nintendocore': 'electronic',
    'ambient': 'ambient',
    'punk ska': 'punk',
    'mathcore': 'metal',
    'dark jazz': 'jazz',
    'underground hip hop': 'hip hop',
    'new rave': 'electronic',
    'albanian pop': 'pop',
    'ethereal wave': 'electronic',
    'underground rap': 'hip hop',
    'geek rock': 'rock',
    'texas blues': 'blues',
    'soca': 'latin',
    'glitch hop': 'electronic',
    "death 'n' roll": 'metal',
    'chicago soul': 'soul',
    'free jazz': 'jazz',
    'abstract': 'experimental',
    'big room': 'electronic',
    'british soul': 'soul',
    'australian rock': 'rock',
    'breaks': 'electronic',
    'finnish metal': 'metal',
    'world': 'world',
    'tropical': 'world',
    'old school hip hop': 'hip hop',
    'therapy': 'ambient',
    'pony': 'soundtrack',
    'swamp blues': 'blues',
    'art rock': 'rock',
    'c86': 'rock',
    'gothic metal': 'metal',
    'pop rap': 'hip hop',
    'new age': 'ambient',
    'german metal': 'metal',
    'classical': 'classical',
    'cool jazz': 'jazz',
    'british invasion': 'rock',
    'jazz funk': 'jazz',
    'melodic metalcore': 'metal',
    'russian rock': 'rock',
    'electropop': 'pop',
    'polish black metal': 'metal',
    'bossa nova': 'latin',
    'korean pop': 'pop',
    'swedish death metal': 'metal',
    'carnaval': 'latin',
    'lullaby': 'ambient',
    'neo soul': 'soul',
    'nwobhm': 'metal',
    'psychedelic doom': 'metal',
    'reggaeton': 'reggae',
    'brazilian death metal': 'metal',
    'nu gaze': 'rock',
    'drama': 'theatrical',
    'gothic symphonic metal': 'metal',
    'punk blues': 'punk',
    'jazz rap': 'hip hop',
    'classic rock': 'rock',
    'contemporary jazz': 'jazz',
    'jazz saxophone': 'jazz',
    'electronica': 'electronic',
    'indietronica': 'electronic',
    'dreamgaze': 'rock',
    'indie folk': 'folk',
    'djent': 'metal',
    'outsider': 'experimental',
    'viking metal': 'metal',
    'noise rock': 'rock',
    'romantico': 'latin',
    'rap metal': 'metal',
    'goregrind': 'metal',
    'broken beat': 'electronic',
    'pop edm': 'electronic',
    'afrikaans': 'world',
    'swedish synthpop': 'pop',
    'electropunk': 'electronic',
    'conscious hip hop': 'hip hop',
    'swing': 'jazz',
    'chicago blues': 'blues',
    'blackgaze': 'metal',
    'jazz guitar': 'jazz',
    'deathgrind': 'metal',
    'power violence': 'punk',
    'lds': 'theatrical',
    'phonk': 'hip hop',
    'hard house': 'electronic',
    'fast melodic punk': 'punk',
    'punk': 'punk',
    'german rock': 'rock',
    'experimental rock': 'rock',
    'latin hip hop': 'hip hop',
    'freakbeat': 'rock',
    'folk metal': 'metal',
    'progressive black metal': 'metal',
    'experimental pop': 'pop',
    'irish folk': 'folk',
    'vaporwave': 'electronic',
    'quiet storm': 'soul',
    'exotica': 'world',
    'broadway': 'theatrical',
    'minimal wave': 'electronic',
    'rock': 'rock',
    'folktronica': 'folk',
    'progressive rock': 'rock',
    'remix': 'electronic',
    'dub': 'electronic',
    'progressive metalcore': 'metal',
    'house': 'electronic',
    'atmospheric black metal': 'metal',
    'celtic': 'folk',
    'symphonic power metal': 'metal',
    'sufi': 'world',
    'progressive doom': 'metal',
    'samba': 'latin',
    'tzadik': 'experimental',
    'no wave': 'rock',
    'jazz trio': 'jazz',
    'tecnobrega': 'latin',
    'tone': 'ambient',
    'microhouse': 'electronic',
    'turntablism': 'hip hop',
    'disney': 'soundtrack',
    'trap latino': 'hip hop',
    'melodic death metal': 'metal',
    'power pop': 'pop',
    'classic soul': 'soul',
    'outlaw country': 'country',
    'doom metal': 'metal',
    'neo-psychedelic': 'rock',
    'rockabilly': 'rock',
    'karaoke': 'pop',
    'focus': 'ambient',
    'alternative metalcore': 'metal',
    'horrorcore': 'metal',
    'crunk': 'hip hop',
    'jazz fusion': 'jazz',
    'latin rock': 'latin',
    'street punk': 'punk',
    'shoegaze': 'rock',
    'black metal': 'metal',
    'country rap': 'hip hop',
    'christian music': 'soul',
    'vocal trance': 'electronic',
    'acoustic blues': 'blues',
    'instrumental rock': 'rock',
    'folk': 'folk',
    'future funk': 'funk',
    'bells': 'ambient',
    'new romantic': 'pop',
    'footwork': 'electronic',
    'folk rock': 'rock',
    'hawaiian': 'world',
    'glitch': 'electronic',
    'speed metal': 'metal',
    'finnish death metal': 'metal',
    'retro metal': 'metal',
    'rock nacional': 'rock',
    'technical brutal death metal': 'metal',
    'retro electro': 'electronic',
    'merengue': 'latin',
    'indie pop': 'pop',
    'experimental black metal': 'metal',
    'slamming deathcore': 'metal',
    'ska': 'punk',
    'garage pop': 'pop',
    'crossover thrash': 'metal',
    'nyhc': 'punk',
    'mashup': 'electronic',
    'electronicore': 'electronic',
    'delta blues': 'blues',
    'halloween': 'soundtrack',
    'ringtone': 'soundtrack',
    'hardstyle': 'electronic',
    'post-grunge': 'rock'
}

# --- Helper Functions ---
def load_and_preprocess_data(data_dir, lyrics_subdir, genres_file, metadata_file, lang_file, tags_file, subset_size=None, output_csv=None):
    """
    Loads and preprocesses the Music4All dataset, filtering for English-language tracks,
    merging relevant metadata and genre information, and mapping subgenres to broad categories.
    
    This function handles:
    - Language filtering (English only)
    - Subsetting (for experimentation or debugging)
    - Lyrics loading (from individual files)
    - Genre parsing and normalization (via SUBGENRE_TO_BROAD)
    - Merging of metadata and tag data
    - Optional saving to a CSV for reuse
    
    Args:
        data_dir (str): Root directory containing all dataset files and folders.
        lyrics_subdir (str): Subdirectory name containing individual lyrics text files.
        genres_file (str): Filename of the genre metadata TSV.
        metadata_file (str): Filename of the general metadata TSV.
        lang_file (str): Filename of the language metadata TSV.
        tags_file (str): Filename of the tags metadata TSV.
        subset_size (int, optional): Number of samples to randomly include. If None, load all.
        output_csv (str, optional): Path to save the final merged dataset. Defaults to "processed_data.csv".
    
    Returns:
        pd.DataFrame or None: Preprocessed and merged DataFrame, or None if any file-related error occurs.
    """
    try:
        # Load language data, filter English songs
        lang_path = os.path.join(data_dir, lang_file)
        lang_df = pd.read_csv(lang_path, delimiter='\t', header=0, names=[LYRICS_ID_COLUMN, 'lang'])
        english_ids = lang_df[lang_df['lang'] == 'en'][LYRICS_ID_COLUMN].unique()

        # Apply subset (if specified) before loading lyrics
        if subset_size is not None:
            rng = np.random.default_rng(RANDOM_STATE)
            english_ids = rng.choice(english_ids, size=subset_size, replace=False)
            print(f"Using a subset of {subset_size} English songs.")

        english_ids_df = pd.DataFrame({LYRICS_ID_COLUMN: english_ids})

        # Load required lyrics files
        lyrics_path = os.path.join(data_dir, lyrics_subdir)
        lyrics_data = []

        for track_id in tqdm(english_ids, desc="Loading Lyrics"):
            filename = f"{track_id}.txt"
            filepath = os.path.join(lyrics_path, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        lyrics = f.read()
                    lyrics_data.append({LYRICS_ID_COLUMN: str(track_id), TEXT_COLUMN: lyrics})
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
            else:
                print(f"Warning: Lyrics file not found for track ID: {track_id}")

        lyrics_df = pd.DataFrame(lyrics_data)

        # Merge to keep only selected songs
        lyrics_df = pd.merge(english_ids_df, lyrics_df, on=LYRICS_ID_COLUMN, how='inner')

        # Load genres
        genres_path = os.path.join(data_dir, genres_file)
        genres_df = pd.read_csv(genres_path, delimiter='\t', header=0, names=[LYRICS_ID_COLUMN, "genres_raw"])
        genres_df[LYRICS_ID_COLUMN] = genres_df[LYRICS_ID_COLUMN].astype(str)
        genres_df['genres'] = genres_df['genres_raw'].str.split(',')
        genres_df = genres_df.drop(columns=['genres_raw'])

        # Merge lyrics and genres
        merged_df = pd.merge(lyrics_df, genres_df, on=LYRICS_ID_COLUMN, how="left")

        # Load metadata
        metadata_path = os.path.join(data_dir, metadata_file)
        metadata_df = pd.read_csv(metadata_path, delimiter='\t', header=0)
        metadata_df[LYRICS_ID_COLUMN] = metadata_df[LYRICS_ID_COLUMN].astype(str)

        # Load tags
        tags_path = os.path.join(data_dir, tags_file)
        tags_df = pd.read_csv(tags_path, delimiter='\t', header=0, names=[LYRICS_ID_COLUMN, TAG_COLUMN])
        tags_df[LYRICS_ID_COLUMN] = tags_df[LYRICS_ID_COLUMN].astype(str)

        # Final merge
        merged_df = pd.merge(merged_df, metadata_df, on=LYRICS_ID_COLUMN, how="left")
        merged_df = pd.merge(merged_df, tags_df, on=LYRICS_ID_COLUMN, how="left")

        # Amalgamate genres for better classification, mapping infrequent to broader genres
        def amalgamate_genres(genres, mapping):
            if isinstance(genres, str) and genres.startswith('['):
                try:
                    genres = ast.literal_eval(genres)
                except Exception:
                    return genres
                
            if isinstance(genres, list):
                broad_genres = [mapping.get(g.strip().lower(), g.strip()) for g in genres]
                return list(set(broad_genres))
            return genres

        merged_df["genres"] = merged_df["genres"].apply(lambda g: amalgamate_genres(g, SUBGENRE_TO_BROAD))
        
        # Save data to CSV file
        if output_csv is None:
            output_csv = "processed_data.csv"

        try:
            merged_df.to_csv(output_csv, index=False, encoding='utf-8')
            print(f"Preprocessed data successfully saved to {output_csv}")
        except Exception as e:
            print(f"Error saving preprocessed data to CSV: {e}")

        return merged_df

    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
        return None
    except pd.errors.EmptyDataError as e:
        print(f"Error: Empty CSV file: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def filter_dataset_for_top_genres(df, genre_column_name, top_genres_list):
    """
    Filters a DataFrame to keep only rows where the genre list contains
    exclusively genres from the provided top_genres_list.

    Args:
        df (pd.DataFrame): The input DataFrame.
        genre_column_name (str): The name of the column containing genre lists.
                                  Assumes this column contains lists of strings.
        top_genres_list (list): A list of the allowed "top" genres.

    Returns:
        pd.DataFrame: A new DataFrame containing only the filtered rows.
                      Returns an empty DataFrame if errors occur.
    """
    if genre_column_name not in df.columns:
        print(f"Error: Column '{genre_column_name}' not found in DataFrame.")
        return pd.DataFrame() # Return an empty DataFrame

    if not isinstance(top_genres_list, list):
        print("Error: top_genres_list must be a list.")
        return pd.DataFrame()

    # Convert the list to a set for efficient checking
    top_genres_set = set(top_genres_list)

    # Checks if all genres in list are within allowed top set
    def check_genres(song_genre_list):
        # Handle rows with missing or invalid genre data
        if not isinstance(song_genre_list, list) or not song_genre_list:
            return False

        # Check if all genres in song's list are present in top_genres_set
        return all(genre in top_genres_set for genre in song_genre_list)

    # Apply the checking function to create a boolean mask
    keep_mask = df[genre_column_name].apply(check_genres)

    # Filter the DataFrame using the mask
    filtered_df = df[keep_mask].copy()

    # Optional print summary
    print("-" * 50)
    print(f"Filtering for top genres: {top_genres_list}")
    print(f"Original dataset size: {len(df)}")
    print(f"Filtered dataset size: {len(filtered_df)}")
    if len(df) > 0:
        print(f"Percentage kept: {100 * len(filtered_df) / len(df):.2f}%")
    print("-" * 50)

    return filtered_df

def analyze_genre_distribution(df, genre_column_name, top_n=20):
    """
    Analyzes and prints the distribution of individual genres and genre combinations
    from a specified DataFrame column containing lists of genres.

    Args:
        df (pd.DataFrame): The DataFrame containing the genre data.
        genre_column_name (str): The name of the column with genre lists.
                                  Assumes this column contains lists of strings
                                  (or NaN/None for missing data).
        top_n (int): The number of top results to display for both
                     individual genres and combinations. Defaults to 20.
    """
    if genre_column_name not in df.columns:
        print(f"Error: Column '{genre_column_name}' not found in DataFrame.")
        return
    print(f"\nAnalyzing Genre Distributions for column: '{genre_column_name}'...")

    individual_genre_counts = Counter()
    genre_combination_counts = Counter()

    for genre_list in df[genre_column_name].dropna():
        if isinstance(genre_list, list) and genre_list:
            # Count individual genres within list
            valid_genres_in_list = []
            
            for genre in genre_list:
                if isinstance(genre, str) and genre:
                    individual_genre_counts[genre] += 1
                    valid_genres_in_list.append(genre)

            # Count combinations with sorted tuple as key
            if valid_genres_in_list:
                # Only count if there was at least one valid genre
                combination_key = tuple(sorted(valid_genres_in_list))
                genre_combination_counts[combination_key] += 1

    # Print top N individual genres
    print(f"\n--- Top {top_n} Individual Genres ---")
    top_individual = individual_genre_counts.most_common(top_n)
    if not top_individual:
        print("No individual genres found to count.")
    else:
        for genre, count in top_individual:
            print(f"- {genre}: {count}")

    # Print top N genre combinations
    print(f"\n--- Top {top_n} Genre Combinations ---")
    top_combinations = genre_combination_counts.most_common(top_n)
    if not top_combinations:
        print("No genre combinations found to count.")
    else:
        for combination_tuple, count in top_combinations:
            combination_str = ", ".join(combination_tuple)
            print(f"- [{combination_str}]: {count}")

    print("-" * (30 + len(str(top_n)))) # Separator line


### FEATURE ENGINEERING ###

def preprocess_lyrics(text):
    """
    Preprocesses a lyric string by converting to lowercase, removing punctuation,
    tokenizing into words, removing English stopwords, and lemmatizing the result.
    
    Args:
        text (str): The input lyric text to preprocess.
    
    Returns:
        list: A list of cleaned and lemmatized word tokens.
    """
    if not isinstance(text, str):
        return []
    
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    return lemmatized_tokens

def calculate_rhyme_density(tokens):
    """
    Calculates the rhyme density of a list of word tokens using phoneme endings
    from the CMU Pronouncing Dictionary.
    
    Args:
        tokens (list of str): A list of lyric tokens.
    
    Returns:
        float: The estimated rhyme density (0.0 to 1.0) based on phoneme overlap.
    """
    if not tokens:
        return 0.0

    # Helper function to get last syllable or short phoneme slice
    def get_rhyme_key(word):
        phones_list = pronouncing.phones_for_word(word)
        if not phones_list:
            return None
        phonemes = phones_list[0].split()

        return "_".join(phonemes[-2:])

    # Count key frequency
    key_counts = Counter()
    for w in tokens:
        k = get_rhyme_key(w)

        # Only count words that have a pronouncing entry
        if k:
            key_counts[k] += 1

    # If fewer than 2 tokens had valid rhyme keys, no measurable rhyme
    if sum(key_counts.values()) < 2:
        return 0.0

    # Get fraction of “rhyming pairs” of all possible pairs
    total_words = sum(key_counts.values())
    total_pairs = total_words * (total_words - 1) / 2
    rhyme_pairs = 0

    for count in key_counts.values():
        rhyme_pairs += count * (count - 1) / 2

    return (rhyme_pairs / total_pairs) if total_pairs > 0 else 0.0

def calculate_lexical_complexity(tokens):
    """
    Calculates lexical complexity as the ratio of unique words to total words.
    
    Args:
        tokens (list of str): A list of lyric tokens.
    
    Returns:
        float: Lexical complexity score (0.0 to 1.0). Returns 0.0 for empty input.
    """
    if not tokens or len(tokens) == 0:
        return 0.0
    
    return len(set(tokens)) / len(tokens)

def calculate_sentiment(text):
    """
    Analyzes sentiment of the lyric text using TextBlob.
    
    Args:
        text (str): The input lyric text.
    
    Returns:
        tuple: A tuple (polarity, subjectivity) where both values are floats.
    """
    if not isinstance(text, str) or not text:
        return 0.0, 0.0
    
    analysis = TextBlob(text)

    return analysis.sentiment.polarity, analysis.sentiment.subjectivity

def calculate_readability(text):
    """
    Calculates the Flesch Reading Ease score of the lyric text.
    
    Args:
        text (str): The input lyric text.
    
    Returns:
        float: A readability score (higher means easier to read). Returns 0.0 on error.
    """
    if not isinstance(text, str) or not text:
        return 0.0
    
    try:
        return textstat.flesch_reading_ease(text)
    except Exception:
        return 0.0

def load_pretrained_word2vec():
    """
    Loads Google's pre-trained Word2Vec embeddings.
    
    Returns:
        KeyedVectors: Loaded Word2Vec model with pre-trained vectors.
    """
    return KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

def generate_word2vec_embeddings(texts, model):
    """
    Generates Word2Vec embeddings by averaging word vectors for each lyric.
    
    Args:
        texts (list of str): List of lyric strings.
        model (KeyedVectors): Pre-trained Word2Vec model.
    
    Returns:
        list of np.ndarray: List of averaged embedding vectors for each lyric.
    """
    embeddings = []
    for text in tqdm(texts, desc="Applying Word2Vec"):
        words = text.split() if isinstance(text, str) else []
        word_vectors = [model[word] for word in words if word in model]
        embeddings.append(np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size))

    return embeddings

def generate_fasttext_embeddings(texts, vector_size=100, window=5, min_count=1):
    """
    Trains a FastText model on the dataset and generates embeddings for each lyric.
    
    Args:
        texts (list of str): List of lyric strings.
        vector_size (int): Dimensionality of the embedding vectors.
        window (int): Window size for context words.
        min_count (int): Minimum word count threshold for training.
    
    Returns:
        list of np.ndarray: List of averaged FastText embedding vectors per lyric.
    """
    tokenized_texts = [text.split() for text in texts]
    model = FastText(sentences=tokenized_texts, vector_size=vector_size, window=window, min_count=min_count)
    embeddings = []

    for tokens in tqdm(tokenized_texts, desc="Generating FastText Embeddings"):
        if tokens:
            embeddings.append(model.wv[tokens].mean(axis=0))
        else:
            embeddings.append(np.zeros(vector_size))

    return embeddings

def generate_bert_embeddings(texts, tokenizer, model):
    """
    Generates BERT embeddings for each lyric using the mean of hidden states.
    
    Args:
        texts (list of str): List of lyric strings.
        tokenizer (transformers.BertTokenizer): Pre-trained BERT tokenizer.
        model (transformers.BertModel): Pre-trained BERT model.
    
    Returns:
        list of np.ndarray: List of 768-dimensional embeddings per lyric.
    """
    embeddings = []

    for text in tqdm(texts, desc="Generating BERT Embeddings"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())

    return embeddings

def extract_features(df):
    """
    Extracts handcrafted and deep learning-based features from song lyrics.
    
    This function manages feature caching to avoid recomputation. It calculates:
    - Tokenized lyrics
    - Rhyme density
    - Lexical complexity
    - Sentiment polarity and subjectivity
    - Readability score
    - Word2Vec, FastText, and BERT embeddings
    
    Args:
        df (pandas.DataFrame): DataFrame containing at least the lyrics text.
    
    Returns:
        pandas.DataFrame: DataFrame with all extracted features added as columns.
    """
    # Utility function for caching or computing each feature
    def load_or_compute_feature(df, feature_name, compute_fn, cache_file):
        """
        If the feature already exists in df, skip.
        Otherwise, try to load from a cache_file.
        If cache_file does not exist, compute the feature, store it in df, and save.
        """
        if feature_name in df.columns:
            print(f"Feature '{feature_name}' already in DataFrame, skipping computation.")
            return df

        if os.path.exists(cache_file):
            print(f"Loading cached feature '{feature_name}' from {cache_file}...")
            cached_series = joblib.load(cache_file)
            df[feature_name] = cached_series
            return df

        print(f"Computing feature '{feature_name}'...")
        df[feature_name] = compute_fn(df)

        # Save feature to joblib file
        to_save = df[feature_name]
        joblib.dump(to_save, cache_file)
        print(f"Feature '{feature_name}' computed and cached to {cache_file}.")

        return df

    # Tokenizes and lemmatizes lyrics using NLTK tools
    def compute_tokens(df):
        tqdm.pandas(desc="Preprocessing Lyrics")

        return df[TEXT_COLUMN].progress_apply(preprocess_lyrics)

    # Computes rhyme density using phoneme matching from CMU Pronouncing Dictionary
    def compute_rhyme_density(df):
        tqdm.pandas(desc="Calculating Rhyme Density")

        return df["tokens"].progress_apply(calculate_rhyme_density)

    # Measures vocabulary diversity (unique tokens / total tokens)
    def compute_lexical_complexity(df):
        tqdm.pandas(desc="Calculating Lexical Complexity")

        return df["tokens"].progress_apply(calculate_lexical_complexity)

    # Computes sentiment polarity using TextBlob
    def compute_sentiment_polarity(df):
        tqdm.pandas(desc="Calculating Sentiment Polarity")

        return df[TEXT_COLUMN].progress_apply(lambda x: calculate_sentiment(x)[0])

    # Computes sentiment subjectivity using TextBlob
    def compute_sentiment_subjectivity(df):
        tqdm.pandas(desc="Calculating Sentiment Subjectivity")

        return df[TEXT_COLUMN].progress_apply(lambda x: calculate_sentiment(x)[1])

    # Computes Flesch Reading Ease score using textstat
    def compute_readability(df):
        tqdm.pandas(desc="Calculating Readability")

        return df[TEXT_COLUMN].progress_apply(calculate_readability)

    # Generates Word2Vec embeddings by averaging pretrained Google vectors
    def compute_word2vec_embedding(df):
        tqdm.pandas(desc="Generating Word2Vec Embeddings")
        word2vec_model = load_pretrained_word2vec()

        return generate_word2vec_embeddings(df[TEXT_COLUMN].tolist(), word2vec_model)

    # Trains FastText model and computes average embeddings
    def compute_fasttext_embedding(df):
        tqdm.pandas(desc="Generating FastText Embeddings")

        return generate_fasttext_embeddings(df[TEXT_COLUMN].tolist())

    # Uses pretrained BERT to extract mean-pooled contextual embeddings
    def compute_bert_embedding(df):
        tqdm.pandas(desc="Generating BERT Embeddings")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_model = BertModel.from_pretrained("bert-base-uncased")

        return generate_bert_embeddings(df[TEXT_COLUMN].tolist(), tokenizer, bert_model)

    # 1) Tokens
    df = load_or_compute_feature(
        df,
        feature_name="tokens",
        compute_fn=compute_tokens,
        cache_file="cache_tokens.joblib"
    )

    # 2) Rhyme density
    df = load_or_compute_feature(
        df,
        feature_name="rhyme_density",
        compute_fn=compute_rhyme_density,
        cache_file="cache_rhyme_density.joblib"
    )

    # 3) Lexical complexity
    df = load_or_compute_feature(
        df,
        feature_name="lexical_complexity",
        compute_fn=compute_lexical_complexity,
        cache_file="cache_lexical_complexity.joblib"
    )

    # 4) Sentiment polarity
    df = load_or_compute_feature(
        df,
        feature_name="sentiment_polarity",
        compute_fn=compute_sentiment_polarity,
        cache_file="cache_sentiment_polarity.joblib"
    )

    # 5) Sentiment subjectivity
    df = load_or_compute_feature(
        df,
        feature_name="sentiment_subjectivity",
        compute_fn=compute_sentiment_subjectivity,
        cache_file="cache_sentiment_subjectivity.joblib"
    )

    # 6) Readability
    df = load_or_compute_feature(
        df,
        feature_name="readability",
        compute_fn=compute_readability,
        cache_file="cache_readability.joblib"
    )

    # 7) Word2Vec embedding
    df = load_or_compute_feature(
        df,
        feature_name="word2vec_embedding",
        compute_fn=compute_word2vec_embedding,
        cache_file="cache_word2vec.joblib"
    )

    # 8) FastText embedding
    df = load_or_compute_feature(
        df,
        feature_name="fasttext_embedding",
        compute_fn=compute_fasttext_embedding,
        cache_file="cache_fasttext.joblib"
    )

    # 9) BERT embedding
    df = load_or_compute_feature(
        df,
        feature_name="bert_embedding",
        compute_fn=compute_bert_embedding,
        cache_file="cache_bert.joblib"
    )

    df.drop(columns=["tokens"], inplace=True, errors='ignore')

    return df


###  MODEL TRAINING ###

def create_train_test_split(df, features, target):
    """
    Splits data into training and testing sets using multi-label stratified sampling.
    
    Extracts handcrafted and deep learning features, binarizes genre labels,
    and applies iterative stratification to maintain label distribution across splits.
    
    Args:
        df (pandas.DataFrame): The full dataset with features and genre labels.
        features (list): List of column names for handcrafted features.
        target (str): Column name containing genre labels.
    
    Returns:
        tuple: X_train_hc, X_test_hc, X_train_dl, X_test_dl, y_train, y_test
    """
    # Convert target column to list if necessary
    def convert_label(x):
        if isinstance(x, str):
            # Try to convert string representation of list to actual list
            try:
                return ast.literal_eval(x)
            except Exception:
                # Split on comma, otherwise
                return [g.strip() for g in x.split(',')]
        elif isinstance(x, list):
            return x
        else:
            return [x]
    
    # Create a new column with multi-label data as lists
    df['genres_list'] = df[target].apply(convert_label)

    # Binarize target
    mlb = MultiLabelBinarizer()
    y_bin = mlb.fit_transform(df['genres_list'])

    # Extract and normalize handcrafted features
    X_handcrafted = df[features].values
    scaler = StandardScaler()
    X_handcrafted = scaler.fit_transform(X_handcrafted)

    # Extract deep learning features (no scaling)
    X_deep_learning = np.hstack([
        np.stack(df["word2vec_embedding"].values),
        np.stack(df["fasttext_embedding"].values),
        np.stack(df["bert_embedding"].values)
    ])
    
    # Use iterative stratification to split indices
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    for train_idx, test_idx in msss.split(np.zeros(len(y_bin)), y_bin):
         X_train_hc = X_handcrafted[train_idx]
         X_test_hc = X_handcrafted[test_idx]
         X_train_dl = X_deep_learning[train_idx]
         X_test_dl = X_deep_learning[test_idx]
         y_train = [df['genres_list'].iloc[i] for i in train_idx]
         y_test = [df['genres_list'].iloc[i] for i in test_idx]

    return X_train_hc, X_test_hc, X_train_dl, X_test_dl, y_train, y_test, train_idx, test_idx

def train_svm(X_train, y_train):
    """
    Trains a multi-label Support Vector Machine (SVM) classifier using a OneVsRest strategy.
    
    Args:
        X_train (np.ndarray): Feature matrix for training.
        y_train (list of list): Multi-label genre targets.
    
    Returns:
        model (OneVsRestClassifier): Trained SVM classifier.
        mlb (MultiLabelBinarizer): Fitted label binarizer for use during testing.
    """
    mlb = MultiLabelBinarizer()
    y_train_bin = mlb.fit_transform(y_train)
    base_svm = LinearSVC(random_state=RANDOM_STATE, max_iter = 10000, class_weight='balanced')
    model = OneVsRestClassifier(base_svm)
    
    with tqdm(total=1, desc="Training SVM classifier") as pbar:
        model.fit(X_train, y_train_bin)
        pbar.update(1)
    
    # Save trained model
    joblib.dump((model, mlb), 'svm_model.joblib')
    print("SVM model saved to svm_model.joblib")
    
    return model, mlb

def train_logistic_regression(X_train, y_train):
    """
    Trains a multi-label Logistic Regression classifier using a OneVsRest strategy.
    
    Args:
        X_train (np.ndarray): Feature matrix for training.
        y_train (list of list): Multi-label genre targets.
    
    Returns:
        model (OneVsRestClassifier): Trained Logistic Regression classifier.
        mlb (MultiLabelBinarizer): Fitted label binarizer for use during testing.
    """
    mlb = MultiLabelBinarizer()
    y_train_bin = mlb.fit_transform(y_train)
    base_lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=10000, class_weight='balanced')
    model = OneVsRestClassifier(base_lr)
    
    with tqdm(total=1, desc="Training Logistic Regression classifier") as pbar:
        model.fit(X_train, y_train_bin)
        pbar.update(1)
    
    # Save trained model
    joblib.dump((model, mlb), 'logistic_regression_model.joblib')
    print("Logistic Regression model saved to logistic_regression_model.joblib")
    
    return model, mlb

def train_random_forest(X_train, y_train):
    """
    Trains a multi-label Random Forest classifier using a OneVsRest strategy.
    
    Args:
        X_train (np.ndarray): Feature matrix for training.
        y_train (list of list): Multi-label genre targets.
    
    Returns:
        model (OneVsRestClassifier): Trained Random Forest classifier.
        mlb (MultiLabelBinarizer): Fitted label binarizer for use during testing.
    """
    mlb = MultiLabelBinarizer()
    y_train_bin = mlb.fit_transform(y_train)
    base_rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    model = OneVsRestClassifier(base_rf)
    
    with tqdm(total=1, desc="Training Random Forest classifier") as pbar:
        model.fit(X_train, y_train_bin)
        pbar.update(1)
    
    # Save trained model
    joblib.dump((model, mlb), 'random_forest_model.joblib')
    print("Random Forest model saved to random_forest_model.joblib")
    
    return model, mlb


### EVALUATION ###

def evaluate_model(model, X_test, y_test, mlb):
    """
    Evaluates a multi-label classifier using standard performance metrics.
    
    Args:
        model: Trained OneVsRestClassifier model.
        X_test (np.ndarray): Feature matrix for the test set.
        y_test (list of lists): True genre labels for the test set.
        mlb (MultiLabelBinarizer): Binarizer used during training.
    
    Returns:
        dict: Dictionary containing evaluation metrics, including accuracy, F1 scores, Jaccard index,
              hamming loss, and a full classification report.
    """
    # Transform true labels into binary matrix using same mlb as training
    y_test_list_of_lists = [lst if isinstance(lst, list) else [lst] for lst in y_test]
    y_test_bin = mlb.transform(y_test_list_of_lists)
    known_labels = set(mlb.classes_)
    unknown = set(g for lst in y_test_list_of_lists for g in lst if g not in known_labels)

    # Obtain binary predictions from model
    y_pred_bin = model.predict(X_test)    
    target_names = mlb.classes_

    # Compute multi-label evaluation metrics
    accuracy = accuracy_score(y_test_bin, y_pred_bin)
    hamming = hamming_loss(y_test_bin, y_pred_bin)
    f1_micro = f1_score(y_test_bin, y_pred_bin, average='micro')
    f1_macro = f1_score(y_test_bin, y_pred_bin, average='macro')
    jaccard = jaccard_score(y_test_bin, y_pred_bin, average='samples')

    report_dict = classification_report(
        y_test_bin,
        y_pred_bin,
        target_names=target_names, # <<< Pass the names here
        output_dict=True,
        zero_division=0
    )
    report_string = classification_report(
        y_test_bin,
        y_pred_bin,
        target_names=target_names,
        zero_division=0
    )

    # Print results
    print("\n--- Evaluation Report ---")
    print(f"Target Genres: {', '.join(target_names)}")
    print(f"Accuracy (Exact Match Ratio): {accuracy:.4f}")
    print(f"Hamming Loss: {hamming:.4f}")
    print(f"F1 Score (Micro): {f1_micro:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"Jaccard Score (Samples): {jaccard:.4f}")
    print("\nClassification Report (Per Genre):\n", report_string)
    # print("\nClassification Report:\n", classification_report(y_test_bin, y_pred_bin, zero_division=0))
    
    return {
        "accuracy": accuracy,
        "hamming_loss": hamming,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "jaccard": jaccard,
        "classification_report": report_dict
    }

def perform_feature_ablation(df, feature_columns, target_column, model_type="SVM"):
    """
    Performs feature ablation by systematically removing each handcrafted feature,
    retraining the model, and comparing performance against the full model.
    
    Args:
        df (pd.DataFrame): The dataset containing feature and label columns.
        feature_columns (list): List of handcrafted features to ablate.
        target_column (str): Name of the genre label column.
        model_type (str): Model to use ("SVM", "LogisticRegression", or "RandomForest").
    
    Returns:
        pd.DataFrame: DataFrame summarizing accuracy drops per ablated feature.
    """
    # base_X_train_hc, base_X_test_hc, base_X_train_dl, base_X_test_dl, base_y_train, base_y_test = create_train_test_split(df, feature_columns, target_column)
    base_X_train_hc, base_X_test_hc, base_X_train_dl, base_X_test_dl, base_y_train, base_y_test, _, _ = create_train_test_split(df, feature_columns, target_column)

    # Combine both feature sets
    base_X_train = np.hstack([base_X_train_dl, base_X_train_hc])
    base_X_test = np.hstack([base_X_test_dl, base_X_test_hc])

    # Choose model type
    if model_type == "SVM":
        model = train_svm
    elif model_type == "LogisticRegression":
        model = train_logistic_regression
    elif model_type == "RandomForest":
        model = train_random_forest
    else:
        raise ValueError("Invalid model type. Choose from 'SVM', 'LogisticRegression', or 'RandomForest'.")

    # Train on full feature set
    base_model, mlb = model(base_X_train, base_y_train)
    base_metrics = evaluate_model(base_model, base_X_test, base_y_test, mlb)
    print(f"Base Accuracy with all features: {base_metrics['accuracy']:.4f}")

    # Perform feature ablation
    ablation_results = []
    for feature in feature_columns:
        reduced_features = [f for f in feature_columns if f != feature]
        X_train_hc, X_test_hc, X_train_dl, X_test_dl, y_train, y_test, _, _ = create_train_test_split(df, reduced_features, target_column)
        X_train = np.hstack([X_train_dl, X_train_hc])
        X_test = np.hstack([X_test_dl, X_test_hc])

        # Train and evaluate
        ablated_model, mlb = model(X_train, y_train)
        ablated_metrics = evaluate_model(ablated_model, X_test, y_test, mlb)
        print(f"Ablated Accuracy: {ablated_metrics['accuracy']:.4f}")

        # Measure accuracy drop
        accuracy_drop = base_metrics["accuracy"] - ablated_metrics["accuracy"]
        ablation_results.append([feature, accuracy_drop])

        # Save to a unique file name for each dropped feature
        one_drop_file = f"feature_ablation_metrics_{model_type}_{feature}.json"
        with open(one_drop_file, "w") as f:
            json.dump(ablated_metrics, f, indent=4)
        print(f"Removing '{feature}' decreased accuracy by {accuracy_drop:.4f}, full metrics in {one_drop_file}")

    # Save results to CSV
    ablation_df = pd.DataFrame(ablation_results, columns=["Feature", "Accuracy Drop"])
    ablation_csv = f"feature_ablation_results_{model_type}.csv"
    ablation_df.to_csv(ablation_csv, index=False)
    print(f"Feature ablation results saved to {ablation_csv}.")
    
    # Plot accuracy drops    
    ablation_df.sort_values(by='Accuracy Drop', ascending=False, inplace=True)
    plt.figure(figsize=(10, 6))
    plt.bar(ablation_df['Feature'], ablation_df['Accuracy Drop'])
    plt.title(f"Feature Ablation Impact on {model_type} Accuracy")
    plt.ylabel("Accuracy Drop")
    plt.xlabel("Feature Removed")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"feature_ablation_{model_type}.png")
    plt.show()
    
    return ablation_df

def analyze_errors(model, X_test, y_test, df, feature_columns, mlb, output_file=None, clf_name="Classifier"):
    """
    Identifies common misclassifications, similar misclassified lyric pairs using cosine similarity,
    and feature contributions to those errors. Results are saved and optionally visualized.

    Args:
        model: Trained classifier used for prediction.
        X_test (np.ndarray): Test feature matrix.
        y_test (list of lists): True genre labels (as lists of strings) for test samples.
        df (pd.DataFrame): Original dataframe subset containing lyrics and features for the test set.
        feature_columns (list): List of numeric feature columns for error attribution.
        mlb (MultiLabelBinarizer): The *fitted* MultiLabelBinarizer instance used during training.
                                    Crucial for converting predictions back to labels.
        output_file (str, optional): Path to save the JSON-formatted error report.
        clf_name (str): Name of the classifier for titles and filenames.

    Returns:
        tuple: (most_common_errors_df, similar_misclassified_pairs, feature_diffs)
               Returns (pd.DataFrame(), [], {}) if no misclassifications.
    """
    print(f"\n--- Analyzing errors for {clf_name} using Genre Names ---")

    # get Predictions and true Labels in binary
    y_pred_bin = model.predict(X_test)
    try:
        y_test_list_of_lists = [lst if isinstance(lst, list) else [lst] for lst in y_test]
        y_test_bin = mlb.transform(y_test_list_of_lists)
    except ValueError as e:
         print(f"Error transforming y_test with MLB: {e}")
         print("Ensure y_test contains lists of genres known to the MLB.")

         known_labels = set(mlb.classes_)
         unknown = set()

         for lst in y_test_list_of_lists:
             for genre in lst:
                 if genre not in known_labels:
                     unknown.add(genre)

         if unknown:
             print(f"Unknown labels found in y_test: {unknown}")

         return pd.DataFrame(), [], {}
    except Exception as e:
         print(f"Unexpected error during y_test transformation: {e}")

         return pd.DataFrame(), [], {}

    # Find indices of misclassified samples
    misclassified_mask = ~np.all(y_test_bin == y_pred_bin, axis=1)
    misclassified_indices_in_test = np.where(misclassified_mask)[0]

    print(f"Total Samples in Test Set: {len(y_test)}")
    print(f"Total Misclassifications: {len(misclassified_indices_in_test)}")

    if len(misclassified_indices_in_test) == 0:
        print("No misclassifications found. Error analysis skipped.")

        if output_file:
             results = {
                 "most_common_errors": [],
                 "similar_misclassified_pairs": [],
                 "feature_diffs": {}
             }
             with open(output_file, "w") as f:
                  json.dump(results, f, indent=4)

        return pd.DataFrame(), [], {}

    # Get genre names for misclassified samples
    true_labels_list = [y_test[i] for i in misclassified_indices_in_test]
    predicted_labels_tuples = mlb.inverse_transform(y_pred_bin[misclassified_indices_in_test])

    # Convert genre lists/tuples to consistent string representations for xounting
    true_label_strings = [", ".join(sorted(lst)) if lst else "[None]" for lst in true_labels_list]
    predicted_label_strings = [", ".join(sorted(list(t))) if t else "[None]" for t in predicted_labels_tuples]

    # Count misclassification pairs using genre strings
    error_pairs_counter = Counter(zip(true_label_strings, predicted_label_strings))

    if error_pairs_counter:
        most_common_errors_list = [
            {"True": true, "Predicted": pred, "count": count}
            for (true, pred), count in error_pairs_counter.most_common()
        ]
        most_common_errors_df = pd.DataFrame(most_common_errors_list)
    else:
        most_common_errors_df = pd.DataFrame(columns=["True", "Predicted", "count"])

    print("\nMost Common Misclassifications (True -> Predicted):")
    print(most_common_errors_df.head(10))

    # Create heatmap
    if not most_common_errors_df.empty:
        top_n_heatmap = 20
        heatmap_data = most_common_errors_df.head(top_n_heatmap)
        try:
            pivot = pd.pivot_table(heatmap_data, values='count', index='True', columns='Predicted', fill_value=0, aggfunc=np.sum)

            if not pivot.empty:
                plt.figure(figsize=(max(10, pivot.shape[1]*0.8), max(8, pivot.shape[0]*0.6))) # Adjust size dynamically
                sns.heatmap(pivot, annot=True, fmt="g", cmap="Reds")
                plt.title(f"Top {top_n_heatmap} Misclassifications ({clf_name})")
                plt.ylabel("True Label Combination")
                plt.xlabel("Predicted Label Combination")
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                plt.savefig(f"misclassification_heatmap_{clf_name}.png")
                plt.close()
                print(f"Misclassification heatmap saved to misclassification_heatmap_{clf_name}.png")
            else:
                print("Pivot table for heatmap is empty.")

        except Exception as e:
            print(f"Could not generate heatmap: {e}")

    # Compute similarity of misclassified samples
    misclassified_texts = df.iloc[misclassified_indices_in_test][TEXT_COLUMN].astype(str)

    similar_misclassified_pairs = []
    if len(misclassified_indices_in_test) > 1:
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(misclassified_texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)

            # Compute adaptive similarity threshold
            if similarity_matrix.shape[0] > 1:
                flat_values = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
                if len(flat_values) > 0:
                    threshold = np.percentile(flat_values, 90)

                    # Identify highly similar misclassified examples
                    for i in range(len(misclassified_indices_in_test)):
                        for j in range(i + 1, len(misclassified_indices_in_test)):
                            if similarity_matrix[i, j] > threshold:
                                # Store the indices relative to the start of the test set (df)
                                similar_misclassified_pairs.append((int(misclassified_indices_in_test[i]), int(misclassified_indices_in_test[j])))
                else:
                    print("Not enough unique pairs to calculate similarity threshold.")
            else:
                 print("Not enough samples for similarity comparison.")

        except Exception as e:
            print(f"Error during similarity calculation: {e}")

    print(f"\nFound {len(similar_misclassified_pairs)} highly similar misclassified pairs (based on TF-IDF).")

    # Feature pmpact on misclassification
    feature_diffs = {}
    correct_mask = ~misclassified_mask
    correct_indices_in_test = np.where(correct_mask)[0]

    if len(correct_indices_in_test) > 0 and len(misclassified_indices_in_test) > 0:
        df_correct = df.iloc[correct_indices_in_test]
        df_misclassified = df.iloc[misclassified_indices_in_test]

        print("\nFeature Contribution to Misclassification (Avg Diff: Correct - Misclassified):")
        for feature in feature_columns:
            if feature in df.columns:
                 try:
                     avg_correct = df_correct[feature].mean()
                     avg_misclassified = df_misclassified[feature].mean()

                     if pd.notna(avg_correct) and pd.notna(avg_misclassified):
                         feature_diffs[feature] = avg_correct - avg_misclassified
                         print(f"- {feature}: {feature_diffs[feature]:.4f}")
                     else:
                         print(f"- {feature}: Could not calculate difference (NaN result).")
                 except Exception as e:
                     print(f"- {feature}: Error calculating difference - {e}")
            else:
                 print(f"- {feature}: Not found in test DataFrame columns.")

        # Plot feature impact
        if feature_diffs:
            sorted_impact = dict(sorted(feature_diffs.items(), key=lambda item: abs(item[1]), reverse=True))
            plt.figure(figsize=(10, 6))
            plt.bar(sorted_impact.keys(), sorted_impact.values())
            plt.title(f"Feature Impact on Misclassification ({clf_name})")
            plt.ylabel("Avg Difference (Correct - Misclassified)")
            plt.xlabel("Feature")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"feature_impact_{clf_name}.png")
            plt.close()
            print(f"Feature impact plot saved to feature_impact_{clf_name}.png")

    # Convert DataFrame to list of dicts for JSON compatibility
    results = {
        "most_common_errors": most_common_errors_df.to_dict(orient="records"),
        "similar_misclassified_pairs": similar_misclassified_pairs,
        "feature_diffs": feature_diffs
    }

    # Ensure all numeric types are JSON serializable (handles numpy types)
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp,)):
            return obj.isoformat()
        return obj

    if output_file is None:
        output_file = f"error_analysis_results_{clf_name}.json"
    try:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4, default=convert_to_serializable)
        print(f"Error analysis results saved to {output_file}")
    except Exception as e:
        print(f"Error saving error analysis results to JSON: {e}")

    return most_common_errors_df, similar_misclassified_pairs, feature_diffs


### DRIVER ###

def main():
    # Download necessary NLTK resources (only needed once)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

    # 1. Load Data
    PREPROCESSED_FILE = "processed_data.csv"

    if os.path.exists(PREPROCESSED_FILE):
        print(f"Loading preprocessed data from {PREPROCESSED_FILE}...")
        data = pd.read_csv(PREPROCESSED_FILE)
    else:
        print("Preprocessed data not found. Running full preprocessing...")
        data = load_and_preprocess_data(
            DATA_DIR, LYRICS_SUBDIR, GENRES_FILE, METADATA_FILE, LANG_FILE, TAGS_FILE, subset_size=SUBSET_SIZE, output_csv=PREPROCESSED_FILE
        )

    if data is None:
        exit()


    # --- OPTIONAL: genre distribution ---
    analyze_genre_distribution(data, 'genres', top_n=20)

    # Define top genres
    TOP_N_GENRES = ['rock', 'pop', 'electronic', 'metal', 'punk', 'folk', 'soul', 'hip hop'] # Example: Top 8

    # Filter dataset
    filtered_data = filter_dataset_for_top_genres(data, 'genres', TOP_N_GENRES)
    analyze_genre_distribution(filtered_data, 'genres', top_n=20)
    data = filtered_data
    # ------------------------------------


    # 2. Feature Extraction 
    data = extract_features(data)

    feature_columns = [
        "rhyme_density",
        "lexical_complexity",
        "sentiment_polarity",
        "sentiment_subjectivity",
        "readability",
    ]

    # 3. Train/Test Split
    X_train_hc, X_test_hc, X_train_dl, X_test_dl, y_train, y_test, train_idx, test_idx = create_train_test_split(data, feature_columns, GENRE_COLUMN)
    # Combine feature sets for training
    X_train = np.hstack([X_train_dl, X_train_hc])
    X_test = np.hstack([X_test_dl, X_test_hc])

    # Create DataFrame subset for the test data using original indices
    test_df = data.iloc[test_idx].reset_index(drop=True)

    # 4. Model Training and Evaluation
    classifiers_models = {}
    all_metrics = {}
    
    print("\n--- Training SVM ---")
    svm_model_path = "svm_model.joblib"
    if os.path.exists("svm_model.joblib"):
        svm_model, svm_mlb = joblib.load(svm_model_path)
        print("Loaded existing SVM model.")
    else:
        svm_model, svm_mlb = train_svm(X_train, y_train)
    svm_metrics = evaluate_model(svm_model, X_test, y_test, svm_mlb)
    all_metrics["SVM"] = svm_metrics
    classifiers_models["SVM"] = (svm_model, svm_mlb)

    print("\n--- Training Logistic Regression ---")
    lr_model_path = "logistic_regression_model.joblib"
    if os.path.exists("logistic_regression_model.joblib"):
        lr_model, lr_mlb = joblib.load(lr_model_path)
        print("Loaded existing Logistic Regression model.")
    else:
        lr_model, lr_mlb = train_logistic_regression(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test, lr_mlb)
    all_metrics["LogisticRegression"] = lr_metrics
    classifiers_models["LogisticRegression"] = (lr_model, lr_mlb)

    print("\n--- Training Random Forest ---")
    rf_model_path = "random_forest_model.joblib"
    if os.path.exists("random_forest_model.joblib"):
        rf_model, rf_mlb = joblib.load(rf_model_path)
        print("Loaded existing Random Forest model.")
    else:
        rf_model, rf_mlb = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, rf_mlb)
    all_metrics["RandomForest"] = rf_metrics
    classifiers_models["RandomForest"] = (rf_model, rf_mlb)

    with open("all_classifier_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)
    print("All classifier metrics saved to all_classifier_metrics.json")
    
    # 5. Summary Table and Visualization 
    print("\n--- Classifier Performance Summary ---")
    metrics_table = pd.DataFrame.from_dict(all_metrics, orient='index')
    print(metrics_table[['accuracy', 'hamming_loss', 'f1_micro', 'f1_macro', 'jaccard']])
     
    plt.figure(figsize=(8, 5))
    metrics_table['accuracy'].plot(kind='bar')
    plt.title('Classifier Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('classifier_accuracy_comparison.png')
    plt.show()
    print("Classifier accuracy comparison chart saved as 'classifier_accuracy_comparison.png'")
    
    # 6. Feature Ablation
    classifier_types = ["SVM", "LogisticRegression", "RandomForest"]
    for clf in classifier_types:
        print(f"\n--- Performing Feature Ablation on {clf} ---")
        perform_feature_ablation(data, feature_columns, GENRE_COLUMN, model_type=clf)

    # 7. Error Analysis
    classifiers = {
        "SVM": svm_model,
        "LogisticRegression": lr_model,
        "RandomForest": rf_model
    }
    print("\n--- Performing Automated Error Analysis ---")

    for clf_name, (clf_model, clf_mlb) in classifiers_models.items():
        print(f"\n--- Analyzing errors for {clf_name} ---")
        analyze_errors(
            model=clf_model,
            X_test=X_test,
            y_test=y_test,
            df=test_df,
            feature_columns=feature_columns,
            mlb=clf_mlb,
            output_file=f"error_analysis_{clf_name}.json",
            clf_name=clf_name
        )

if __name__ == "__main__":
    main()