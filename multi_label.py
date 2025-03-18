import os
import re
import string
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, hamming_loss, jaccard_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pronouncing
import textstat
from textblob import TextBlob

# --- Constants ---
DATA_DIR = r"C:\Users\nikol\Documents\School\MIR\project\music4all"  # !!! CHANGE THIS !!!
LYRICS_SUBDIR = "lyrics"
GENRES_FILE = "id_genres.csv"
# METADATA_FILE = "id_metadata.csv"
LANG_FILE = "id_lang.csv"
# TAGS_FILE = "id_tags.csv"
LYRICS_ID_COLUMN = "id"
TEXT_COLUMN = "lyrics"
GENRE_COLUMN = "genres"
# TAG_COLUMN = 'tags'
TEST_SIZE = 0.2
RANDOM_STATE = 42
SUBSET_SIZE = None  # Use None for the full dataset
OUTPUT_CSV = "processed_data_subset.csv" # Original output for initial loading
PROCESSED_DATA_FILE = "processed_features.csv"  # New file to save processed features

# --- 1. Data Loading and Preprocessing ---
def load_and_preprocess_data(data_dir, lyrics_subdir, genres_file, lang_file, subset_size=None, output_csv=None):
    """Loads, preprocesses, and combines data, efficiently handling subsets."""
    try:
        # 1. Load Language Data and Filter for English Songs
        lang_path = os.path.join(data_dir, lang_file)
        lang_df = pd.read_csv(lang_path, delimiter='\t', header=0, names=[LYRICS_ID_COLUMN, 'lang'])
        english_ids = lang_df[lang_df['lang'] == 'en'][LYRICS_ID_COLUMN].unique()

        # 2. Apply Subset (if specified) *BEFORE* loading lyrics
        if subset_size is not None:
            rng = np.random.default_rng(RANDOM_STATE)  # Use a consistent random number generator
            english_ids = rng.choice(english_ids, size=subset_size, replace=False)
            print(f"Using a subset of {subset_size} English songs.")

        english_ids_df = pd.DataFrame({LYRICS_ID_COLUMN: english_ids})

        # 3. Load *ONLY* the required lyrics files
        lyrics_path = os.path.join(data_dir, lyrics_subdir)
        lyrics_data = []
        for track_id in tqdm(english_ids, desc="Loading Lyrics"):  # Iterate over the subset IDs!
            filename = f"{track_id}.txt"
            filepath = os.path.join(lyrics_path, filename)
            if os.path.exists(filepath): # Check if file exists. CRUCIAL!
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        lyrics = f.read()
                    lyrics_data.append({LYRICS_ID_COLUMN: str(track_id), TEXT_COLUMN: lyrics}) # Keep consistent type
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
                    # No continue here.  If the file exists and we can't read it, we want to know.
            else:
                print(f"Warning: Lyrics file not found for track ID: {track_id}") #Important warning

        lyrics_df = pd.DataFrame(lyrics_data)
        # No type conversion needed if we add track_id as string above

        # 4. Merge to keep only selected songs (no change from here on)
        lyrics_df = pd.merge(english_ids_df, lyrics_df, on=LYRICS_ID_COLUMN, how='inner')

        # 5. Load Genres
        genres_path = os.path.join(data_dir, genres_file)
        genres_df = pd.read_csv(genres_path, delimiter='\t', header=0, names=[LYRICS_ID_COLUMN, "genres_raw"])
        genres_df[LYRICS_ID_COLUMN] = genres_df[LYRICS_ID_COLUMN].astype(str)
        genres_df['genres'] = genres_df['genres_raw'].str.split(',')
        genres_df = genres_df.drop(columns=['genres_raw'])

        # 6. Merge Lyrics and Genres
        merged_df = pd.merge(lyrics_df, genres_df, on=LYRICS_ID_COLUMN, how="left")
        merged_df['genres'] = merged_df['genres'].apply(lambda x: x if isinstance(x, list) else [])

        # 7. Load Metadata
        # metadata_path = os.path.join(data_dir, metadata_file)
        # metadata_df = pd.read_csv(metadata_path, delimiter='\t', header=0)
        # metadata_df[LYRICS_ID_COLUMN] = metadata_df[LYRICS_ID_COLUMN].astype(str)

        # # 8. Load Tags
        # tags_path = os.path.join(data_dir, tags_file)
        # tags_df = pd.read_csv(tags_path, delimiter='\t', header=0, names=[LYRICS_ID_COLUMN, TAG_COLUMN])
        # tags_df[LYRICS_ID_COLUMN] = tags_df[LYRICS_ID_COLUMN].astype(str)

        # 9. Final Merge
        # merged_df = pd.merge(merged_df, metadata_df, on=LYRICS_ID_COLUMN, how="left")
        # merged_df = pd.merge(merged_df, tags_df, on=LYRICS_ID_COLUMN, how="left")

        #SAVE
        if output_csv:
            try:
                merged_df.to_csv(output_csv, index=False, encoding='utf-8')
                print(f"DataFrame saved to {output_csv}")
            except Exception as e:
                print(f"Error saving DataFrame to CSV: {e}")

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


# --- 2. Feature Engineering ---

def select_top_n_genres(df, genre_column='genres', top_n=10):
    """Selects the top N most frequent genres and filters the DataFrame."""
    all_genres = [genre for sublist in df[genre_column] for genre in sublist]
    genre_counts = pd.Series(all_genres).value_counts()
    top_genres = genre_counts.head(top_n).index.tolist()
    print(f"Top {top_n} genres: {top_genres}")

    # Filter the DataFrame to include only the top genres
    df[genre_column] = df[genre_column].apply(lambda x: [genre for genre in x if genre in top_genres])
    return df, top_genres


def preprocess_lyrics(text):
    """Lowercase, remove punctuation, tokenize, remove stopwords, lemmatize."""
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
    """Calculates rhyme density efficiently and handles errors."""
    if not tokens:
        return 0.0

    rhyme_count = 0
    total_count = 0
    rhymed_words = set()  # Keep track of words we've already counted rhymes for

    for word in tokens:
        if word in rhymed_words:
            continue  # Skip if we've already checked this word

        try:
            rhymes = pronouncing.rhymes(word)
            if rhymes:
                rhyme_count += len(set(rhymes) & set(tokens))  # Efficient intersection
                rhymed_words.add(word)  # Add the word to the set
        except KeyError:
            pass  # Word not in pronouncing dictionary, ignore
        except Exception:
            return 0

    # Avoid division by zero, and return 0 if we only have 1 word.
    return rhyme_count / len(tokens) if len(tokens) > 1 else 0.0

def calculate_lexical_complexity(tokens):
    """Calculates lexical complexity (unique words / total words)."""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens) if tokens else 0.0

def calculate_sentiment(text):
    """Calculates sentiment polarity and subjectivity using TextBlob."""
    from textblob import TextBlob  # Import within the function
    if not isinstance(text, str) or not text:
        return 0.0, 0.0
    analysis = TextBlob(text)
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity

def calculate_readability(text):
    """Calculates Flesch Reading Ease score."""
    if not isinstance(text, str) or not text:
        return 0.0
    try:
        return textstat.flesch_reading_ease(text)
    except Exception:
        return 0.0 #Handle potential errors

def extract_features(df):
    """Extracts features and adds them as new columns."""
    tqdm.pandas(desc="Preprocessing Lyrics")
    df["tokens"] = df[TEXT_COLUMN].progress_apply(preprocess_lyrics)

    tqdm.pandas(desc="Calculating Rhyme Density")
    df["rhyme_density"] = df["tokens"].progress_apply(calculate_rhyme_density)

    tqdm.pandas(desc="Calculating Lexical Complexity")
    df["lexical_complexity"] = df["tokens"].progress_apply(calculate_lexical_complexity)

    tqdm.pandas(desc="Calculating Sentiment")
    df[["sentiment_polarity", "sentiment_subjectivity"]] = df[TEXT_COLUMN].progress_apply(
        calculate_sentiment).tolist()

    tqdm.pandas(desc="Calculating Readability")
    df["readability"] = df[TEXT_COLUMN].progress_apply(calculate_readability)

    df = df.drop(columns=["tokens"])  # Drop tokens after use
    return df

# --- 3. Model Training and Evaluation ---

def one_hot_encode_genres(df, genre_list, genre_column='genres'):
    """One-hot encodes a pre-defined list of genres."""

    for genre in genre_list:
        df[genre] = df[genre_column].apply(lambda x: 1 if genre in x else 0)
    df = df.drop(columns=[genre_column])
    return df

def create_multilabel_train_test_split(X, y, test_size=0.2, random_state=42):
    """Performs a stratified train/test split for multi-label data."""
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_index, test_index = next(msss.split(X, y))

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    return X_train, X_test, y_train, y_test


def train_multilabel_naive_bayes(X_train, y_train):
    """Trains a multi-label Naive Bayes classifier."""
    model = MultiOutputClassifier(MultinomialNB())
    model.fit(X_train, y_train)
    return model

def train_multilabel_svm(X_train, y_train):
    model = MultiOutputClassifier(SVC(kernel='linear', probability=True, random_state=RANDOM_STATE, C=10))
    model.fit(X_train, y_train)
    return model

def train_multilabel_logistic_regression(X_train, y_train):
    """Trains a multi-label Logistic Regression classifier."""
    model = MultiOutputClassifier(LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, solver='liblinear'))
    model.fit(X_train, y_train)
    return model

def train_multilabel_random_forest(X_train, y_train):
    model = MultiOutputClassifier(RandomForestClassifier(random_state=RANDOM_STATE))
    model.fit(X_train, y_train)
    return model


def evaluate_multilabel_model(model, X_test, y_test, genres):
    """Evaluates a multi-label model."""
    y_pred = model.predict(X_test)

    print(f"Hamming Loss: {hamming_loss(y_test, y_pred):.4f}")
    print(f"Jaccard Score: {jaccard_score(y_test, y_pred, average='samples'):.4f}") #samples is good
    print("\nClassification Report (per label):\n", classification_report(y_test, y_pred, target_names=genres, zero_division=0))




# --- Main Execution ---
if __name__ == "__main__":
#    Download necessary NLTK resources (only needed once)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt') # Download ALL of punkt

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')



    # --- 1. Load or Load Processed Data ---
    if os.path.exists(PROCESSED_DATA_FILE):
        print(f"Loading processed data from {PROCESSED_DATA_FILE}...")
        data = pd.read_csv(PROCESSED_DATA_FILE, encoding='utf-8') # Ensure correct encoding
        # Re-convert genres column from string back to list of strings (if saved as string in CSV)
        if 'genres' in data.columns: # Check if 'genres' column exists before trying to convert
            data['genres'] = data['genres'].apply(eval) # Use eval cautiously, ensure CSV is trusted or use safer parsing if genres are saved differently
        else:
            print("Warning: 'genres' column not found in loaded data. Ensure genres are correctly processed and saved.")


    else:
        print("Processed data file not found. Loading and processing data from scratch...")
        data = load_and_preprocess_data(
            DATA_DIR, LYRICS_SUBDIR, GENRES_FILE, LANG_FILE,
            subset_size=SUBSET_SIZE, output_csv=OUTPUT_CSV  # Save initial subset if needed
        )

        if data is None:
            exit()  # Exit if data loading failed

    # --- 2. Feature Extraction ---
        data = extract_features(data)

        # --- Save processed data to CSV ---
        try:
            data.to_csv(PROCESSED_DATA_FILE, index=False, encoding='utf-8') # Save AFTER feature extraction
            print(f"Processed data with features saved to {PROCESSED_DATA_FILE}")
        except Exception as e:
            print(f"Error saving processed data to CSV: {e}")

    # --- 2.5. Select Top N Genres ---
    data, genre_list = select_top_n_genres(data, top_n=20)


    # --- 3. One-Hot Encode Genres ---
    data = one_hot_encode_genres(data, genre_list)


     # --- 4. Define Features and Target ---
    feature_columns = [
        "rhyme_density",
        "lexical_complexity",
        "sentiment_polarity",
        "sentiment_subjectivity",
        "readability",
    ]

    X = data[feature_columns]
    y = data[genre_list]



    # --- 5. Train/Test Split (Multi-Label) ---
    X_train, X_test, y_train, y_test = create_multilabel_train_test_split(X, y)



    # --- Scale Features (AFTER splitting) ---
    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_columns)  # Use transform, not fit_transform



    # --- 6. Model Training and Evaluation ---
    print("\n--- Training Multi-Label Naive Bayes ---")
    nb_model = train_multilabel_naive_bayes(X_train, y_train)
    evaluate_multilabel_model(nb_model, X_test, y_test, genre_list)

    print("\n--- Training Multi-Label SVM ---")
    svm_model = train_multilabel_svm(X_train, y_train)
    evaluate_multilabel_model(svm_model, X_test, y_test, genre_list)

    print("\n--- Training Multi-Label Logistic Regression ---")
    lr_model = train_multilabel_logistic_regression(X_train, y_train)
    evaluate_multilabel_model(lr_model, X_test, y_test, genre_list)

    print("\n--- Training Multi-Label Random Forest ---")
    rf_model = train_multilabel_random_forest(X_train, y_train)
    evaluate_multilabel_model(rf_model, X_test, y_test, genre_list)
