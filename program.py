
# IMPORTS
import os
import re
import string
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # For progress bars
import pronouncing  # For rhyme detection
import textstat  # For readability
from textblob import TextBlob

# --- 1. Configuration and Setup ---


# --- Constants ---
DATA_DIR = r'C:\Users\nikol\Documents\School\MIR\project\music4all'  # !!! CHANGE THIS !!!
LYRICS_SUBDIR = "lyrics"
GENRES_FILE = "id_genres.csv"
METADATA_FILE = "id_metadata.csv"
LANG_FILE = "id_lang.csv"
TAGS_FILE = "id_tags.csv"
LYRICS_ID_COLUMN = "id"  # Consistent ID column name
TEXT_COLUMN = "lyrics"
GENRE_COLUMN = "genre"  # will become 'genres'
TAG_COLUMN = 'tags'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- Helper Functions ---

def load_and_preprocess_data(data_dir, lyrics_subdir, genres_file, metadata_file, lang_file, tags_file, subset_size=None, output_csv=None):
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

        # 7. Load Metadata
        metadata_path = os.path.join(data_dir, metadata_file)
        metadata_df = pd.read_csv(metadata_path, delimiter='\t', header=0)
        metadata_df[LYRICS_ID_COLUMN] = metadata_df[LYRICS_ID_COLUMN].astype(str)

        # 8. Load Tags
        tags_path = os.path.join(data_dir, tags_file)
        tags_df = pd.read_csv(tags_path, delimiter='\t', header=0, names=[LYRICS_ID_COLUMN, TAG_COLUMN])
        tags_df[LYRICS_ID_COLUMN] = tags_df[LYRICS_ID_COLUMN].astype(str)

        # 9. Final Merge
        merged_df = pd.merge(merged_df, metadata_df, on=LYRICS_ID_COLUMN, how="left")
        merged_df = pd.merge(merged_df, tags_df, on=LYRICS_ID_COLUMN, how="left")
        
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
    """Calculates rhyme density."""
    if not tokens:
        return 0.0
    rhyming_pairs = 0
    total_words = len(tokens)
    try:
        for i in range(total_words - 1):
            for j in range(i + 1, total_words):
                if tokens[i] in pronouncing.rhymes(tokens[j]):
                    rhyming_pairs += 1
    except KeyError:
        return 0
    except Exception: #Best practice to catch general exception
        return 0
    return (2 * rhyming_pairs) / total_words if total_words > 1 else 0.0


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
def create_train_test_split(df, features, target):
    """Splits data into training and testing sets, handling list-like genres."""
    # Convert genre lists to strings for train_test_split
    df['genres_str'] = df[target].apply(lambda x: ','.join(x) if isinstance(x, list) else str(x))

    X = df[features]
    y = df['genres_str']  # Use string representation for splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    return X_train, X_test, y_train, y_test


def train_naive_bayes(X_train, y_train):
    """Trains a Multinomial Naive Bayes classifier."""
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train):
    """Trains a Support Vector Machine classifier."""
    model = SVC(kernel='linear', probability=True, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train, y_train):
    """Trains a Logistic Regression classifier."""
    model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluates a model and prints metrics."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", confusion)

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
    return accuracy, report, confusion




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


    # --- 1. Load Data --- 
    data = load_and_preprocess_data(
        DATA_DIR, LYRICS_SUBDIR, GENRES_FILE, METADATA_FILE, LANG_FILE, TAGS_FILE, subset_size=None, output_csv= "processed_data.csv" #Still using subset!
    )

    if data is None:
        exit()

    # # --- 2. Feature Extraction ---
    # data = extract_features(data)

    # # --- 3. Define Features and Target ---
    # feature_columns = [
    #     "rhyme_density",
    #     "lexical_complexity",
    #     "sentiment_polarity",
    #     "sentiment_subjectivity",
    #     "readability",
    # ]

    # # --- 4. Train/Test Split ---
    # X_train, X_test, y_train, y_test = create_train_test_split(data, feature_columns, GENRE_COLUMN)

    # # --- 5. Model Training and Evaluation ---
    # print("\n--- Training Naive Bayes ---")
    # nb_model = train_naive_bayes(X_train, y_train)
    # nb_accuracy, _, _ = evaluate_model(nb_model, X_test, y_test)

    # print("\n--- Training SVM ---")
    # svm_model = train_svm(X_train, y_train)
    # svm_accuracy, _, _ = evaluate_model(svm_model, X_test, y_test)

    # print("\n--- Training Logistic Regression ---")
    # lr_model = train_logistic_regression(X_train, y_train)
    # lr_accuracy, _, _ = evaluate_model(lr_model, X_test, y_test)

    # # Print the shapes of your training and testing sets.
    # print("X_train shape:", X_train.shape)
    # print("X_test shape:", X_test.shape)
    # print("y_train shape:", y_train.shape)
    # print("y_test shape:", y_test.shape)

    # print("\n--- Summary ---")
    # print(f"Naive Bayes Accuracy: {nb_accuracy:.4f}")
    # print(f"SVM Accuracy: {svm_accuracy:.4f}")
    # print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")

if __name__ == "__main__":
    main()