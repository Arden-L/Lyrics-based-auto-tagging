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
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # For progress bars
import pronouncing  # For rhyme detection
import textstat  # For readability
from textblob import TextBlob
from gensim.models import Word2Vec, FastText, KeyedVectors
from transformers import BertTokenizer, BertModel
import torch

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
    """Calculates lexical complexity (unique words / total words), handling empty tokens."""
    if not tokens or len(tokens) == 0:
        return 0.0
    return len(set(tokens)) / len(tokens)


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


def load_pretrained_word2vec():
    """Loads pre-trained Word2Vec embeddings."""
    return KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

def generate_word2vec_embeddings(texts, model):
    """Uses pre-trained Word2Vec embeddings and handles empty text cases."""
    embeddings = []
    for text in texts:
        words = text.split() if isinstance(text, str) else []
        word_vectors = [model[word] for word in words if word in model]
        embeddings.append(np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size))
    return embeddings

def generate_fasttext_embeddings(texts, vector_size=100, window=5, min_count=1):
    """Generates FastText embeddings from lyrics."""
    tokenized_texts = [text.split() for text in texts]
    model = FastText(sentences=tokenized_texts, vector_size=vector_size, window=window, min_count=min_count)
    return [model.wv[text].mean(axis=0) if text else np.zeros(vector_size) for text in tokenized_texts]

def generate_bert_embeddings(texts, tokenizer, model):
    """Generates BERT embeddings from lyrics using a pre-loaded model."""
    embeddings = []

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())

    return embeddings

def extract_features(df):
    """Extracts handcrafted and deep learning-based features from lyrics."""
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

    tqdm.pandas(desc="Generating Word2Vec Embeddings")
    word2vec_model = load_pretrained_word2vec()
    df["word2vec_embedding"] = generate_word2vec_embeddings(df[TEXT_COLUMN].tolist(), word2vec_model)

    tqdm.pandas(desc="Generating FastText Embeddings")
    df["fasttext_embedding"] = generate_fasttext_embeddings(df[TEXT_COLUMN].tolist())

    tqdm.pandas(desc="Generating BERT Embeddings")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")

    df["bert_embedding"] = generate_bert_embeddings(df[TEXT_COLUMN].tolist(), tokenizer, bert_model)

    df = df.drop(columns=["tokens"])  # Drop tokens after use
    return df


# --- 3. Model Training and Evaluation ---
def create_train_test_split(df, features, target):
    """Splits data into training and testing sets, handling list-like genres."""
    df['genres_str'] = df[target].apply(lambda x: ','.join(x) if isinstance(x, list) else str(x))

    # Separate handcrafted and deep learning features
    X_handcrafted = df[features].values
    X_deep_learning = np.hstack([
        np.stack(df["word2vec_embedding"].values),
        np.stack(df["fasttext_embedding"].values),
        np.stack(df["bert_embedding"].values)
    ])
    
    y = df['genres_str']
    X_train_hc, X_test_hc, y_train, y_test = train_test_split(X_handcrafted, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    X_train_dl, X_test_dl, _, _ = train_test_split(X_deep_learning, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    return X_train_hc, X_test_hc, X_train_dl, X_test_dl, y_train, y_test


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
    """Evaluates a model and returns structured results."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)  # Structured output
    confusion = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion)

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": confusion
    }

def perform_feature_ablation(df, feature_columns, target_column, model_type="SVM"):
    """Performs feature ablation testing by systematically removing features and measuring performance impact."""
    base_X_train_hc, base_X_test_hc, base_X_train_dl, base_X_test_dl, base_y_train, base_y_test = create_train_test_split(df, feature_columns, target_column)
    # For ablation, combine both feature sets
    base_X_train = np.hstack([base_X_train_dl, base_X_train_hc])
    base_X_test = np.hstack([base_X_test_dl, base_X_test_hc])

    # Choose a model type
    if model_type == "SVM":
        model = train_svm
    elif model_type == "NaiveBayes":
        model = train_naive_bayes
    elif model_type == "LogisticRegression":
        model = train_logistic_regression
    else:
        raise ValueError("Invalid model type. Choose from 'SVM', 'NaiveBayes', or 'LogisticRegression'.")

    # Train on full feature set
    base_model = model(base_X_train, base_y_train)
    base_accuracy, _, _ = evaluate_model(base_model, base_X_test, base_y_test)
    print(f"Base Accuracy with all features: {base_accuracy:.4f}")

    # Perform feature ablation
    ablation_results = []
    for feature in feature_columns:
        reduced_features = [f for f in feature_columns if f != feature]
        X_train_hc, X_test_hc, X_train_dl, X_test_dl, y_train, y_test = create_train_test_split(df, reduced_features, target_column)
        X_train = np.hstack([X_train_dl, X_train_hc])
        X_test = np.hstack([X_test_dl, X_test_hc])

        # Train and evaluate
        ablated_model = model(X_train, y_train)
        ablated_accuracy, _, _ = evaluate_model(ablated_model, X_test, y_test)

        # Measure accuracy drop
        accuracy_drop = base_accuracy - ablated_accuracy
        ablation_results.append([feature, accuracy_drop])
        print(f"Removing '{feature}' decreased accuracy by {accuracy_drop:.4f}")

    # Save results to CSV
    ablation_df = pd.DataFrame(ablation_results, columns=["Feature", "Accuracy Drop"])
    ablation_df.to_csv("feature_ablation_results.csv", index=False)
    print("Feature ablation results saved to feature_ablation_results.csv.")

    return ablation_df

def analyze_errors(model, X_test, y_test, df, feature_columns):
    """Analyzes systematic misclassifications without manual review."""
    y_pred = model.predict(X_test)

    # Find misclassified instances
    misclassified = y_test[y_test != y_pred]
    misclassified_indices = misclassified.index

    print(f"\nTotal Misclassifications: {len(misclassified)}")

    # Identify most common misclassifications
    error_counts = pd.DataFrame({'True': y_test[misclassified_indices], 'Predicted': y_pred[misclassified_indices]})
    most_common_errors = error_counts.groupby(['True', 'Predicted']).size().reset_index(name='count')
    print("\nMost Common Misclassifications:")
    print(most_common_errors.sort_values(by='count', ascending=False).head(10))

    # Compute similarity of misclassified samples
    misclassified_texts = df.loc[misclassified_indices, TEXT_COLUMN]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(misclassified_texts)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    def adaptive_similarity_threshold(sim_matrix):
        """Compute adaptive similarity threshold based on dataset distribution."""
        flat_values = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
        return np.percentile(flat_values, 90)

    # Compute adaptive threshold
    threshold = adaptive_similarity_threshold(similarity_matrix)

    # Identify highly similar misclassified examples
    similar_misclassified_pairs = []
    for i in range(len(misclassified_indices)):
        for j in range(i + 1, len(misclassified_indices)):
            if similarity_matrix[i, j] > threshold:
                similar_misclassified_pairs.append((misclassified_indices[i], misclassified_indices[j]))

    print(f"\nHighly Similar Misclassified Pairs: {len(similar_misclassified_pairs)}")
    
    # Feature impact on misclassification
    feature_diffs = {}
    for feature in feature_columns:
        avg_misclassified_value = df.loc[misclassified_indices, feature].mean()
        avg_correct_value = df.loc[df.index.difference(misclassified_indices), feature].mean()
        feature_diffs[feature] = avg_correct_value - avg_misclassified_value

    print("\nFeature Contribution to Misclassification:")
    for feature, impact in sorted(feature_diffs.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"{feature}: {impact:.4f}")

    return most_common_errors, similar_misclassified_pairs, feature_diffs


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
        DATA_DIR, LYRICS_SUBDIR, GENRES_FILE, METADATA_FILE, LANG_FILE, TAGS_FILE, subset_size=None, output_csv="processed_data.csv"
    )

    if data is None:
        exit()

    # --- 2. Feature Extraction ---
    data = extract_features(data)

    # --- 3. Define Features and Target ---
    feature_columns = [
        "rhyme_density",
        "lexical_complexity",
        "sentiment_polarity",
        "sentiment_subjectivity",
        "readability",
    ]

    # --- 4. Train/Test Split ---
    X_train_hc, X_test_hc, X_train_dl, X_test_dl, y_train, y_test = create_train_test_split(data, feature_columns, GENRE_COLUMN)
    # Combine feature sets for training
    X_train = np.hstack([X_train_dl, X_train_hc])
    X_test = np.hstack([X_test_dl, X_test_hc])

    # --- 5. Model Training and Evaluation ---
    print("\n--- Training SVM ---")
    svm_model = train_svm(X_train, y_train)
    svm_accuracy, _, _ = evaluate_model(svm_model, X_test, y_test)

    print("\n--- Performing Feature Ablation on SVM ---")
    perform_feature_ablation(data, feature_columns, GENRE_COLUMN, model_type="SVM")

    print("\n--- Performing Automated Error Analysis ---")
    analyze_errors(svm_model, X_test, y_test, data, feature_columns)

if __name__ == "__main__":
    main()