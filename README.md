# Lyrics-Based Auto-Tagging

This project explores whether **lyrical content alone** can be used to classify songs by **genre** in a multi-label setting. We compare **handcrafted linguistic features** with **deep learning-based embeddings**, analyzing their individual and combined effectiveness using a structured classification pipeline.

## Project Goals

This project aimed to explore:
- Whether lyrics can support accurate multi-label genre classification.
- The contribution of handcrafted features (rhyme density, sentiment, readability, lexical complexity) vs. deep embeddings (Word2Vec, FastText, BERT).
- The causes of model failure via feature ablation and error analysis.
- The impact of genre label design (e.g., using top-N genres or amalgamating subgenres).

## Implementation Overview

### Feature Engineering
- **Handcrafted Features**:
  - Rhyme density (via CMUdict)
  - Lexical complexity (unique/total tokens)
  - Sentiment polarity & subjectivity (TextBlob)
  - Readability (Flesch Reading Ease via `textstat`)
- **Deep Learning Embeddings**:
  - Pretrained Word2Vec (Google News)
  - FastText (trained on dataset)
  - BERT (mean-pooled contextual embeddings)

### Classification Models
Models were trained under three configurations:
- Handcrafted-only
- Deep-only
- Combined

Classifiers include:
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest

All models use `OneVsRestClassifier` to support multi-label prediction.

### Evaluation & Analysis
- **Feature Ablation**: Assesses accuracy impact when removing individual handcrafted features.
- **Error Analysis**: Identifies common genre confusions and quantifies feature differences in errors.
- **Genre Limiting**: Evaluates performance using top-5/top-8 genres and amalgamated genre categories.
- **Performance Metrics**: Accuracy, F1 (macro/micro), Jaccard score, and Hamming loss.

## Results Summary

Handcrafted features alone offered limited predictive value, but combining them with deep embeddings improved performance. Readability was the most impactful handcrafted feature, particularly in error separation. Genre amalgamation and label space reduction significantly improved classification metrics on a subset of data. Random Forest achieved the highest exact match accuracy on the entire dataset (~19%), while SVM and Logistic Regression yielded stronger partial-label metrics (e.g., micro-F1 ≈ 0.35).

## Setup

- Python version: `3.12.9`
- Use a virtual environment (e.g., Conda or venv)
- Install requirements:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset Requirements

This project requires a **local copy of the music4all dataset**.

**Before running:**
- Update the dataset path in `lyric_based_classification.py`
- Ensure the following files exist in the dataset folder:
  - Metadata TSVs (`genres.tsv`, `metadata.tsv`, `language.tsv`, `tags.tsv`)
  - Lyrics directory with individual `.txt` files

## Running the Program

The main logic is inside `lyric_based_classification.py`. It:
- Loads and preprocesses the dataset
- Extracts and caches features
- Trains and evaluates models
- Conducts feature ablation and error analysis
- Outputs metrics, plots, and CSVs for reproducibility

**Tip for testing**: Use `subset_size` option to reduce runtime during development.

## Output Artifacts

- `all_classifier_metrics.json` – Summary of all evaluation results
- `feature_ablation_results.csv` – Accuracy drops per ablated feature
- `*.png` – Visual comparisons (accuracy, error heatmaps, feature impact)
- `error_analysis_*.json` – Feature differences and common misclassifications
- Cached files: `*.joblib` and `*.csv`feature representations