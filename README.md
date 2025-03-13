# Lyrics-based-auto-tagging

This repository contains the lyrics-based auto-tagging project.


## Installation of dependencies

Python version used was Python 3.12.9

We recommend the use of a virtual environment for dealing with the dependencies of this project. The Python libraries necessary for the project are included in the <code>requirements.txt</code>. You could use [Conda](https://anaconda.org/anaconda/conda), or another virtual environment handler of your choosing.

One of the many sequence of steps is as follows:
1. Prepare your virtual environment of choosing. Create your virtual environment. Ensure pip is installed.
2. Use <code>pip install -r requirements.txt</code> (or an equivalent command)

## program.py

requires a local copy of the music4all dataset
    - make sure to change the path to the data

currently just creates a pandas dataframe using csv's and lyric folder from music4all dataset, 
but has commented out code from Google's AI Gemini 2.0 Pro Experimental model outlining:
    - handcrafted features
    - classifiers

when troubleshooting, set the subset_size to a small number, as running it on the entire database takes about 5 minutes on my desktop.

### Experimental Branch Summary

- Modifications align the implementation more closely with the **original project proposal**, particularly in how genre classification is approached using both **handcrafted features** and **deep learning-based embeddings**. 

- The objective of this project is to develop a **lyric-based music auto-tagging system** that can classify songs by **genre** and potentially other characteristics. Unlike most previous works that rely on **audio-based features**, this project explores **whether lyrical content alone** can provide robust classification results and, if not, explores why this is the case. 

- This implementation adds a **structured evaluation of feature effectiveness**, distinction in analyzing **handcrafted versus automatically learned features**, and **systematically testing** the role of different types of features, improving **alignment with the proposal**.

---

#### **How This Branch Aligns with Project Goals**

Breakdown of **how these changes improve alignment**:

1. **Feature Extraction Refinement** → *Aligns with proposal goal: Evaluating handcrafted vs. deep learning features separately.*
   - **Why change it?** The original version lacked a distinction between **handcrafted** (rhyme density, sentiment, readability) and **deep learning** (Word2Vec, FastText, BERT) features.
   - **How does it improve alignment?** Now, each type of feature is **explicitly separated**, allowing us to **directly test their relative importance**.

2. **Feature Ablation Testing** → *Aligns with proposal goal: Identifying the most impactful features for classification.*
   - **Why change it?** The original approach trained models without systematically **evaluating which features contributed most** to classification success or failure.
   - **How does it improve alignment?** Feature ablation now **removes individual features and measures performance drop**, allowing us to **quantify their importance**.

3. **Automated Error Analysis** → *Aligns with proposal goal: Understanding the limitations of lyric-based classification.*
   - **How does it improve alignment?** Error analysis now **automatically detects systematic misclassifications** using **TF-IDF cosine similarity**, making it possible to analyze **why lyrics fail to classify correctly** without manual labeling.

4. **Pre-Trained Word Embeddings** → *Aligns with proposal goal: Enhancing efficiency and reproducibility.*
   - **How does it improve alignment?** We now **load pre-trained embeddings** (e.g., Google’s Word2Vec) to ensure **faster computation and standardized results**.

5. **Separation of Feature-Based Models** → *Aligns with proposal goal: Determining whether handcrafted or deep learning features perform better.*
   - **Why change it?** The original script **mixed handcrafted and deep learning features** in training, making it **impossible to isolate their effects**.
   - **How does it improve alignment?** Models are now **trained separately** using **only handcrafted features, only deep learning features, and both together**—allowing us to **directly compare their effectiveness**.

6. **Saving Feature Ablation Results to CSV** → *Aligns with proposal goal: Ensuring reproducibility and documentation of findings.*
   - **How does it improve alignment?** Results are now **saved in CSV format**, ensuring that findings can be used for **further analysis and reporting**.

---

This branch is designed for **evaluation and discussion** before merging into the main repository. Please review these changes and assess how they align with the broader goals of the project.