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