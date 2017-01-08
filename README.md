# character-meta
Semester project @ DHLAB, EPFL on metadata extraction on characters in French XIXth century literature

*Based on work by Cyril Bornet from DHLAB*

Assuming the pickled data is available for a given book (check in `books-txt/predicted-data/` if not, run `characterStats.py`), the code is run from the `metaStats.py` file, by giving it the book title and predictors to run, e.g.

      python metaStats.py --book=aubonheurdesdames --job --gender --sentiment

This will compute the predictions for a variety of settings on these predictors and save the results in a `.csv` file in `metadata/`.

For the predictor code, see `computeMeta.py`.

In `results.ipynb`, a *Jupyter Notebook*, we plotted the results of these predictions, as shown in the report.

## Python dependencies

This project was always run in a Python 2.7 **virtual environment**.

The list of dependencies and versions are provided in `requirements.txt`. To install them, run

      pip install -r requirements.txt

### hunspell
      sudo apt-get update
    sudo apt-get install python2.7-dev
    sudo apt-get install libhunspell-dev
    sudo pip install hunspell

### treetagger3
To install treetagger3, see the instructions on the [website](http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/)

Additionally, we need

      sudo pip install treetaggerwrapper

### treetagger-python

Configure this by updating

      os.environ["TREETAGGER_HOME"]

in `characterStats.py`
