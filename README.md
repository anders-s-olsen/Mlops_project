# DTU-mlops group 24 - Vision transformers for MNIST classification

### Overall goal of the project
In this project we will focus on MNIST classification using vision transformers.

### What framework are you going to use
We will use the transformer pytorch ecosystem.

### How do you intend to include the framework in your project
Vision transformers are implemented in the python package `vit-pytorch` which we will use for MNIST classification. These vision transformers will be added as an extension to the MNIST classification codebase already developed during the course. Thus, we will utilize other frameworks in conjunction with pytorch, including cookie-cutter, weights and biases, data version control, etc.

### What data are you going to run on
We will use the standard MNIST data set as included in `torchvision` datasets. The dataset contains roughly 50.000 handwritten digits between 0-9 for training and 10.000 images for testing.

### What deep learning models do you expect to use
Transformers from the 2017-paper "Attention is all you need" have evolved to be near-SOTA models for a wide variety of applications. Transformers were initially developed for text or time-series analysis but have recently been extended to 2D data with the vision transformers introduced in the 2020-paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale".


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
