# DL-and-NN

A multilayer perceptron (MLP) built from scratch in NumPy, with an interactive Streamlit GUI, that classifies penguin species.

## Overview

This project implements a fully-connected neural network without any deep-learning
framework — the forward pass, backpropagation, and weight updates are all written
directly with NumPy in `mlp.py`. The model is trained on the penguins dataset to
predict one of three species (Adelie, Chinstrap, Gentoo) from five numeric features.

It was built as a neural-networks course assignment. A Streamlit app (`main.py`)
wraps the model in a GUI so you can configure the network and hyperparameters,
train it, and inspect the results visually.

## Features

- MLP implemented from scratch in NumPy — feedforward, backpropagation, and weight updates (`mlp.py`).
- Configurable architecture: any number and size of hidden layers.
- Choice of **sigmoid** or **tanh** activation, with an optional bias term.
- Adjustable learning rate and number of epochs.
- Interactive Streamlit GUI (`main.py`): set hyperparameters, train, and view a live training-accuracy curve (Plotly) and a confusion matrix (seaborn).
- Reports train and test accuracy after training.
- Data preprocessing with pandas / scikit-learn: mean imputation of missing values, label encoding, standard scaling, and a 30-samples-per-class train/test split.
- A Jupyter notebook (`mlp.ipynb`) version for experimentation.

## Tech stack

- **Python**
- **NumPy** — the neural network itself
- **pandas**, **scikit-learn** — data loading, preprocessing, and metrics
- **Streamlit** — the interactive GUI
- **Plotly**, **seaborn**, **matplotlib** — plots

## Getting started

Prerequisites: Python 3.

```bash
# (optional) create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Usage

Run the Streamlit app:

```bash
streamlit run main.py
```

This opens the GUI in your browser, where you can set the hidden-layer sizes
(e.g. `10` or `10,5,3`), learning rate, epochs, activation function, and bias,
then train the model and view the accuracy curve and confusion matrix.

To explore the model step by step instead, open the notebook:

```bash
jupyter notebook mlp.ipynb
```

## Project structure

```
mlp.py          # MLP class: feedforward, backpropagation, weight updates
main.py         # Streamlit GUI: preprocessing, training, and visualizations
mlp.ipynb       # Notebook version for experimentation
penguins.csv    # Dataset (3 species, 5 numeric features)
requirements.txt
```
