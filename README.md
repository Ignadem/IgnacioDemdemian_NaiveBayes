# Naive Bayes Project

This repository is now set up with a working baseline classification pipeline using `GaussianNB` from `scikit-learn`.

The first version trains a model on the built-in breast cancer dataset, evaluates it, and saves the trained model under `models/`.

## Structure

- **`src/app.py`** → Runs the full training and evaluation pipeline.
- **`src/utils.py`** → Loads the baseline dataset and saves the trained model.
- **`src/explore.ipynb`** → Notebook for quick experimentation.
- **`models/`** → Output folder for trained models.
- **`data/`** → Reserved for future project datasets.

## Run the project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
python src/app.py
```

## Current behavior

When the script runs, it:

- splits the dataset into train and test sets,
- trains a `GaussianNB` model,
- prints accuracy, confusion matrix, and classification report,
- saves the model to `models/gaussian_nb_breast_cancer.joblib`.

## Natural next steps

From here we can evolve the project by:

- plugging in a real CSV file from `data/raw/`,
- adding preprocessing,
- comparing different Naive Bayes variants,
- exporting metrics and predictions as reproducible artifacts.
