# Artificial Neural Networks — Regression & Classification

This repo has two notebooks where I built ANNs from scratch using PyTorch, 
one for a regression problem and one for a multi-class classification problem. Both follow the full pipeline
— loading data, preprocessing, building the model, training, and evaluating.

## What's Inside
```
├── ANN_regression.ipynb          # Power plant energy output prediction
├── ANN_classification.ipynb      # Date fruit variety classification
├── powerplant_data.csv           # Dataset for regression
├── DateFruit_Dataset.csv         # Dataset for classification
```

## Notebook 1 — Regression (Power Plant Energy Output)
**Dataset:** Combined Cycle Power Plant data (`powerplant_data.csv`)
The goal here is to predict the net hourly electrical energy output (PE) of a power plant based on four 
environmental features — temperature, ambient pressure, relative humidity, and exhaust vacuum.

**What I did:**
- Loaded and explored the dataset, checked for nulls
- Split into train/test sets (80/20) and applied StandardScaler
- Converted everything to PyTorch tensors and set up DataLoaders with batch size 32
- Built a feedforward ANN with 2 hidden layers (6 neurons each, ReLU activation)
- Trained for 100 epochs using MSE loss and the Adam optimizer
- Tracked training and validation loss every epoch and saved the best model (`best_model.pt`) based on lowest validation loss
- Evaluated with MSE and R² score, and compared predicted vs actual values in a DataFrame

**Architecture:**
```
Input (4) → Linear(6) → ReLU → Linear(6) → ReLU → Linear(1)
```

## Notebook 2 — Classification (Date Fruit Varieties)
**Dataset:** Date Fruit Dataset (`DateFruit_Dataset.csv`)
Here the goal is to classify date fruit samples into one of 7 varieties based on physical 
features extracted from images (like area, perimeter, color values, etc.).

**What I did:**
- Loaded the data and confirmed no missing values
- Used LabelEncoder to convert the string class labels into integers
- Split into train/test sets (80/20) and scaled features with StandardScaler
- Converted to PyTorch tensors (features as float32, labels as long for CrossEntropyLoss)
- Built a deeper ANN with 2 hidden layers (64 neurons each, ReLU activation) and a 7-neuron output layer
- Trained for 100 epochs using CrossEntropyLoss and Adam optimizer
- Evaluated by counting correct predictions and computing accuracy over the test set

**Architecture:**
```
Input (34) → Linear(64) → ReLU → Linear(64) → ReLU → Linear(7)
```

## How to Run
1. Clone the repo and make sure the datasets are in the same directory as the notebooks
2. Install the dependencies:
```bash
pip install torch pandas numpy scikit-learn matplotlib
```

3. Open either notebook in Jupyter and run all cells top to bottom

## Dependencies
- Python 3.x
- PyTorch
- pandas
- NumPy
- scikit-learn
- matplotlib

## Notes
Both notebooks are kept intentionally simple and readable — the focus is on understanding the full workflow of building 
and training a neural network with PyTorch rather than squeezing out the last bit of performance. Good starting point 
if you're learning how ANNs work end-to-end.
