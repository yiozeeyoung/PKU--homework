**Data Source:** Pre-loaded datasets on provided USB drives

**Available Datasets**

 **QM9star:** Molecular property dataset with 1.9 million ions and radicals

 **Reaction Dataset A:** Pd-catalyzed C–N coupling reaction ML prediction

 **Reaction Dataset B:** Bayesian optimization for reaction condition optimization

 **AHO Dataset:** Asymmetric hydrogenation with limited data (few-shot learning

scenario)

**Assignment Overview**

This homework consists of four interconnected practical problems designed to provide

hands-on experience with AI applications in organic chemistry. All required datasets

are pre-loaded on your USB drive - no internet connection required.

Each student must use 5 specific random seeds based on their student ID number. If

your student ID is XXX (e.g., 001, 002, 015), use the following seeds for reproducible

results:

• Seed 1: 1XXX (e.g., 1001 for student 001)

• Seed 2: 2XXX (e.g., 2001 for student 001)

• Seed 3: 3XXX (e.g., 3001 for student 001)

• Seed 4: 4XXX (e.g., 4001 for student 001)

• Seed 5: 5XXX (e.g., 5001 for student 001)

Use these seeds across different models and cross-validation splits to ensure

reproducibility while maintaining individual variation.

**Problem 1: Molecular Property Prediction**

**Objective**

Develop and compare multiple molecular representation methods for predictingHOMO-LUMO energy gaps using the QM9star dataset.

**Tasks**

**1. Data Loading and Exploration**

 • Load the QM9star dataset from datasets/qm9star_data.csv

 • Perform exploratory data analysis on molecular structures and energy gaps

 • Select a representative subset of 100,000 molecules for each molecular species

type

 • Identify and handle any missing or invalid SMILES strings

**2. Molecular Representation Implementation**

 • Implement Morgan fingerprints (radius=2, 2048 bits)

 • Calculate RDKit molecular descriptors

 • Generate atom-centered feature vectors

 • Create a comparison table of representation methods

**3. Model Development and Evaluation**

 • Train Random Forest models using each representation

 • Implement cross-validation (5-fold)

 • Compare model performance using MAE, RMSE, and R²

 • Analyze feature importance for the best-performing model

**Deliverables**

 • Python code with clear documentation

 • Performance comparison table

 • Analysis of which molecular representation works best and why

**Problem 2: Reaction Condition Prediction**

**Objective**

Build a multi-class classification model to predict optimal reaction conditions using the

**Reaction Dataset A** dataset.

**Tasks**

**1. Reaction Data Processing**

 Load **Reaction Dataset A** dataset from datasets/Reaction_Dataset_A.csv

 Analyze the distribution of catalysts, bases, and solvents

 Handle categorical variables and create appropriate encodings

**2. Reaction Fingerprint Development**

 Implement reaction fingerprints using reactant molecular fingerprints

 Create condition feature vectors (catalyst, base, solvent combinations)

 Combine molecular and condition features appropriately**3. Classification Model Implementation**

 Train classification models for predicting: Optimal catalyst selection, best base

choice, suitable solvent prediction

 Use appropriate evaluation metrics (accuracy, precision, recall, F1-score)

 Implement confusion matrix analysis

**Deliverables**

 Reaction fingerprint implementation

 Multi-class classification results

 Chemical interpretation of model predictions

**Problem 3: Bayesian Optimization for Reaction Yield**

**Objective**

Implement Bayesian optimization to predict and optimize reaction yields using the

**Reaction Dataset B** dataset.

**Tasks**

**1. Gaussian Process Model Setup**

 Load Doyle dataset from datasets/ Reaction_Dataset_B.csv

 Implement Gaussian Process Regressor with appropriate kernels

 Handle feature scaling and preprocessing

**2. Acquisition Function Implementation**

 Implement Expected Improvement (EI) acquisition function

 Code Upper Confidence Bound (UCB) as alternative

 Compare acquisition function performance

**3. Bayesian Optimization Loop**

 Implement active learning loop for yield optimization

 Start with 20 initial points, iteratively select 10 new experiments

 Track optimization progress and convergence

 Analyze optimal reaction conditions discovered

**Deliverables**

 Complete Bayesian optimization implementation

 Optimization trajectory plots

 Analysis of discovered optimal conditions

 Comparison of different acquisition functions**Problem 4: Transfer Learning and Few-Shot Learning**

**Objective**

Apply transfer learning techniques to handle the limited AHO dataset using knowledge

from larger datasets.

**Tasks**

**1. Baseline Model Training**

 Train a baseline model using only AHO dataset (datasets/aho.csv)

 Train a baseline model using only out-of-sample (OOS) data dataset

(datasets/aho_oos.csv)

 Document challenges with small dataset training with 5-fold cv by comparing

performances between AHO and OOS datasets

**2. Transfer Learning Implementation**

 Train more models on AHO dataset

 Tune the trained model on limited OOS data

 Implement domain adaptation techniques

 Compare transfer learning vs. baseline performance

**3. Uncertainty Quantification**

 Implement prediction uncertainty estimation

 Use ensemble methods or Bayesian approaches

 Analyze model confidence in predictions

 Identify regions where more data is needed

**Deliverables**

 Transfer learning implementation

 Performance comparison (baseline vs. transfer learning)

 Uncertainty analysis and visualization

**Submission Requirements**

**Code Requirements**

 All code must be in Python with clear comments

 Use modular programming with well-defined functions

 Include error handling for common issues

 Provide requirements.txt for package dependencies

**Documentation Requirements**

 Brief methodology explanation for each problem Results interpretation with chemical insights

 Discussion of limitations and potential improvements

 Code documentation following Python standards

**File Structure**

_submission/_

_├── problem1_molecular_properties.py_

_├── problem2_reaction_conditions.py_

_├── problem3_bayesian_optimization.py_

_├── problem4_transfer_learning.py_

_├── results/_

_│ ├── problem1_results.csv_

_│ ├── problem2_results.csv_

_│ ├── problem3_results.csv_

_│ └── problem4_results.csv_

_├── figures/_

_│ └── [relevant plots and visualizations]_

_└── README.md_