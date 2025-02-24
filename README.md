# Loan Approval Prediction 

## Project Overview
This project aims to predict loan approval status based on various features. The goal is to develop machine learning models from scratch without relying on libraries such as `sklearn`. Several approaches are explored, including traditional machine learning, ensemble methods, and deep learning.

## Approaches and Models
The following methods are implemented and compared:

1. **Logistic Regression & KNN**  
   - Basic classification models for loan approval prediction.
   - Implemented from scratch without using `sklearn`.

2. **Model Ensemble (Random Forest, XGBoost)**  
   - Combining multiple decision trees for better performance.
   - XGBoost and Random Forest are implemented from scratch.

3. **Neural Network**  
   - A multi-layer perceptron (MLP) model trained for classification.
   - Uses custom backpropagation and optimization techniques.

4. **Stacking Model**  
   - Combines multiple models by training a meta-model on their predictions.
   - Helps improve overall performance by leveraging strengths of individual models.

5. **Voting Model**  
   - Uses majority voting from multiple classifiers to make the final decision.
   - Helps to stabilize predictions and reduce variance.

## Dataset and Preprocessing
The dataset consists of loan applicants' features such as income, credit score, loan amount, and employment status. The dataset is preprocessed to handle missing values, outliers, categorical encoding, feature engineering, feature scaling and oversampling for unbalance reduction before feeding into models.

## Implementation Details
- No use of `sklearn` or other high-level ML libraries.
- Manual implementation of logistic regression, KNN, decision trees, ensemble methods, and neural networks.
- Custom training loops and hyperparameter tuning.

## Requirements

- Python 3.11.8

To install the required packages, run:

```sh
pip install -r requirements.txt
```

## Evaluation Metrics
Models are evaluated using:
- **Precision**
- **Recall**
- **F1-score**

## Final Submission Results

| Model                                      | Precision | Recall  | F1-Score |
|--------------------------------------------|-----------|---------|----------|
| Logistic Regression                        | 0.7436    | 0.8534  | 0.7947   |
| KNN                                        | 0.7566    | 0.8171  | 0.7857   |
| Random Forest                              | 0.8498    | 0.8195  | **0.8344** |
| XG Boost                                   | 0.8005    | 0.8033  | 0.8019   |
| Neural Network                             | 0.7954    | 0.8411  | 0.8190   |
| Bagging Neural Network                     | 0.8112    | 0.8468  | 0.8286   |
| Stacking (LR - meta learner, RF, XGBoost, Bagging NN) | 0.8685    | 0.8019  | **0.8339** |
| Hard Voting (RF, XGBoost, Bagging NN)      | 0.8274    | 0.8302  | 0.8288   |
| Soft Voting (RF, XGBoost, Bagging NN)      | 0.8340    | 0.8285  | **0.8312** |




