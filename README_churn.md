# Customer Churn Prediction

Predicting which telecom customers are likely to leave, using the Telco Customer Churn dataset (~7,000 customers). Focus: not just predicting churn, but understanding what drives it.

## Approach

- Cleaned the data, including a hidden issue where `TotalCharges` was stored as text with blank values for new customers.
- Explored churn by contract type and tenure.
- One-hot encoded categorical features.
- Compared two models on a class-imbalanced target (~27% churn), where accuracy alone is misleading.

## Results

Since the data is imbalanced, the key metric is recall on the churn class (how many leaving customers the model catches).

| Model | Accuracy | Churn recall |
|---|---|---|
| Random Forest (balanced) | 79% | 0.45 |
| Logistic Regression (balanced) | 75% | 0.82 |

Logistic Regression traded a little accuracy for much higher recall. Since missing a leaving customer costs more than a false alarm, it is the better choice here.

## Key findings

- Churn is highest among new customers (low tenure) and month-to-month contracts.
- Long contracts and higher tenure strongly reduce churn.
- Fiber optic customers churn more, worth investigating further.

## Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook churn_prediction.ipynb
```

Dataset: Telco Customer Churn (Kaggle).
