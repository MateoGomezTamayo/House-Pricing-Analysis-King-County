# House Pricing Analysis — King County, WA

Identifies the key drivers of house sale prices in King County (Seattle area) and compares six regression models to predict them.

**Best model:** Gradient Boosting, **R² = 0.739** on the test set, RMSE ≈ $195,688.

## What's inside

| File | Description |
|---|---|
| `house_pricing_analysis.ipynb` | Full analysis: cleaning, EDA, correlation analysis, modeling and evaluation |
| `gradient_boosting_house_price_model.pkl` | Trained Gradient Boosting model (loadable with `joblib`) |

## Process

1. **Data wrangling:** handle missing values and fix data types in the King County house sales dataset.
2. **Exploratory analysis:** price distribution (strong right skew), price by bedrooms, waterfront premium, living area vs. price.
3. **Correlation analysis:** `sqft_living` (r = 0.70) and `grade` (r = 0.67) are the strongest linear predictors.
4. **Modeling:** from simple linear regression up to tree ensembles, with polynomial features, Ridge regularization and a log-target variant.
5. **Evaluation:** residual diagnostics (heteroscedasticity), actual vs. predicted, feature importance and SHAP.

## Results (test set)

| Model | R² | RMSE |
|---|---|---|
| **Gradient Boosting** | **0.739** | **$195,688** |
| Random Forest | 0.734 | $197,817 |
| Polynomial Ridge (degree 2) | 0.705 | $208,109 |
| Log-Ridge (degree 2) | 0.676 | $218,271 |
| Multiple Linear Regression (11 features) | 0.658 | — |
| Simple Linear Regression (`sqft_living`) | 0.493 | — |

## Key findings

- **Construction grade** is the top driver in the Gradient Boosting model (≈ 37% of feature importance), followed by living area and location (latitude).
- **Waterfront** homes sell at a large premium: median ≈ $1.4M.
- Linear models systematically **under-predict ultra-luxury homes** (> $2M). That segment likely needs its own model.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook house_pricing_analysis.ipynb
```

The dataset loads directly from the IBM Skills Network course URL inside the notebook.

## Tools

Python · pandas · NumPy · scikit-learn · SHAP · Matplotlib · seaborn · SciPy

---
Mateo Gómez Tamayo · [LinkedIn](https://www.linkedin.com/in/mateo-g%C3%B3mez-tamayo) · [GitHub](https://github.com/MateoGomezTamayo)
