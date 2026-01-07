
# Real Estate Price Prediction (Tabular + Satellite Imagery)

This repository contains an end-to-end workflow to predict **house prices** using:
1) **Tabular features** (sqft, bedrooms, bathrooms, grade, condition, zipcode, lat/long, etc.)  
2) **Satellite imagery** downloaded using coordinates (multimodal attempt)

**Final submission uses the Tabular model** because the **multimodal CNN + tabular fusion** performed worse on validation (higher RMSE, negative R² in our run).

---

## Repository Contents (as uploaded)


|------|------------------|
| `real_state_price_prediction.ipynb` | Complete notebook: EDA → preprocessing → feature engineering → winsorisation/outliers → zipcode OHE → training → evaluation → submission |
| `submission_tabular.csv` | Final predictions in required format: `id, predicted_price` |
| `Real_Estate_Price_Prediction_Report.pdf` | Detailed project report (EDA + workflow + model results) |


---

## Problem Statement (Short)

Predict the **Price** of a property using historical housing data.  
The project also includes fetching **satellite images** using `lat` and `long` to capture neighborhood context (green cover, roads, etc.), then combining image features with tabular features.

---

## Workflow Summary (Notebook)

### 1) Data Loading
- Read train and test CSVs
- Separate:
  - **Target**: `price` (train only)
  - **Features**: all remaining columns

### 2) EDA (Exploratory Data Analysis)
Covered in the notebook:
- Price distribution (raw and log-transformed)
- Correlation of important features with price  
  (e.g., `sqft_living`, `grade`, `sqft_above`, `bathrooms`, location)
- Feature-feature relationships and multicollinearity
- Location patterns using `lat` / `long`
- Outlier detection for strong numeric drivers

### 3) Preprocessing + Feature Engineering
Key steps implemented:
- `date` parsing and deriving time-based features (if applicable)
- Creating interaction/ratio features (example):
  - `above_to_living`, `basement_to_living`
  - `house_age`, `is_renovated`
  - `bed_bath_interaction`
- Handling missing values
- Winsorisation / clipping using quantile caps (to reduce extreme outlier influence)

### 4) Encoding
- `zipcode` converted to string and One-Hot Encoded (OHE)
- Train and test aligned so both have identical final feature columns

### 5) Model Training
**Tabular baseline (Final model used):**
- Trained using `log1p(price)` target
- Evaluated on a validation split (RMSE and R²)

**Multimodal model (Built but not used):**
- Satellite images downloaded using coordinates
- CNN extracts embeddings + fused with tabular features
- Performance was worse than tabular-only in this run

### 6) Submission Generation
- Predict on test set
- Convert predictions back to original scale using `expm1`
- Export `submission_tabular.csv` in format:

```csv
id,predicted_price
