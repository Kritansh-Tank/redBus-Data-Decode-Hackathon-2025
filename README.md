# 🚌 Bus Demand Forecasting

## 🎯 Objective

This project predicts the final number of bus seats booked (`final_seatcount`) for different bus routes on specific journey dates. The forecasts help transport companies optimize:

- Fleet management  
- Dynamic pricing  
- Resource planning  

---

## 🔍 1. Data Loading and Cleaning

### 📂 Datasets Used:
- `train.csv`: Historical route and booking data.
- `test.csv`: Future journeys to predict demand for.
- `transactions.csv`: Search and booking activity logs.

### 🧹 Preprocessing:
- **Date Parsing**: Converts `doj` and `doi` into datetime format.
- **Data Cleaning**: Removes rows with invalid/missing dates.

---

## 🧠 2. Feature Engineering

### 📅 Date-Based Features (`create_holiday_features`)
- Extracts: year, month, day, weekday, quarter, weekend flag
- Flags for Indian holidays: Diwali, Holi, Dussehra, Wedding Season, Summer/Winter Vacations
- Identifies potential long weekends

### 🧮 Lag Features (`create_lag_features`)
- Past seat counts from 7, 14, 21, and 30 days ago per route

### 📊 Aggregated Transaction Features (`create_aggregated_features`)
- Aggregates search and booking counts from `transactions.csv` at the `(srcid, destid, doj)` level
- Statistics: `mean`, `max`, `std`

### 🗺️ Route Features (`create_route_features`)
- Adds route-level average demand, standard deviation, and frequency
- Encodes categorical features like `srcid`, `destid` using `LabelEncoder`

### 🔗 Interaction Features
- Combines features like `month_dayofweek` and `is_peak_season`

---

## 🧪 3. Data Preparation

- Final feature matrix is cleaned (removes NaNs, infinities)
- Splits features and target (`final_seatcount`)
- **Time-based validation split**: 
  - **80%** earliest dates for training
  - **20%** latest dates for validation

---

## 🤖 4. Model Training and Selection (`train_models`)

### Models Used:
- `LightGBM`
- `XGBoost`
- `Random Forest`
- `Gradient Boosting`
- (Fallback: `Linear Regression` if all else fails)

### Model Selection:
- Evaluation metric: **RMSE**
- Best model chosen based on lowest validation RMSE
- Feature importance recorded (if supported)

---

## 🔄 5. Ensemble Prediction (`create_ensemble_prediction`)

Blends predictions from all trained models using weighted averaging:

| Model              | Weight |
|-------------------|--------|
| LightGBM          | 0.40   |
| XGBoost           | 0.30   |
| Random Forest     | 0.20   |
| Gradient Boosting | 0.10   |

---

## 🧾 6. Submission and Output (`predict`)

- Generates predictions on test data using the best model
- Outputs CSV file: `submission.csv`
- Columns: `route_key`, `final_seatcount`

---

## 📈 7. Visualization

- **Feature Importance Plot**: Top N features by importance score
- **EDA Function**: `exploratory_data_analysis()` shows:
  - Demand distribution
  - Monthly trends
  - Weekday booking patterns

---

## ✅ 8. Final Pipeline Execution (`run_full_pipeline`)

Full pipeline includes:

Load → Feature Engineering → Train → Predict → Visualize → Submit

---

## 📊 Summary Output

When run, the pipeline prints:

- A preview of the submission
- Total number of predictions
- Average, min, and max predicted demand

---

## 🧠 Highlights

- Modular and robust pipeline with fallback models  
- Handles structured data and time series effectively  
- Feature-rich with domain-specific + statistical features  
- Scalable for different data sizes and future enhancements  

---

## 🏁 How to Run

```bash
python eda_model_pipeline.py
