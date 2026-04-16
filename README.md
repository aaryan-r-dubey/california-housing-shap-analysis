# California Housing Price Analysis using SHAP and LightGBM

## Project Overview

This project explores the California Housing dataset from `sklearn` using a LightGBM regression model. The goal is to analyze how housing features influence predictions and to interpret the model using SHAP explainability tools.

## Objectives

- Train a LightGBM regressor on the California Housing dataset
- Analyze feature importance and model behavior with SHAP
- Visualize non-linear relationships using SHAP dependence plots
- Compare SHAP explanations with partial dependence plots
- Highlight the most influential features for predicted house prices

## Technologies Used

- Python
- scikit-learn
- LightGBM
- SHAP
- pandas
- NumPy
- matplotlib
- seaborn

## Dataset Description

The dataset is the `California Housing` dataset from the `sklearn.datasets` module. It includes features such as:

- `MedInc` — median income in the block group
- `HouseAge` — median age of the houses
- `AveRooms` — average number of rooms
- `AveOccup` — average occupancy
- `Longitude`, `Latitude` — location coordinates
- `MedHouseVal` — target median house value

## Installation

1. Clone the repository or download the project files.
2. Create a Python environment (recommended):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. Install required packages:
   ```bash
   pip install numpy pandas scikit-learn lightgbm shap matplotlib seaborn
   ```

## How to Run the Notebook

1. Open the project folder in Jupyter Notebook or JupyterLab.
2. Launch `regression_lgbm_shap.ipynb`.
3. Run the cells in order from top to bottom.

Alternatively, you can run the Python script directly:

```bash
python regression_lgbm_shap.py
```

## Key Insights

- `MedInc` is typically the strongest predictor of median house value.
- SHAP dependence plots reveal that the effect of `MedInc` on predicted price is non-linear, with early gains at lower income levels and diminishing returns at higher levels.
- `AveRooms` also has a positive impact, but very high room counts may show diminishing returns.
- `HouseAge` can behave non-linearly, with newer homes contributing positively while older homes may reduce the predicted value.
- Interaction analysis shows that `MedInc` and `AveRooms` jointly influence predictions, reflecting complex relationships in housing data.

## Screenshots / Plots

- `![SHAP Summary Plot](path/to/shap_summary_placeholder.png)`
- `![SHAP Dependence Plot](path/to/shap_dependence_placeholder.png)`
- `![Partial Dependence Plot](path/to/partial_dependence_placeholder.png)`

> Replace the placeholder paths with actual exported images if needed.

## Conclusion

This project demonstrates how SHAP can be used to interpret a LightGBM regression model for housing price prediction. By combining model performance metrics, feature importance, and dependence plots, the analysis provides a clear and accessible view of non-linear relationships in real-world housing data.
