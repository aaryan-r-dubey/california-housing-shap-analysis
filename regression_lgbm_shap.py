import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor
import shap


def load_data():
    data = fetch_california_housing(as_frame=True)
    df = pd.concat([data.data, data.target.rename("MedHouseVal")], axis=1)
    return df


def print_data_info(df):
    print("Dataset head:")
    print(df.head(), "\n")

    print("Dataset shape:")
    print(df.shape, "\n")

    print("Dataset description:")
    print(df.describe(), "\n")


def plot_correlation_heatmap(df):
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()
    return corr


def summarize_target_correlations(corr, target="MedHouseVal", threshold=0.4):
    target_corr = corr[target].drop(target)
    strong_corr = target_corr[abs(target_corr) >= threshold].sort_values(ascending=False)

    print(f"Features strongly correlated with {target} (|corr| >= {threshold}):")
    print(strong_corr, "\n")

    print("Important features summary:")
    if "MedInc" in corr.columns:
        print("- MedInc: usually the strongest positive predictor of median house value.")
    if "AveRooms" in corr.columns:
        print("- AveRooms: more rooms per household often align with higher values, but may indicate family size.")
    if "HouseAge" in corr.columns:
        print("- HouseAge: older homes can be less expensive, although this depends on location and renovation.")
    print("\n")
    return strong_corr


def prepare_data(df, test_size=0.2, random_state=42):
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]
    # Split into training and test sets with an 80-20 split and fixed random state
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, random_state=42):
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        random_state=random_state,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2 = r2_score(y_test, predictions)
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R2 score: {r2:.4f}\n")
    return rmse, r2


def explain_model(model, X_data):
    explainer = shap.Explainer(model, X_data)
    shap_values = explainer(X_data)
    return explainer, shap_values


def plot_shap_summary(shap_values, X_data):
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_data, plot_type="bar", show=False)
    plt.title("SHAP Summary - Feature Importance")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_data, show=False)
    plt.title("SHAP Summary - Beeswarm Plot")
    plt.tight_layout()
    plt.show()


def compute_shap_interaction_values(model, X_data):
    tree_explainer = shap.TreeExplainer(model)
    interaction_values = tree_explainer.shap_interaction_values(X_data)
    return interaction_values


def plot_shap_interaction(interaction_values, X_data, feature, interaction_feature):
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        feature,
        interaction_values,
        X_data,
        interaction_index=interaction_feature,
        show=False,
        alpha=0.7,
        dot_size=50,
    )
    plt.title(f"SHAP Interaction: {feature} vs {interaction_feature}")
    plt.xlabel(feature)
    plt.ylabel("SHAP value")
    plt.tight_layout()
    plt.show()


def plot_partial_dependence(model, X_data, features=None):
    features = features or ["MedInc", "AveRooms", "HouseAge", ("MedInc", "AveRooms")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    display = PartialDependenceDisplay.from_estimator(
        model,
        X_data,
        features,
        kind="average",
        ax=axes,
        grid_resolution=50,
    )
    fig.suptitle("Partial Dependence Plots")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def print_shap_dependence_insights():
    shap_mean_abs = np.abs(shap_values.values).mean(axis=0)
    feature_importance = pd.Series(shap_mean_abs, index=X_data.columns).sort_values(ascending=False)
    print(f"Top {top_n} SHAP-important features:")
    print(feature_importance.head(top_n), "\n")
    print("SHAP interpretation summary:")
    print("- MedInc is typically the strongest driver of price predictions, reflecting household income.")
    print("- AveRooms often contributes positively, suggesting larger homes are valued higher.")
    print("- HouseAge can be important too; older homes may lower predicted value compared to newer ones.")
    print("- Longitude and latitude capture geographical influence on value.")
    print("- Population and AveOccup may also impact the final prediction through neighborhood density.")
    print("\n")


def plot_feature_importance(model, X_train):
    importance = pd.Series(model.feature_importances_, index=X_train.columns)
    importance = importance.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance.values, y=importance.index, palette="viridis")
    plt.title("LightGBM Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


def plot_shap_dependence(shap_values, X_data, feature, interaction_index=None):
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        feature,
        shap_values.values,
        X_data,
        interaction_index=interaction_index,
        show=False,
        alpha=0.7,
        dot_size=50,
    )
    plt.title(f"SHAP Dependence Plot: {feature}")
    plt.xlabel(feature)
    plt.ylabel("SHAP value")
    plt.tight_layout()
    plt.show()


def print_shap_dependence_insights():
    print("SHAP dependence insights:")
    print("- MedInc has a strong positive non-linear impact on predicted house prices.")
    print("  As median income rises, predicted value increases rapidly at lower income levels,")
    print("  then shows a saturation effect where additional income has smaller marginal gains.")
    print("- AveRooms typically contributes positively, indicating larger homes are worth more,")
    print("  but the effect flattens as room count becomes very high, suggesting diminishing returns.")
    print("- HouseAge often shows a threshold effect: newer homes add value, while older homes")
    print("  may reduce predicted values after a certain age, reflecting depreciation.")
    print("- Interaction-aware plots can highlight how these features behave differently with")
    print("  other attributes such as location, income, or population density.\n")


def print_final_summary():
    print("Final summary:")
    print("- The LightGBM model successfully captures non-linear relationships in the California Housing data.")
    print("- SHAP helped interpret the model by showing how individual features contributed to each prediction.")
    print("- MedInc is the most important driver, with strong early gains and diminishing returns at higher values.")
    print("- AveRooms and HouseAge also matter, showing that room count and home age influence value in non-linear ways.")
    print("- Understanding non-linear effects is important for real-world housing prediction because prices")
    print("  do not move uniformly with a single feature; they depend on thresholds, saturation, and feature interactions.")
    print("- Using SHAP and partial dependence together makes these complex relationships easier to explain and trust.\n")


def main():
    df = load_data()
    print_data_info(df)

    corr = plot_correlation_heatmap(df)
    summarize_target_correlations(corr)

    X_train, X_test, y_train, y_test = prepare_data(df)

    model = train_model(X_train, y_train)
    print("Model training complete.")

    evaluate_model(model, X_test, y_test)

    explainer, shap_values = explain_model(model, X_test)
    print("SHAP explanation generated for the test set.")

    plot_feature_importance(model, X_train)
    plot_shap_summary(shap_values, X_test)
    print_shap_importance_summary(shap_values, X_test)

    plot_shap_dependence(shap_values, X_test, "MedInc")
    plot_shap_dependence(shap_values, X_test, "AveRooms")
    plot_shap_dependence(shap_values, X_test, "HouseAge")

    # SHAP interaction values highlight how MedInc and AveRooms jointly affect the model.
    interaction_values = compute_shap_interaction_values(model, X_test)
    plot_shap_interaction(interaction_values, X_test, "MedInc", "AveRooms")

    plot_partial_dependence(model, X_test)
    print_shap_dependence_insights()
    print_final_summary()


if __name__ == "__main__":
    main()
