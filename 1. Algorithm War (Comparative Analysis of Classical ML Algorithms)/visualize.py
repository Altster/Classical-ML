import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay

from harness import get_preprocessor
from models import get_models

MODEL_COLORS = {
    "Logistic Regression": "#1f77b4",
    "SVM (Linear)": "#4c9aff",
    "Decision Tree": "#2ca02c",
    "Random Forest": "#228B22",
    "Gradient Boosting": "#7cb342",
    "XGBoost": "#66bb6a",
    "Gaussian Naive Bayes": "#d62728",
    "Multinomial Naive Bayes": "#d62728",
    "K-Nearest Neighbors": "#ff9800",
    "SVM (RBF)": "#9467bd",
}


def plot_test_accuracy(results, dataset_name):
    colors = [MODEL_COLORS.get(m, "gray") for m in results["model"]]

    plt.style.use('Solarize_Light2')

    plt.figure(figsize=(12,6))

    bars = plt.bar(
        results["model"],
        results["mean_test_score"],
        color=colors,
        linewidth=2,
        edgecolor="black",
        yerr=results["std_test_score"],
        capsize=6
    )

    plt.bar_label(bars, fmt="%.2f", padding=3)

    plt.xticks(rotation=35)
    plt.ylabel("Mean Cross-Validation Accuracy")
    plt.title(f"{dataset_name} - Accuracy (Mean ± Std)")
    plt.show()


def plot_accuracy_vs_fit_time(results, dataset_name):

    plt.style.use("Solarize_Light2")
    plt.figure(figsize=(10,6))

    for _, row in results.iterrows():

        plt.scatter(
            row["mean_fit_time"],
            row["mean_test_score"],
            color=MODEL_COLORS.get(row["model"], "gray"),
            s=150,
            edgecolors="black",
            linewidths=1.8
        )

        plt.annotate(
            row["model"],
            (row["mean_fit_time"], row["mean_test_score"]),
            xytext=(0, 8),             
            textcoords="offset points", # Interpret xytext as point offsets
            ha="center",                # Center horizontally
            fontsize=9
        )

    plt.xscale("log")
    plt.xlabel("Mean Fit Time (seconds)")
    plt.ylabel("Mean Cross-Validation Accuracy")
    plt.title(f"{dataset_name} - Accuracy vs Training Time")

    plt.grid(True, which="both", linestyle="--", linewidth=0.8, alpha=0.8)

    plt.show()

#AI generated code
def plot_decision_boundaries(X, y, numerical_columns, categorical_columns, preprocessor_type="tabular"):
    # Preprocess the data
    preprocessor = get_preprocessor(
        preprocessor_type,
        numerical_columns,
        categorical_columns
    )

    X_processed = preprocessor.fit_transform(X)

    # Reduce to 2 dimensions
    pca = PCA(n_components=2, random_state=37)
    X_2d = pca.fit_transform(X_processed)

    # Get all models
    models = get_models(preprocessor_type)

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.ravel()

    for ax, (name, model) in zip(axes, models.items()):

        # Train on PCA representation
        model.fit(X_2d, y)

        # Plot decision boundary
        DecisionBoundaryDisplay.from_estimator(
            model,
            X_2d,
            response_method="predict",
            grid_resolution=300,
            alpha=0.35,
            ax=ax,
        )

        # Plot data points
        ax.scatter(
            X_2d[:, 0],
            X_2d[:, 1],
            c=y,
            cmap="coolwarm",
            edgecolors="black",
            linewidths=0.5,
            s=25
        )

        ax.set_title(name, fontsize=12)
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")

    # Remove unused subplot (8 models -> 9 axes)
    for i in range(len(models), len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle("Decision Boundaries of Machine Learning Models", fontsize=18)
    plt.tight_layout()
    plt.show()