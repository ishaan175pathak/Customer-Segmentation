import os
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
from sklearn.decomposition import PCA


class DataVisualizer:
    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str = "subscription_status",
        save_dir: Optional[str] = None
    ):
        self.df = df.copy()
        self.target_col = target_col
        self.save_dir = save_dir

        sns.set_theme(style="whitegrid")

        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

    def _save_or_show(self, filename: Optional[str] = None):
        plt.tight_layout()
        if self.save_dir and filename:
            path = os.path.join(self.save_dir, filename)
            plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_class_distribution(self):
        plt.figure(figsize=(10, 8))
        
        def __normalize_target_val(value):
            if pd.isna(value):
                return np.nan

            value_str = str(value).strip().lower()

            if value_str in {"true", "1", "yes", "y", "1.0"}:
                return 1
            if value_str in {"false", "0", "no", "n", "0.0"}:
                return 0

            return np.nan

        self.df[self.target_col] = self.df[self.target_col].apply(__normalize_target_val)

        ax = sns.countplot(y=self.target_col, data=self.df)
        ax.set_title("Class Distribution")
        ax.set_xlabel("Subscription Status")
        ax.set_ylabel("Count")
        self._save_or_show("class_distribution.png")

    def plot_missing_values(self):
        missing = self.df.isnull().sum().sort_values(ascending=False)
        missing = missing[missing > 0]

        plt.figure(figsize=(10, 5))
        if missing.empty:
            plt.text(0.5, 0.5, "No missing values found", ha="center", va="center", fontsize=14)
            plt.axis("off")
        else:
            ax = missing.plot(kind="bar")
            ax.set_title("Missing Values per Column")
            ax.set_xlabel("Columns")
            ax.set_ylabel("Missing Count")
        self._save_or_show("missing_values.png")

    def plot_numeric_histograms(self, columns: Optional[Sequence[str]] = None, bins: int = 20):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.int64, np.float64]).columns.tolist()

        columns = [col for col in columns if col in self.df.columns and col != self.target_col]

        if not columns:
            print("No numeric columns available for histogram plotting.")
            return

        n_cols = 3
        n_rows = int(np.ceil(len(columns) / n_cols))

        plt.figure(figsize=(5 * n_cols, 4 * n_rows))
        for i, col in enumerate(columns, 1):
            plt.subplot(n_rows, n_cols, i)
            sns.histplot(data=self.df, x=col, bins=bins, kde=True)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Count")

        self._save_or_show("numeric_histograms.png")

    def plot_correlation_heatmap(self):
        numeric_df = self.df.select_dtypes(include=[np.int64, np.float64])
        if numeric_df.shape[1] < 2:
            print("Not enough numeric columns for correlation heatmap.")
            return

        plt.figure(figsize=(10, 8))
        corr = numeric_df.corr()
        ax = sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
        ax.set_title("Correlation Heatmap")
        self._save_or_show("correlation_heatmap.png")

    def plot_boxplots_by_class(self, features: Sequence[str]):
        features = [col for col in features if col in self.df.columns and col != self.target_col]

        if not features:
            print("No valid features provided for boxplot.")
            return

        n_cols = 2
        n_rows = int(np.ceil(len(features) / n_cols))

        plt.figure(figsize=(6 * n_cols, 4 * n_rows))
        for i, col in enumerate(features, 1):
            plt.subplot(n_rows, n_cols, i)
            sns.boxplot(x=self.target_col, y=col, data=self.df)
            plt.title(f"{col} by {self.target_col}")
            plt.xlabel(self.target_col)
            plt.ylabel(col)

        self._save_or_show("boxplots_by_class.png")

    def plot_scatter(
        self,
        x_col: str,
        y_col: str,
        hue_col: Optional[str] = None
    ):
        if x_col not in self.df.columns or y_col not in self.df.columns:
            print("One or both scatter plot columns are invalid.")
            return

        if hue_col is None:
            hue_col = self.target_col if self.target_col in self.df.columns else None

        plt.figure(figsize=(8, 6))
        ax = sns.scatterplot(
            data=self.df,
            x=x_col,
            y=y_col,
            hue=hue_col
        )
        ax.set_title(f"{y_col} vs {x_col}")
        self._save_or_show(f"scatter_{x_col}_{y_col}.png")

    def plot_pairplot(self, features: Optional[Sequence[str]] = None, sample_size: int = 500):
        if features is None:
            features = self.df.select_dtypes(include=[np.int64, np.float64]).columns.tolist()

        features = [col for col in features if col in self.df.columns and col != self.target_col]

        if len(features) < 2:
            print("Need at least two numeric features for pairplot.")
            return

        sample_df = self.df[features + [self.target_col]].dropna()
        if len(sample_df) > sample_size:
            sample_df = sample_df.sample(sample_size, random_state=42)

        sns.pairplot(sample_df, hue=self.target_col, diag_kind="hist")
        plt.suptitle("Pairplot of Selected Features", y=1.02)
        self._save_or_show("pairplot.png")

    def plot_pca_2d(self, X: pd.DataFrame, y: pd.Series):
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_values)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=y,
            cmap="viridis",
            alpha=0.8
        )
        plt.title("2D PCA Projection")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar(scatter, label=self.target_col)
        self._save_or_show("pca_2d.png")

    def plot_accuracy_vs_k(self, k_values: Sequence[int], train_scores: Sequence[float], test_scores: Sequence[float]):
        plt.figure(figsize=(8, 5))
        plt.plot(k_values, train_scores, marker="o", label="Train Accuracy")
        plt.plot(k_values, test_scores, marker="o", label="Test Accuracy")
        plt.title("Accuracy vs K")
        plt.xlabel("K Value")
        plt.ylabel("Accuracy")
        plt.legend()
        self._save_or_show("accuracy_vs_k.png")

    def plot_confusion_matrix(self, cm, class_labels: Optional[List[str]] = None, title: str = "Confusion Matrix"):
        plt.figure(figsize=(6, 5))
        ax = sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_labels if class_labels else True,
            yticklabels=class_labels if class_labels else True
        )
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        self._save_or_show(f"{title.lower().replace(' ', '_')}.png")

    def plot_feature_means_by_class(self, features: Sequence[str]):
        features = [col for col in features if col in self.df.columns and col != self.target_col]

        if not features:
            print("No valid features provided for class mean plot.")
            return

        class_means = self.df.groupby(self.target_col)[features].mean().T

        plt.figure(figsize=(12, 6))
        ax = class_means.plot(kind="bar", figsize=(12, 6))
        ax.set_title("Feature Means by Class")
        ax.set_xlabel("Features")
        ax.set_ylabel("Mean Value")
        plt.xticks(rotation=45, ha="right")
        self._save_or_show("feature_means_by_class.png")

    def plot_model_evaluation(self, results, y_test):

        k_vals = []
        train_accs = []
        test_accs = []
        precisions = []
        recalls = []

        for result in results:
            k = result["k"]
            y_pred = result["y_test_pred"]

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)

            k_vals.append(k)
            train_accs.append(result["train_accuracy"])
            test_accs.append(result["test_accuracy"])
            precisions.append(prec)
            recalls.append(rec)

            # Confusion Matrix per K
            self.plot_confusion_matrix(
                cm,
                class_labels=["Not Subscribed", "Subscribed"],
                title=f"Confusion Matrix (K={k})"
            )

        # Accuracy vs K
        self.plot_accuracy_vs_k(k_vals, train_accs, test_accs)

        # Precision & Recall bar chart
        import numpy as np
        import matplotlib.pyplot as plt

        x = np.arange(len(k_vals))
        width = 0.35

        plt.figure(figsize=(8, 5))
        plt.bar(x - width/2, precisions, width, label='Precision')
        plt.bar(x + width/2, recalls, width, label='Recall')

        plt.title("Precision and Recall vs K")
        plt.xlabel("K Value")
        plt.ylabel("Score")
        plt.xticks(x, k_vals)
        plt.ylim(0, 1)
        plt.legend()

        self._save_or_show("precision_recall_vs_k.png")

        # Misclassification Error plot
        misclassification_errors = [1 - acc for acc in test_accs]

        plt.figure(figsize=(8, 5))
        plt.plot(k_vals, misclassification_errors, marker='o')
        plt.title("Misclassification Error vs K")
        plt.xlabel("K Value")
        plt.ylabel("Error")

        self._save_or_show("misclassification_vs_k.png")