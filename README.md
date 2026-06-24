# Machine Learning - Applying ML Algorithms

##  Project Overview
This project applies end-to-end Machine Learning pipelines to analyze a marketing campaign dataset. The goal is to predict customer spending, classify campaign responses, and segment customers to discover hidden revenue opportunities and design targeted marketing strategies.

##  1. Data Preprocessing
Cleaned and transformed the raw data to ensure optimal model performance:

* **Missing Values:** Handled null values in the `Income` column using median imputation.
* **Feature Engineering:** Created informative features including `Age` (calculated from `Year_Birth`), `TotalSpending`, and `TotalChildren`.
* **Data Filtering:** Removed invalid records, keeping only realistic ages (18-100) and strictly positive incomes.
* **Categorical Encoding:** Applied Label Encoding to `Education` and grouped `Marital_Status` into the top 4 categories, applying One-Hot Encoding for the rest.

##  2. Supervised Learning - Regression
**Target Variable:** `TotalSpending` (Continuous)

* **Data Prep:** 80/20 train-test split, scaled numerical features using `StandardScaler`.
* **Models Evaluated:** Linear Regression, Ridge Regression (alpha=1.0), and Decision Tree Regressor (max_depth=5).

**Key Insights:**
* **Linear & Ridge Regression:** Both explained 78.16% of the variance. The identical performance indicates that the default L2 regularization penalty (alpha=1.0) did not significantly alter the baseline linear relationships.
* **Decision Tree Regressor:** Achieved a higher variance capture of 84.81%. However, the model exhibited overfitting, capturing noise specific to the training set rather than generalizing smoothly.

##  3. Supervised Learning - Classification
**Target Variable:** `Response` (Binary: 1 if accepted the campaign, 0 otherwise)

* **Challenge:** The dataset is highly imbalanced, heavily skewing towards "No" (0).
* **Models Evaluated:** Logistic Regression, K-Nearest Neighbors (KNN), and Random Forest Classifier.

**Key Insights & The "Accuracy Illusion":**
* **KNN & Random Forest:** Achieved the highest overall accuracies (87.50% and 86.83% respectively) but completely failed the business objective by mostly predicting the majority class ("No").
* **Winning Model:** **Logistic Regression** (Accuracy: 81.03%). By utilizing `class_weight='balanced'`, it successfully captured the minority class, identifying 51 out of 72 converting customers. In imbalanced datasets, capturing the true positive conversions is vastly more valuable than overall accuracy.

##  4. Unsupervised Learning - Clustering
**Goal:** Segment customers using `TotalSpending`, `Income`, `Age`, and `TotalChildren`.

* **Methodology:** Used the Elbow Method to determine the optimal number of clusters (k=3), then fitted a K-Means model on the scaled features. Visualized the clusters using PCA (2 components).

**Results (Customer Segments Discovered):**
* **Cluster 0:** Average age: 64 years old, moderate income ($47K), low spending ($298) with the highest number of children (average ~2).
* **Cluster 1 (High-Value):** Average age: 58 years old, high income ($74K), high spending ($1289) with the lowest number of children (average ~0).
* **Cluster 2 (The Youngest):** Average age: 48 years old, lowest income ($33K), lowest spending ($160) with a moderate number of children (average ~1).

##  Conclusion
This end-to-end analysis truly showcases the undeniable importance of Machine Learning. It shifts a business from guessing to predicting, allowing us to capture hidden revenue opportunities and design hyper-targeted marketing strategies with absolute confidence based on the extracted customer segments.
