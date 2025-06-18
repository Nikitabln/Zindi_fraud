## Notebook: Fraud Detection in Electricity and Gas Consumption

This notebook details the process of building a machine learning model to detect fraud in electricity and gas consumption data. The primary steps include data loading, exploratory data analysis, feature engineering, and model training.

The Zindi challenge can be found here:

https://zindi.africa/competitions/fraud-detection-in-electricity-and-gas-consumption-challenge

### 1. Data Loading and Initial Processing
- **Import Libraries**: All necessary libraries such as `pandas`, `numpy`, `matplotlib`, `seaborn`, and `sklearn` are imported.
- **Load Data**: The `client_train`, `invoice_train`, `client_test`, and `invoice_test` datasets are loaded from CSV files.
- **Data Merging**: The client and invoice datasets are merged on `client_id` to create a comprehensive training (`df`) and testing (`df_test`) DataFrame.
- **Type Conversion**: Date columns (`invoice_date`, `creation_date`) are converted to datetime objects for time-based analysis.

### 2. Exploratory Data Analysis (EDA)
- Visualizations are created to understand the distribution of the `target` variable across different categorical features like `district`, `client_catg`, `region`, and `tarif_type`. This helps in identifying initial patterns and potential predictive features.

### 3. Data Cleaning & Feature Engineering
- **Column Renaming**: The `disrict` column is corrected to `district`.
- **Categorical Data Cleaning**:
    - The `counter_statue` column is standardized by mapping various string and integer representations to a consistent set of values.
    - The `counter_type` column is binary encoded (`GAZ`: 0, `ELEC`: 1).
- **Feature Creation**:
    - **Time-Based Features**: `client_since`, `creation_month`, `creation_year`, and `is_weekday` are engineered from the `creation_date`.
    - **Categorical Features**: `region_group` is created by binning the `region` codes.
    - **Boolean Features**: `is_billed_level_1` through `is_billed_level_4` are created to indicate if a consumption level was billed.
- **Anomaly Detection**: An `IsolationForest` is used to identify and score anomalies in the dataset. This helps in potentially cleaning the data or creating an `anomaly` feature.

### 4. Modeling
- **Baseline Model**: A basic `DecisionTreeClassifier` is trained on a subset of the original features to establish a performance baseline.
- **Preprocessing Pipeline**: A `ColumnTransformer` is set up to handle numerical and categorical features separately.
    - **Numerical Pipeline**: Imputes missing values with the median and applies `StandardScaler`.
    - **Categorical Pipeline**: Imputes missing values with the most frequent value and applies `OneHotEncoder`.
- **Handling Imbalance**: `SMOTE` (Synthetic Minority Over-sampling Technique) is applied to the training data to address the class imbalance in the `target` variable.
- **LGBM Classifier**: An `LGBMClassifier` is chosen for its performance and efficiency.
    - It is first trained with default parameters and a custom prediction threshold to evaluate its effectiveness.
    - **Hyperparameter Tuning**: `RandomizedSearchCV` is used on a sample of the data to find the optimal hyperparameters for the LGBM model, using an F2-score to prioritize recall.

### 5. Prediction and Submission
- The best-performing, tuned model is used to predict fraud probabilities on the preprocessed test set (`df_test`).
- A `submission.csv` file is generated containing the `client_id` and the corresponding predicted `target` probabilities, ready for submission.
---
## Machine Learning Concepts
- **Classification**: A supervised learning task to predict a categorical label. Here, the model classifies consumption as fraudulent (`1`) or not (`0`).
- **Feature Engineering**: Creating new, informative features from existing data to improve model performance. Examples include `client_since` from dates and `is_billed` flags.
- **Data Preprocessing**:
    - **Imputation**: Filling missing values using median and most frequent strategies.
    - **Scaling**: Normalizing numerical features with `StandardScaler`.
    - **One-Hot Encoding**: Converting categorical features into a numerical format.
- **Handling Imbalanced Data**: Using `SMOTE` (Synthetic Minority Over-sampling Technique) to balance the dataset by creating synthetic examples of the minority (fraud) class.
- **Ensemble Learning (Gradient Boosting)**: Employing `LightGBM`, which builds a strong model by sequentially combining weak decision trees, each correcting the errors of the previous one.
- **Anomaly Detection**: Using `IsolationForest` to identify unusual data points that could be errors or a distinct type of fraud.
- **Hyperparameter Tuning**: Using `RandomizedSearchCV` to find the optimal model parameters, optimizing for the F2-score to prioritize recall.
- **Model Evaluation**: Assessing performance with metrics like the **Classification Report** (precision, recall, F1-score), **Confusion Matrix**, and **ROC-AUC Score**.

---
## Tech Stack
- **Python**: The core programming language used for the project.
- **Jupyter Notebook**: For interactive development and analysis.
- **Pandas & NumPy**: For data manipulation and numerical operations.
- **Scikit-learn**: For machine learning tasks including preprocessing, modeling, and evaluation.
- **LightGBM**: The gradient boosting framework used for the final model.
- **Imbalanced-learn**: For handling imbalanced data with SMOTE.
- **Matplotlib & Seaborn**: For data visualization.

---
---
## Set up your Environment



### **`macOS`** type the following commands : 



- For installing the virtual environment and the required package you can either follow the commands:

    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
Or ....
-  use the [Makefile](Makefile) and run `make setup` or install it manually with the following commands:

     ```BASH
    make setup
    ```
    After that active your environment by following commands:
    ```BASH
    source .venv/bin/activate
    ```

### **`WindowsOS`** type the following commands :

- Install the virtual environment and the required packages by following commands.

   For `PowerShell` CLI :

    ```PowerShell
    pyenv local 3.11.3
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    For `Git-bash` CLI :
  
    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/Scripts/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    **`Note:`**
    If you encounter an error when trying to run `pip install --upgrade pip`, try using the following command:
    ```Bash
    python.exe -m pip install --upgrade pip
    ```


---
