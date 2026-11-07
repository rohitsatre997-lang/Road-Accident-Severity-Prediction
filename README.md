# Road Accident Severity Prediction Using Machine Learning

A Google Colab-based machine learning project that analyzes historical
road accident records and predicts the **severity class of a recorded
accident**.

> **Important scope note:** The supplied dataset contains accident
> records and the target column `Accident_Severity_C`. Therefore, this
> project performs *accident severity classification*. It does not
> predict whether an accident will occur before it happens.

## Project Overview

Road accidents can result in injuries, fatalities, and economic losses.
This project uses historical accident data to study patterns and build a
classification model that predicts the severity category using recorded
accident attributes.

### Objectives

-   Understand and inspect the road accident dataset.
-   Clean the data and handle missing values and exact duplicate
    records.
-   Explore accident patterns using statistical summaries and
    visualizations.
-   Engineer useful time-related features from date and time fields.
-   Train and compare machine learning classification algorithms.
-   Evaluate model performance using appropriate classification metrics.
-   Present model interpretation, limitations, and road-safety
    recommendations.

## Dataset

**Dataset file:** `ETP_4_New_Data_Accidents.csv`

The dataset includes fields such as:

-   Date and day of week
-   Time of accident
-   Accident location and chainage
-   Accident severity (`Accident_Severity_C`) --- target variable
-   Accident causes
-   Road features and road conditions
-   Weather conditions
-   Vehicle types involved

Some columns contain coded category values. Their exact meanings should
be interpreted using the official dataset codebook. Do not assume
meanings for numeric codes if a codebook is unavailable.

The dataset also contains missing values, including substantial
missingness in `Vehicle_Type_Involved_J_V2`, and repeated complete rows.
The notebook reports these issues and applies preprocessing steps.
Confirm with the dataset provider whether repeated records are
accidental duplicates or part of the original data design.

## Machine Learning Workflow

1.  **Data loading:** Upload the CSV dataset in Google Colab.
2.  **Data understanding:** Inspect dimensions, data types, descriptive
    statistics, unique values, and missing values.
3.  **Data cleaning:** Standardize column names, investigate exact
    duplicates, and remove exact duplicate rows.
4.  **Exploratory Data Analysis:** Visualize severity distribution and
    accident patterns across available fields.
5.  **Feature engineering:** Extract year, month, day, hour, and minute
    from date/time fields where parsing is possible.
6.  **Preprocessing:** Impute missing values, scale numerical features,
    and one-hot encode categorical features.
7.  **Model training:** Compare a Dummy Classifier baseline, Logistic
    Regression, Random Forest, and Extra Trees.
8.  **Validation and tuning:** Use stratified cross-validation and
    randomized hyperparameter search for Random Forest.
9.  **Evaluation:** Review accuracy, balanced accuracy, macro F1,
    weighted F1, classification report, and confusion matrix.
10. **Interpretation:** Calculate permutation feature importance and
    discuss business implications.

## Algorithms Used

-   **Dummy Classifier:** Baseline for comparison.
-   **Logistic Regression:** Multi-class classification baseline.
-   **Random Forest Classifier:** Ensemble of decision trees.
-   **Extra Trees Classifier:** Randomized tree ensemble.
-   **Tuned Random Forest:** Randomized hyperparameter search using
    macro F1 as the selection metric.

The final selected model is chosen using training cross-validation
results. The held-out test set is used for final evaluation.

## Evaluation Metrics

-   **Accuracy:** Overall proportion of correct predictions.
-   **Balanced Accuracy:** Average recall across classes.
-   **Macro F1-score:** Treats each severity class equally.
-   **Weighted F1-score:** Weights class F1-scores by their support.
-   **Confusion Matrix:** Shows correct and incorrect predictions for
    each class.
-   **Per-class Recall:** Shows how well the model identifies each
    severity class.

Because severity classes may be imbalanced, accuracy alone is not
sufficient. Pay particular attention to macro F1 and per-class recall.

## How to Run in Google Colab

1.  Download `Road_Accident_Severity_Prediction_PBL_Improved.ipynb`.
2.  Open [Google Colab](https://colab.research.google.com/).
3.  Select **File → Upload notebook** and upload the `.ipynb` file.
4.  Choose **Runtime → Run all**.
5.  When prompted, upload `ETP_4_New_Data_Accidents.csv`.
6.  Wait for all cells to finish and review the generated tables and
    charts.
7.  Save a copy of the executed notebook with outputs.

The notebook uses `google.colab.files.upload()` for dataset upload. Run
it in Google Colab rather than a standard local Jupyter environment
unless you adapt the upload cell.

## Project Files

``` text
Road-Accident-Severity-Prediction/
│
├── Road_Accident_Severity_Prediction_PBL_Improved.ipynb
├── ETP_4_New_Data_Accidents.csv
└── README.md
```

## Limitations

-   The model predicts severity for records representing accidents that
    have already occurred; it is not a pre-crash accident-risk model.
-   Several dataset fields are coded categories. An official codebook is
    needed to interpret the codes correctly.
-   Duplicate records and missing values should be confirmed with the
    dataset source.
-   Retrospective test performance does not guarantee performance on
    future data, new locations, or different road conditions.
-   Some recorded fields, such as accident cause or vehicle involvement,
    may only be known after an accident. Their availability must be
    considered before claiming real-time predictive use.
-   The model is an academic decision-support prototype and should not
    be used for emergency decisions without external validation and
    appropriate safeguards.

## Future Improvements

-   Obtain and document the official dataset codebook.
-   Validate the model using a later time period or an independent
    dataset.
-   Investigate class imbalance and improve recall for underrepresented
    severity classes.
-   Add only features that are available at the intended prediction
    time.
-   For genuine pre-crash risk prediction, collect suitable
    non-accident/exposure observations and relevant traffic, speed,
    lighting, and road-environment data.
-   Review model performance with road-safety domain experts before any
    operational deployment.

## Technologies

-   Python
-   Google Colab / Jupyter Notebook
-   Pandas and NumPy
-   Matplotlib and Seaborn
-   Scikit-learn
-   Machine Learning: classification, preprocessing pipelines,
    cross-validation, hyperparameter tuning

## Academic Note

This project is prepared for educational and Problem-Based Learning
(PBL) purposes. Explain the dataset, preprocessing decisions, model
choices, evaluation results, and limitations during the project
presentation and viva.
