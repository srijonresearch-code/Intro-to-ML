# Intro to Machine Learning (Kaggle)
 
I created two notebooks for this course, using the Melbourne Housing Snapshot dataset.
 
## Files
 
- `first_ml_model.ipynb` — building and evaluating a basic model
- `overfitting_underfitting.ipynb` — comparing model complexity using MAE
- `melb_data.csv` — dataset used in both notebooks
## What I learned
 
### 1. Loading and exploring data
- Read a CSV with `pd.read_csv()`
- Checked the data with `.head()`, `.describe()`, `.info()` and `.columns`
### 2. Handling missing values (basic)
- Dropped rows with missing values using `dropna(axis=0)`
### 3. Selecting target and features
- Target (`y`): the column I want to predict — `Price`
- Features (`X`): the columns used to predict it — `Rooms`, `Bathroom`, `Landsize`, `Lattitude`, `Longtitude`
### 4. Splitting data
- Used `train_test_split()` to split data into training and validation sets, so the model can be tested on data it hasn't seen
### 5. Building a model
- Used `DecisionTreeRegressor` from scikit-learn
- Trained it with `.fit(train_X, train_y)`
- Made predictions with `.predict(val_X)`
### 6. How models actually work
- Understood the basic flow: a model learns patterns from `train_X`/`train_y` during `.fit()`, then uses those patterns to guess outputs for new, unseen data with `.predict()`
### 7. Measuring error
- Used `mean_absolute_error(val_y, val_prediction)` to check how far off the predictions are, on average
### 8. Overfitting vs underfitting
- Wrote a `get_mae()` function to test different `max_leaf_nodes` values
- Compared MAE across tree sizes (5, 50, 500, 5000 leaf nodes)
- Saw that:
  - Too few leaf nodes → underfitting (model too simple, high error)
  - Too many leaf nodes → overfitting (model memorizes training data, error goes back up)
  - There's a middle value that gives the best (lowest) validation MAE
## Note
I already knew pandas before this, but working through these notebooks made me a lot more confident using it for real data (filtering columns, checking data with `.describe()`/`.head()`, handling missing values, etc.).
 
## Dataset
[Melbourne Housing Snapshot](https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot) from Kaggle.
