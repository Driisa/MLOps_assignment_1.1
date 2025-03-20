# Penguins of Madagascar Detector

![Penguins of Madagascar](images/penguins.jpg)

## Project Overview

This MLOps project aims to find Skipper, Private, Rico, and Kowalski - the Penguins of Madagascar! These four penguins are Adelie penguins that have escaped to New York City. We're trying to identify them based on physical measurements.

Our mission is to build a classification system that analyzes penguin measurements (bill length, bill depth, flipper length, and body mass) and identifies whether a detected penguin is an Adelie - potentially one of our Madagascar friends!

Every day at 7:30 AM, the system automatically fetches data about a new penguin spotted in NYC, makes a prediction about its species, and updates the results on our project's GitHub Pages.

## Repository Structure

```
.
├── .github
│   └── workflows
│       └── daily_prediction.yml    # GitHub Actions workflow for daily predictions
├── data
│   ├── predictions                 # Prediction results
│   │   ├── history.json            # History of all predictions
│   │   ├── latest.json             # Most recent prediction
│   │   ├── latest_visualization.png # Visualization of latest prediction
│   │   └── index.html              # GitHub Pages dashboard
│   └── penguins.db                 # SQLite database with penguin data
├── models
│   ├── adelie_detector.pkl         # Dedicated model for Adelie detection
│   ├── best_model.pkl              # Best performing species classifier
│   └── model_metrics.json          # Performance metrics of trained models
├── scripts
│   ├── data-processing-script.py   # Data loading and database creation
│   ├── model-training-script.py    # Model training and evaluation
│   └── prediction-api-script.py    # Script for making daily predictions
├── requirements.txt                # Python dependencies
└── workflow-diagram.mermaid        # Workflow diagram
```

## Technical Implementation

### 1. Data Processing (Task 2)

The `data-processing-script.py` handles:
- Loading penguin dataset from seaborn
- Cleaning data by removing rows with missing values
- Creating an SQLite database with two tables:
  - `PENGUINS`: Contains penguin measurements and species
  - `ISLANDS`: Lookup table for island information
- Adding appropriate indices for better query performance

**Database Schema:**

```
PENGUINS
- species (TEXT)
- bill_length_mm (REAL)
- bill_depth_mm (REAL)
- flipper_length_mm (REAL)
- body_mass_g (REAL)
- sex (TEXT)
- island_id (INTEGER) - Foreign key
- animal_id (INTEGER) - Primary key

ISLANDS
- island_id (INTEGER) - Primary key
- name (TEXT)
```

### 2. Model Training (Task 3)

The `model-training-script.py` handles:
- Loading data from the SQLite database
- Feature selection using only API-available features: bill length, bill depth, flipper length, and body mass
- Data preprocessing with StandardScaler
- Training and evaluating multiple models:
  - RandomForest classifier for general species classification
  - XGBoost classifier for general species classification
  - A dedicated RandomForest classifier optimized specifically for Adelie detection
- Hyperparameter tuning using GridSearchCV
- Model evaluation using accuracy, precision, recall, and F1 score
- Saving the best performing models and their metrics

### 3. Daily Prediction (Task 4)

The `prediction-api-script.py` handles:
- Fetching new penguin data from the API endpoint: http://130.225.39.127:8000/new_penguin/
- Loading trained models from disk
- Making predictions using both the species classifier and Adelie detector
- Creating visualizations of the prediction results
- Saving prediction results to JSON files
- Updating prediction history
- Generating HTML content for GitHub Pages

### 4. Workflow Automation

The GitHub Actions workflow in `daily_prediction.yml` automates:
- Running the prediction script daily at 7:30 AM UTC
- Committing and pushing updated prediction files to the repository
- Deploying the updated GitHub Pages site

## Technologies Used

- **Python 3.9** - Core programming language
- **Data Processing & Analysis**:
  - pandas - Data manipulation
  - numpy - Numerical operations
  - seaborn - Data source and visualization
  - matplotlib - Data visualization
- **Database**:
  - SQLite3 - Lightweight database
- **Machine Learning**:
  - scikit-learn - Model training and evaluation
  - XGBoost - Gradient boosting framework
  - joblib - Model serialization
- **Web & API**:
  - requests - API interactions
- **DevOps & Automation**:
  - GitHub Actions - Workflow automation
  - GitHub Pages - Result visualization

## How to Run

1. Clone the repository:
   ```
   git clone https://driisa.github.io/MLOps_assignment_1/
   cd penguins-of-madagascar
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run data processing script to create the database:
   ```
   python scripts/data-processing-script.py
   ```

4. Train the models:
   ```
   python scripts/model-training-script.py
   ```

5. Run prediction manually (normally handled by GitHub Actions):
   ```
   python scripts/prediction-api-script.py
   ```

## Results

The project creates a daily updated GitHub Pages site with:
- Latest penguin species prediction
- Visualization of the penguin's measurements
- History of recent predictions
- Statistics on detected Adelie penguins

You can view the live results at: https://Driisa.github.io/penguins-of-madagascar/

## Contributing

This project was created as part of an MLOps university assignment. Contributions, suggestions, and improvements are welcome!

## License

This project is available under the MIT License.
