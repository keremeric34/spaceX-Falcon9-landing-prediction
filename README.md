# SpaceX Falcon 9 First Stage Landing Prediction

An end-to-end data science project that predicts whether a Falcon 9 first stage will land successfully. Reusing the first stage is what makes SpaceX launches cheaper, so predicting the landing helps estimate launch cost.

## Questions

1. How did landing success change over time?
2. Which factors (orbit, payload mass, launch site, booster version) affect the outcome?
3. Which classification model predicts landings best?

## Pipeline

| Part | What it does |
|------|--------------|
| 1. Web scraping | Collects Falcon 9 launch records from a pinned Wikipedia revision (`requests` + `BeautifulSoup`) |
| 2. Data wrangling | Cleans the data and builds the binary `Class` label from the landing `Outcome` |
| 3. SQL analysis | 10 SQLite queries (launch sites, payload totals, first ground-pad landing, mission outcomes…) |
| 4. EDA and feature engineering | Seaborn/matplotlib plots and one-hot encoding |
| 5. Interactive map | Folium map with launch sites, success/failure markers and distances to coast, highway, railway and city |
| 6. Machine learning | Logistic regression, SVM, decision tree and KNN tuned with `GridSearchCV` (cv = 10) |

## Results

- 90 launches, 60 successful first-stage landings (66.7%)
- All four models reach 83.33% test accuracy (15/18) on an 80/20 split (`random_state=2`)
- The test set is tiny (18 samples), so cross-validation scores are more reliable for comparing models

## Getting started

```bash
pip install -r requirements.txt
python spacex_falcon9_tam_proje.py
```

The script runs all six parts in order and writes `spacex_web_scraped.csv`, `dataset_part_2.csv`, `dataset_part_3.csv` and `spacex_launch_map.html`. Plots open in blocking windows; close each one to continue. The hyperparameter searches (especially SVM and decision tree) can take a few minutes.

## Repository structure

```
spacex_falcon9_tam_proje.py   # combined, runnable version of all six parts
requirements.txt
README.md
```

## Data sources

- Wikipedia: List of Falcon 9 and Falcon Heavy launches (pinned revision)
- IBM Data Science capstone datasets (`dataset_part_1/2/3.csv`, `Spacex.csv`, `spacex_launch_geo.csv`)
