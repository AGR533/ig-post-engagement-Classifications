# Instagram Post Engagement Classification

**Assignment 3 – Mini AI/Data Project**
Bachelor of Artificial Intelligence and Data Analysis, Media Design Hochschule (MDH) Berlin

## Problem Definition

Predict whether an Instagram post will get **High** or **Low** engagement (likes) based on posting time, caption features, and media type. Understanding what drives engagement helps content creators optimize their posting strategy with evidence instead of guesswork.

## Dataset

- **Source:** Instagram Graph API (own account)
- **Rows:** 70 posts
- **Columns:** 10

| Column | Description |
|---|---|
| `id` | Unique post ID (numeric) |
| `timestamp` | Date/time posted (datetime) |
| `media_type` | VIDEO / IMAGE / CAROUSEL_ALBUM (categorical) |
| `caption` | Post caption text |
| `like_count` | Number of likes |
| `comments_count` | Number of comments |
| `reach` | Accounts reached |
| `saved` | Number of saves |
| `views` | Number of views |
| `permalink` | Link to the post |

## Files

- `instagramanalysis.ipynb` — full analysis notebook (EDA, cleaning, feature engineering, model training, evaluation)
- `instagram_posts.csv` — raw dataset (70 real posts)
- `README.md` — this file

## Methodology

1. **Data Cleaning** — checked missing values, duplicates, and inconsistent categories; flagged 4 like-count outliers using the IQR method.
2. **Exploratory Data Analysis** — descriptive statistics (mean, median, mode, range, std dev) for likes, comments, reach, saves, and views; distribution and category breakdowns.
3. **Feature Engineering** — derived `caption_length`, `hour` (posting hour), `is_weekend`, and a binary `engagement_label` (High/Low, median-split, balanced 35/35).
4. **Modeling** — Decision Tree Classifier (`max_depth=4`, `random_state=42`), 80/20 stratified train-test split (56 train / 14 test).

## Results

- **Model:** Decision Tree Classifier
- **Accuracy:** 92.9% (13 of 14 held-out posts correctly classified)
- **Strongest predictors:** `saved` and `views`

## Key Insights

1. **Likes are right-skewed** — the median (103.5) is a more realistic "typical" outcome than the mean (137.1), since a few high-performing posts pull the average up.
2. **Media format has a measurable, modest effect on likes** — CAROUSEL_ALBUM posts average the most likes (151), ahead of IMAGE (138) and VIDEO (136), though this is based on only 4 carousel posts.
3. **Comment count is an unreliable signal** — distorted by one extreme outlier post (1,508 comments), likely spam/bot activity rather than organic engagement.

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook instagramanalysis.ipynb
```

Run all cells top to bottom; `instagram_posts.csv` must be in the same directory as the notebook.

## Author

Agrima Sharma (Aera) — Bachelor of AI and Data Analysis, MDH Berlin
