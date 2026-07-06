# Instagram Post Engagement — Data Analysis

Final assignment for Fundamentals of AI and Data Analytics — a small end-to-end AI/Data project analyzing engagement on a personal Instagram account, using data cleaning, exploratory analysis, and machine learning classification.

**Author:** Agrima Sharma
**Date:** July 2026

---

## Dataset

This dataset contains 140 of my own Instagram posts, pulled via the Meta Graph API, spanning January 2024 to July 2026. It covers post metadata (media type, caption, timestamp) and engagement metrics (likes, comments, reach, saves, views).

## Research Question

Which post factors — media type, posting time, caption length, reach, saves, views — most strongly predict whether a post gets High or Low engagement (likes), and can this be predicted reliably?

## Repo Contents

| File | Description |
|---|---|
| `instagram_engagement_analysis.ipynb` | Full analysis notebook |
| `instagram_posts.csv` | Raw dataset (140 posts) |
| `instagram_posts_cleaned.csv` | Cleaned dataset used for modeling |
| `Instagram_Final_Assignment_Report.docx` | Written report covering the full rubric |
| `Instagram_Engagement_Presentation_UPDATED.pptx` | 10-slide presentation deck |
| `Instagram_Presentation_Script.md` | Timed ~15-minute speaker script |

## Notebook Structure

1. **Dataset** / **Research Question**
2. **Loading libraries**
3. **Loading Data Set**
4. **Table dimensions**, `head`/`tail`, `info`, `isna`, `describe`, `duplicated` + `value_counts`
5. **Cleaning the data set**
6. **Visualizations**
   - Distribution of likes
   - Average Likes by Media Type
   - Comments vs Likes
   - Scatterplot (Views vs Reach)
   - Views by Media Type
   - Average Reel Views by Month
   - Box With Whisker — Video posts vs. Non-video posts
   - High vs Low engagement posts
   - Caption length vs. Engagement
   - Correlation Heat Map
7. **Statistical test** — chi-square (media_type vs. engagement) and t-test on views (High vs. Low engagement)
8. **Training data and Test sets**
9. **Which model actually fits this data best?** — six classifiers run individually:
   - RandomForestClassifier
   - k-nearest neighbors (KNN) method
   - Ada boost model
   - XGBoost Model
   - SVC
   - Logistic Regression
10. **Which model actually fits best here — the verdict**
11. **Conclusion**

## Results

| Model | Accuracy | Precision | Recall |
|---|---|---|---|
| Random Forest | 85.7% | 91.7% | 78.6% |
| KNN (k=5) | 85.7% | 91.7% | 78.6% |
| Logistic Regression | 85.7% | 91.7% | 78.6% |
| **XGBoost** | **85.7%** | **85.7%** | **85.7%** |
| AdaBoost | 82.1% | 84.6% | 78.6% |
| SVC (RBF) | 50.0% | 50.0% | 100.0% |

**XGBoost is the best fit** for this dataset — it matches the top accuracy with the most balanced precision and recall. **SVC collapsed entirely** without feature scaling, predicting every post as "High engagement."

## Conclusion (from the notebook)

**Research question:** which post factors most strongly predict whether an Instagram post gets High or Low engagement, and can this be predicted reliably? Yes — with the important caveat that "reliably" here means on a 140-post, self-collected dataset, not a production-scale one.

**Key findings from the EDA:**
- Likes are right-skewed — a handful of breakout posts pull the mean well above the median.
- The account is heavily Reels-first (78% of posts are VIDEO), and views is a Reels-only metric — IMAGE and CAROUSEL_ALBUM posts don't get a views count from Instagram at all.
- Reach and views are the strongest numeric drivers of likes; comments are the weakest, skewed by a single outlier post.
- Media type alone is only weakly/borderline associated with engagement (chi² test, p ≈ 0.052); caption length shows no significant difference between High and Low engagement posts (t-test, p ≈ 0.31).
- Views, on the other hand, is a genuinely strong and statistically significant signal (t-test, p < 0.001) — High-engagement posts average ~8,362 views vs. ~2,275 for Low-engagement posts.

**Model performance:** six classifiers were compared on the same train/test split. Random Forest, KNN, and Logistic Regression tied at 85.7% accuracy; XGBoost matched that accuracy with a more balanced precision/recall (85.7%/85.7%), making it the best fit overall; AdaBoost trailed slightly (82.1%); and SVC collapsed entirely (50%) without feature scaling — the clearest demonstration in the notebook of why distance/margin-based models need scaling, especially on a small dataset.

**What this means practically:** views, reach, and saves — not media type or caption length or comments — are what actually separates a High-engagement post from a Low one on this account, and a simple model can already predict that split correctly about 6 times out of 7.

**Limitations:** 140 posts is a small sample for machine learning — results (especially the model comparison) may shift meaningfully as more posts are collected. The High/Low split is based on this dataset's own median and will move as the account grows. Caption *content* (beyond length) wasn't analyzed. Views can't be compared across media types since Instagram only tracks it for Reels. This analysis should be re-run periodically as new data comes in, rather than treated as a one-time answer.

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost scipy
jupyter notebook instagram_engagement_analysis.ipynb
```

Run all cells top to bottom — the notebook regenerates every table, chart, and model result from `instagram_posts.csv`.
