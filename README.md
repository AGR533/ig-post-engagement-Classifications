# Instagram Post Engagement — Data Analysis

**Author:** Agrima Sharma
**Date:** July 2026

## Dataset

70 of my own Instagram posts, pulled via the Meta Graph API, spanning December 2024 to July 2026. Covers post metadata (media type, caption, timestamp) and engagement metrics (likes, comments, reach, saves, views).

| Column | Description |
|---|---|
| `id` | Unique post ID |
| `timestamp` | Date/time posted |
| `media_type` | VIDEO / IMAGE / CAROUSEL_ALBUM |
| `caption` | Post caption text |
| `like_count` | Number of likes |
| `comments_count` | Number of comments |
| `reach` | Accounts reached |
| `saved` | Number of saves |
| `views` | Number of views (Reels only — Instagram doesn't report this for IMAGE/CAROUSEL_ALBUM) |
| `permalink` | Link to the post |

## Research Question

Which post factors — media type, posting time, caption length, reach, saves, views — most strongly predict whether a post gets High or Low engagement (likes), and can this be predicted reliably?

## Files

- `instagram_engagement_analysis.ipynb` — full analysis notebook
- `instagram_posts.csv` — raw dataset (70 real posts)
- `README.md` — this file

## Methodology

1. **Data Cleaning** — checked missing values (caption: 6, reach: 3, saved: 3, views: 10) and duplicates; `views` gaps filled with 0 for IMAGE/CAROUSEL_ALBUM (metric doesn't exist for those formats, not truly zero); `reach`/`saved` gaps filled with median to avoid outlier-driven mean bias.
2. **EDA** — distribution plots, average likes/views by media type, comments-vs-likes scatter, views-vs-reach scatter, monthly Reel view trends, box plots, correlation heatmap, media-type breakdown pie chart.
3. **Feature Engineering** — `caption_length`, `hour`, `is_weekend`, one-hot encoded `media_type`, binary `engagement_label` (High/Low, median-split, balanced 35/35).
4. **Statistical Testing** — chi-square test (media type vs. engagement), independent t-test (views: High vs. Low engagement).
5. **Modeling** — six classifiers compared on the same 80/20 stratified split (56 train / 14 test, `random_state=42`): Random Forest, KNN, AdaBoost, XGBoost, SVC, Logistic Regression (scaled).

## Results

| Model | Accuracy | Precision | Recall |
|---|---|---|---|
| **KNN (k=5)** | **92.9%** | 87.5% | **100.0%** |
| Random Forest | 85.7% | 85.7% | 85.7% |
| XGBoost | 85.7% | 85.7% | 85.7% |
| Logistic Regression | 85.7% | 100.0% | 71.4% |
| AdaBoost | 78.6% | 83.3% | 71.4% |
| SVC | 57.1% | 100.0% | 14.3% |

**Best model: KNN** — highest accuracy and the only model with perfect recall (never misses a genuine High-engagement post). Note: with only 14 test posts, this ranking should be read cautiously — a different split could reshuffle the four models scoring 85.7%+.

## Statistical Findings

- **Chi-square test** (media type vs. engagement): χ² ≈ 5.27, p ≈ 0.072 — not quite statistically significant; media type alone is a weak-to-borderline signal.
- **t-test** (views: High vs. Low engagement): t ≈ 2.91, p ≈ 0.005 — highly significant. High-engagement posts average ~9,217 views vs. ~2,274 for Low-engagement posts.
- **Correlation with likes:** reach (r ≈ 0.92) and views (r ≈ 0.85) are the strongest drivers; comments (r ≈ 0.18) are the weakest, skewed by one outlier post with 1,508 comments.

## Key Insights

1. **Likes are right-skewed** — mean (137.1) sits well above the median (103.5), pulled up by a handful of breakout posts.
2. **The account is heavily Reels-first** (85.7% VIDEO) — `views` is effectively a Reels-only metric.
3. **Views, reach, and saves — not media type, caption length, or comments — actually separate High from Low engagement**, and the best model predicts that split correctly ~13 times out of 14.

## Limitations

70 posts is a small sample for machine learning; results (especially the model ranking) may shift as more data is collected. The High/Low split is based on this dataset's own median and will move as the account grows. Caption *content* (beyond length) wasn't analyzed. Views can't be compared across media types since Instagram only tracks it for Reels.

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy xgboost joblib
jupyter notebook instagram_engagement_analysis.ipynb
```

Run all cells top to bottom; `instagram_posts.csv` must be in the same directory as the notebook.

## Author

Agrima Sharma (Aera) — Bachelor of AI and Data Analysis, MDH Berlin
