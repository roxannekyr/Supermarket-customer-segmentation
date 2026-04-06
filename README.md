# Supermarket Customer Segmentation with KMeans Clustering

Identifying high-value customer groups to power targeted marketing campaigns using unsupervised machine learning.

## Business Problem

A supermarket collects basic customer data through membership cards. The marketing team needs to know **which customers to target and how** but without clear segments, they're spending budget on everyone equally.

This project segments customers into distinct groups based on spending behaviour, income, and age, and hands off **actionable cluster profiles** directly usable for campaign planning.

## Insights

| Cluster | Profile | Avg Spending Score | Avg Income | Count |
|---|---|---|---|---|
| 🟢 Priority 1 | High Income · High Spending · Adults (~33y) | 81.5 | ~$85k | 39 |
| 🟡 Priority 2 | Low Income · High Spending · Teenagers (~26y) | 60.0 | ~$40k | 57 |
| 🟠 Priority 3 | Low Income · Low Spending · Retired (~54y) | 40.0 | ~$48k | 65 |
| 🔴 Do Not Target | High Income · Very Low Spending · Middle-Aged (~40y) | 20.0 | ~$85k | 37 |

**Marketing recommendation:** Prioritise Cluster 1 for premium campaigns, Cluster 2 for value/discount campaigns, and deprioritise Cluster 4 — they earn more but choose not to spend here.

## Analytical Approach

1. Exploratory Data Analysis
- Full automated EDA function covering shape, dtypes, nulls, duplicates, value counts, and summary statistics
- Correlation heatmap + pairplot revealed spherical cluster structure → justified KMeans as the right algorithm
- Age shows a -0.33 correlation with Spending Score (younger customers spend more); Gender correlation was negligible (-0.06)

2. Preprocessing
- IQR-based outlier removal on Annual Income
- One-Hot Encoding for Gender
- StandardScaler applied before clustering (essential for distance-based algorithms to prevent income scale dominating)

3. Feature Selection
- Features used: `Annual_Income`, `Spending_Score`, `Age`
- Gender excluded — correlation analysis showed it had minimal predictive value for segmentation

4. Finding Optimal k
Two complementary methods were applied and compared:

- **Elbow Method** → suggested k=4 or k=6 as inflection points
- **Silhouette Score** (tested up to k=15) → k=4 returned the highest score (~0.44), confirming cluster cohesion

> Both methods were run as modular, reusable functions rather than inline code.

5. Model Comparison
Three models were built and evaluated side by side (k=4, k=5, k=6):

- k=5 and k=6 produced **two nearly identical low-spending clusters** differing only in income — not meaningful for a supermarket context
- k=4 produced the cleanest, most actionable segmentation with no redundant clusters
- **Final selection: k=4** — supported by Silhouette Score and business logic

6. KMeans Configuration

## Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `Matplotlib` · `Seaborn`

## What I'd Improve With More Time

- Test HDBSCAN for non-spherical cluster validation
- Add a spend-uplift simulation to estimate revenue impact of targeting each cluster
- Build a simple Streamlit dashboard for the marketing team to explore segments interactively
