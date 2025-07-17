# 🎾 Tennis Match Outcome Predictor

This project predicts the outcome of ATP tennis matches using advanced **machine learning techniques** like **XGBoost** with strong **feature engineering**, including player stats, head-to-head history, surface-based performance, and serve data.

## 📂 Dataset

- Source: ATP match statistics  
- Format: CSV file with match-level stats  
- Included file: `data.csv`

---

## 📌 Features Engineered

| Feature Type       | Description |
|--------------------|-------------|
| `POINT_DIFF`       | Difference in ranking points between winner and loser |
| `RANK_DIFF`        | Difference in ranks |
| `HT_DIFF`          | Height difference |
| `ACE_DIFF`, `DF_DIFF` | Serve ace and double fault differences |
| `FIRST_SERVE_PCT_DIFF` | First serve in % difference |
| `FIRST_SERVE_WIN_DIFF` | First serve win % difference |
| `SECOND_SERVE_WIN_DIFF` | Second serve win % difference |
| `BP_SAVE_PCT_DIFF` | Break point save % difference |
| `H2H_WIN`, `H2H_WIN_SURFACE` | Historical wins against opponent overall and on current surface |
| `BEST_OF`, `DRAW_SIZE` | Match metadata |
| `RESULT`           | 1 = player1 wins, 0 = player2 wins |

---

## 🧠 Model

- **Algorithm**: `XGBoostClassifier`
- **Accuracy**:
  - Train: 94.9%
  - Test: 92.9%
  - ROC AUC Score: 0.9823
- **Validation**: Stratified K-Fold cross-validation
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score, ROC AUC

---

## 📈 Visualizations

- Pairplots of features vs target
- SHAP values for feature importance
- Confusion matrix, ROC curve

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/tennis-match-predictor.git
cd tennis-match-predictor

