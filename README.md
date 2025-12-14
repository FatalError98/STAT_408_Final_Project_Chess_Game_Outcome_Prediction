# Chess Game Outcome Prediction

---

## 📌 Overview

This project analyzes a large dataset of online chess games and builds a predictive statistical model to estimate the probability that **White wins** based on game characteristics such as:

- Rating differences  
- Opening choice  
- Material imbalances  
- Tactical indicators (checks, captures)

The final model uses **Elastic Net (LASSO)** regularization to select the most informative predictors.

---

## 📂 Dataset Description

The dataset includes **117,148 chess games** collected from the *Lichess* online platform.

Each row represents one completed game and includes:

| Variable | Description |
|---------|-------------|
| `game_type` | Time control: Classical, Blitz, Bullet, Correspondence |
| `whiteWin` | Binary outcome: 1 = White wins, 0 = Black wins |
| `opening` | Categorical: 269 unique openings |
| `elo_diff` | White ELO − Black ELO |
| `pawn_count_diff` | Pawn count difference |
| `bishop_count_diff` | Bishop count difference |
| `knight_count_diff` | Knight count difference |
| `rook_count_diff` | Rook count difference |
| `queen_count_diff` | Queen count difference |
| `captured_score_diff` | Material score difference (1/3/5/9 values) |
| `capture_diff` | Total captured pieces difference |
| `check_diff` | Difference in number of checks |

---

## 🧹 Data Cleaning & Preparation

Key preprocessing steps:

- Removed games that ended in a **draw** to maintain a binary outcome.  
- Converted categorical variables (`game_type`, `opening`) into factors.  
- Removed rows with missing outcome or missing rating information.  
- Created several material advantage variables (queen, rook, bishop, knight, pawn differences).  
- Generated tactical variables such as `check_diff`, `captured_score_diff`, `capture_diff`.

### Summary Observations

- Players were mostly evenly matched (mean ~0 rating difference).  
- **Blitz** was the most common game type.  

---

## 📊 Model Building

### Logistic Regression Model

The response variable is:

```
whiteWin = 1  if White wins
whiteWin = 0  otherwise
```

Initial logistic regression model:


$$
logit(p) = β0 + β1 gameType + β2 opening + β3 eloDiff +
           β4 pawnCountDiff + β5 knightCountDiff +
           β6 bishopCountDiff + β7 rookCountDiff +
           β8 queenCountDiff + β9 capturedScoreDiff +
           β10 capturedDiff + β11 checkDiff
$$

---

### 🕵️ Influential Observations

- Identified using studentized residuals with α = 0.0001.  
- Several rare but valid observations were found.  
- **None were removed**.

---

### ⚠️ Multicollinearity Check

Using VIF revealed strong correlations among:

- pawn_count_diff  
- queen_count_diff  
- bishop_count_diff  
- knight_count_diff  
- rook_count_diff  
- captured_score_diff  

To fix this, the model moved to **Elastic Net (LASSO)** to shrink/remove redundant predictors.

---

## 🔧 Elastic Net + Cross Validation

- 5-fold CV used with α values from 0 to 1.  
- **Best α = 1 → pure LASSO**  
- About half of the opening levels were removed.  
- Multicollinearity was resolved.

Final selected model:

$$
logit(p) = β0 + β1 gameType + β2 opening + β3 eloDiff +
           β4 queenCountDiff + β5 bishopCountDiff +
           β6 capturedScoreDiff + β7 capturedDiff +
           β8 checkDiff
$$

---

## 🧪 Prediction Example

A constructed game with:

| Variable | Value |
|---------|-------|
| elo_diff | +8 |
| pawn_count_diff | +1 |
| queen_count_diff | +1 |
| rook_count_diff | -2 |
| captured_score_diff | 0 |
| capture_diff | 0 |
| check_diff | +1 |
| opening | Italian Game |

**Predicted probability White wins:**  
**0.673089 → 67.30%**

---

## 📈 Interpretation of Key Coefficients

Model:

$$
logit(p) = 0.1878 − 0.1229 gameTypeClassical + 0.0037 eloDiff +
           0.1973 queenCountDiff + 0.0763 bishopCountDiff +
           0.261 capturedScoreDiff + 0.0928 capturedDiff +
           0.0984 checkDiff − 0.1101 openingItalianGame + ...
$$

### Highlights:

- **Queen advantage (0.1973):** Strongly increases White’s chances.
- **Rating advantage (0.0037 per point):** Positive but small effect.
- **Classical games (−0.1229):** White performs slightly worse than in Blitz.

---

## 📝 Evaluation & Discussion

### ✔️ Strengths

- LASSO effectively removed redundant predictors.  
- Rating, material, and tactical variables significantly improve prediction.  
- Opening selection plays a moderate role.

### ❗ Limitations

- Dataset captures only final board states.  
- Pawn advantages were underestimated by LASSO.  
- Draws excluded — multinomial models could provide more insight.

---
