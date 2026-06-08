# World Cup Prediction AI

AI-powered football match predictor using Transformer neural networks and Poisson distribution analysis.

A machine learning system that predicts international football match outcomes with probabilistic scoreline predictions. Combines historical match data, ELO ratings, and advanced deep learning to estimate expected goals and generate accurate match forecasts.

## Overview

This project implements a Transformer-based neural network trained on international football match results to predict match outcomes with quantified probabilities. The model analyzes team performance histories, accounts for ELO ratings, and uses Poisson statistics to generate realistic score predictions.

## Key Features

- **Advanced Transformer Architecture:** Multi-head self-attention mechanism captures temporal patterns in team performance.
- **Positional Encoding:** Chronological encoding preserves the temporal context of match sequences.
- **Poisson-Based Predictions:** Converts predicted expected goals (λ) into probability distributions for all possible scorelines.
- **ELO Rating System:** Implements dynamic team strength estimation for more accurate predictions.
- **Dual Dataset Support:** Trains on both national team and club-level match data.
- **Interactive GUI:** Streamlit-based interface for easy match predictions.
- **Live Prediction Engine:** Real-time prediction capability with Hebrew/English team name support.
- **Model Persistence:** Pre-trained model weights for immediate predictions without retraining.

## Project Structure

```plaintext
world_cup_prediction/
├── src/world_cup_prediction/     # Installable Python package
│   ├── config.py                   # Paths and hyperparameters
│   ├── elo.py                      # ELO rating calculations
│   ├── features.py                 # Shared match feature vectors
│   ├── model.py                    # Transformer architecture
│   ├── dataset.py                  # FootballDataset
│   ├── train.py                    # Training pipeline
│   ├── predictor.py                # Live match predictions
│   └── memory.py                   # Tournament state updates
├── app/
│   └── gui.py                      # Streamlit web interface
├── scripts/
│   └── download_club_data.py       # Club match data downloader
├── data/
│   ├── club_results.csv            # Club match historical data
│   ├── national_results.csv        # International match data (user-provided)
│   └── teams_mapping.json          # Hebrew/English team name mapping
├── artifacts/                      # Generated model outputs (gitignored)
│   ├── pretrained_clubs.pth
│   ├── world_cup_model.pth
│   └── tournament_state.pkl
├── requirements.txt
├── pyproject.toml
├── Dockerfile
└── README.md
```

## Model Architecture

### Transformer-Based Design

```plaintext
Team A Match History              Team B Match History
        ↓                                  ↓
 Input Projection                 Input Projection
        ↓                                  ↓
Positional Encoding               Positional Encoding
        ↓                                  ↓
Transformer Encoder               Transformer Encoder
        ↓                                  ↓
Extract Final State               Extract Final State
        ↓                                  ↓
                  Concatenate & Merge
                           ↓
                 Fully Connected Head
                           ↓
               Output: λ_team_a, λ_team_b
                           ↓
          Poisson PMF → Score Probabilities
```

### Key Components

| Component | Purpose |
|-----------|---------|
| Input Projection | Maps raw features to d_model dimensions |
| Positional Encoding | Adds temporal information to sequences |
| Transformer Encoder | Multi-head attention (4 heads) across match sequences |
| Output Head | Dense layers predicting expected goals (λ) |
| Softplus Activation | Ensures λ remains non-negative |

## Quick Start

### Installation

**Requirements:**

- Python 3.8+
- PyTorch
- Pandas, NumPy, SciPy
- Streamlit (for GUI)

**Setup:**

```bash
pip install -e .
```

### Migration from the old flat layout

If you have existing files at the project root from a previous version:

1. Move `national_results.csv` → `data/national_results.csv`
2. Move model artifacts to `artifacts/`:
   - `pretrained_clubs.pth`
   - `world_cup_final_model.pth` or `world_cup_model_weights.pth` → rename to `artifacts/world_cup_model.pth`
   - `tournament_state.pkl` → `artifacts/tournament_state.pkl`
3. Re-run training if your `tournament_state.pkl` was built with the old feature schema.

### Usage

#### 1. Train the Model

```bash
python -m world_cup_prediction.train
```

Trains the Transformer on historical match data and saves weights to `artifacts/world_cup_model.pth`.

#### 2. Web Interface (Recommended)

```bash
streamlit run app/gui.py
```

Opens an interactive web app for match predictions with bilingual team name support (Hebrew/English).

#### 3. Live Predictions (Programmatic)

```python
from world_cup_prediction import main_live_predictor

predictions = main_live_predictor("France", "England")
# Returns top 3 scoreline predictions with probabilities
```

#### 4. Download Club Data

```bash
python scripts/download_club_data.py
```

#### 5. Docker Deployment

```bash
docker build -t world-cup-prediction .
docker run world-cup-prediction
```

## Data Format

### Input CSV Structure

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| date | YYYY-MM-DD | Match date | 2023-11-14 |
| home_team | string | Home team name | France |
| away_team | string | Away team name | England |
| home_score | int | Goals scored by home team | 2 |
| away_score | int | Goals scored by away team | 1 |
| tournament | string | Match type (exclude "Friendly") | UEFA Euro |

### Data Processing Pipeline

- **Load:** Parse CSV with temporal sorting.
- **Filter:** Remove friendly matches, keep competitive tournaments only.
- **Validation:** Ensure complete match records with valid scores.
- **Sliding Window:** Create sequences of last N matches per team (default: 5).
- **Normalization:** Scale features for neural network input.

## Model Parameters

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| SEQ_LEN | 5 | Past matches per team in sequence |
| NUM_FEATURES | 5 | Input features (elo_diff, GF, GA, goal_diff, home_adv) |
| BATCH_SIZE | 32 | Training batch size |
| EPOCHS | 20 | Training iterations (15 club + 4 national) |
| d_model | 64 | Transformer embedding dimension |
| nhead | 4 | Number of attention heads |
| num_layers | 2 | Transformer encoder layers |
| dropout | 0.1 | Regularization rate |
| dim_feedforward | 128 | Hidden layer size in FFN |

## Example Prediction

```plaintext
Input: France vs England

Processing...

Expected Goals (xG):
  France: 2.34
  England: 1.89

Top 3 Predicted Outcomes:
  1. Score: 2-1  |  Probability: 18.3%
  2. Score: 3-1  |  Probability: 14.7%
  3. Score: 2-2  |  Probability: 13.2%

Prediction Confidence: High (recent data, established teams)
```

## Technical Details

### Poisson Distribution for Football Scoring

Goal scoring in football approximates a Poisson distribution. Given predicted expected goals (λ), we calculate exact score probabilities:

$$P(X = k) = \frac{e^{-\lambda} \lambda^k}{k!}$$

For a match between Team A (λ_a) and Team B (λ_b):

$$P(\text{Score} = k_1-k_2) = P_A(k_1) \times P_B(k_2)$$

### Why Transformers for This Task?

- **Temporal Dependencies:** Self-attention captures which past matches influence predictions.
- **Variable Importance:** Handles important recent matches vs. older patterns automatically.
- **Parallel Processing:** Efficient computation on GPU with batch processing.
- **Long-Range Context:** No vanishing gradient problems like RNNs.
- **Transfer Learning:** Attention patterns interpretable and analyzable.

### ELO Rating System

The model implements dynamic team strength estimation:

$$\text{Expected Score} = \frac{1}{1 + 10^{\frac{\text{opponent rating} - \text{team rating}}{400}}}$$

$$\text{New Rating} = \text{Old Rating} + K \times (\text{Actual} - \text{Expected})$$

## Advanced Usage

### Training on Custom Data

```python
from world_cup_prediction import FootballDataset, WorldCupTransformer

dataset = FootballDataset('data/your_data.csv', seq_len=5, split_type='train')
model = WorldCupTransformer(num_features=5, d_model=64, nhead=4)
# Train your model...
```

### Batch Predictions

```python
from world_cup_prediction import predict_match_outcome

predictions = predict_match_outcome(model, home_seq, away_seq, "France", "England")
print(predictions)
```

## Model Performance and Limitations

### Strengths

- Captures historical team performance patterns accurately.
- Provides calibrated probability estimates.
- Handles sequences of varying importance effectively.
- Generalizes well to unseen matchups.

### Limitations

- Historical data bias (past ≠ future performance).
- Doesn't account for: injuries, tactical changes, player transfers, manager effects.
- Relies on match quality (friendly matches skewed).
- External events (COVID, breaks) affect prediction accuracy.

### Performance Metrics

The model is evaluated using:

- **Poisson Negative Log-Likelihood (NLL):** Primary loss function.
- **Exact Score Accuracy:** Percentage of correctly predicted exact scores.
- **Outcome Accuracy:** Win/Draw/Loss prediction accuracy.
- **Calibration:** How well predicted probabilities match actual frequencies.

## Future Enhancements

- [ ] **Player-Level Data:** Incorporate squad information and player ratings.
- [ ] **Injury Database:** Account for key player absences.
- [ ] **Home Advantage Factor:** Dynamic home/away adjustment.
- [ ] **Head-to-Head Analysis:** Special handling for frequent opponents.
- [ ] **Ensemble Methods:** Combine multiple model architectures.
- [ ] **Real-Time Updates:** Live team form tracking during tournaments.
- [ ] **Betting Odds Integration:** Compare predictions vs. market odds.
- [ ] **Weather and Venue:** Environmental factors (altitude, weather patterns).

## Dependencies

See `requirements.txt`:

- **torch** — Deep learning framework
- **pandas** — Data manipulation and analysis
- **numpy** — Numerical computing
- **scipy** — Scientific computing (Poisson distribution)
- **streamlit** — Web interface (for GUI)

## License

Specify your license here (MIT, Apache 2.0, etc.)

## Data Source

Historical match data sourced from:

- **Kaggle:** International Football Results Dataset
- **Club Data:** Downloaded via `scripts/download_club_data.py`

## Disclaimer

This model is for **educational and entertainment purposes only**.

Football match outcomes are influenced by countless factors beyond historical scores:

- Player injuries and roster changes
- Tactical adjustments and team morale
- External events and unpredictability
- Referee decisions and luck

Do not use for gambling or financial decisions without additional analysis and risk management.

## Contributing

Contributions welcome! Areas for improvement:

- Model architecture enhancements
- Data pipeline optimization
- Feature engineering
- Documentation and examples

## Author

Developed as a machine learning project combining deep learning with sports analytics.

Questions or Issues? Open an issue on GitHub.
