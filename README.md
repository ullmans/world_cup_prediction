# World Cup Prediction

A machine learning model that predicts football match outcomes using a Transformer-based architecture. The model learns from historical match data and generates probabilistic predictions for different scorelines using Poisson distribution.

## Overview

This project implements a **Transformer neural network** trained on international football match results to predict match outcomes. The model analyzes team performance histories and estimates the expected number of goals (λ) for each team, then uses Poisson statistics to generate the most likely score predictions.

### Key Features

- **Transformer Architecture**: Uses multi-head self-attention to capture temporal patterns in team performance
- **Positional Encoding**: Encodes match history chronologically to preserve temporal context
- **Poisson-Based Predictions**: Converts predicted goal expectations into probability distributions for all possible scorelines
- **Data Cleaning**: Automatically filters competitive matches (excludes friendlies) for more reliable predictions
- **Sliding Window Dataset**: Uses sequences of the last N matches for each team as training examples

## Project Structure

```
├── world_cup_model.py      # Main model implementation and training script
├── results.csv             # Historical match data (from Kaggle)
├── results.csv             # Model predictions output
├── req.txt                 # Python dependencies
├── dockerfile              # Docker configuration for containerization
└── README.md               # This file
```

## How It Works

### 1. **Data Preparation**
- Loads international football match results from CSV
- Filters matches from 2015 onwards and removes friendly matches
- Builds chronological match histories for each team
- Creates training examples using sliding windows (default: last 5 matches)

### 2. **Model Architecture**

The `WorldCupTransformer` model has three main components:

```
Input (team match sequences)
    ↓
Input Projection (embed to d_model dimensions)
    ↓
Positional Encoding (add temporal information)
    ↓
Transformer Encoder (self-attention layers)
    ↓
Fully Connected Head (predict goal expectations λ)
    ↓
Output (λ_team_a, λ_team_b)
```

### 3. **Training**
- Uses **Poisson NLL Loss** to match predicted goal distributions with actual match results
- Trains on batches of match data using Adam optimizer
- Learns to estimate λ (expected goals) for each team

### 4. **Prediction**
- Takes team match histories as input
- Outputs expected goals (λ) for each team
- Uses Poisson probability mass function to calculate probabilities for all scorelines (0-5 goals each)
- Returns the top 3 most likely exact score predictions with probabilities

## Installation

### Requirements
- Python 3.7+
- PyTorch
- Pandas
- NumPy
- SciPy

### Setup

1. Install dependencies:
```bash
pip install -r req.txt
```

2. Prepare your data:
   - Ensure `results.csv` is in the project root with columns: `date`, `home_team`, `away_team`, `home_score`, `away_score`, `tournament`

## Usage

### Running the Model

```bash
python world_cup_model.py
```

This will:
1. Load and preprocess the match data
2. Train the Transformer model for 20 epochs
3. Generate predictions for sample matches (France vs Iraq, France vs England)
4. Display top 3 likely scorelines with probabilities

### Example Output

```
[France vs Iraq]
Expected goals (xG): France (2.45) - Iraq (0.82)
-----------------------------------
Top 3 predicted scores:
Score: 3-0  |  Probability: 18.5%
Score: 2-0  |  Probability: 15.2%
Score: 2-1  |  Probability: 12.8%
-----------------------------------
```

## Model Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SEQ_LEN` | 5 | Number of past matches to consider per team |
| `NUM_FEATURES` | 3 | Features: opponent strength, goals for, goals against |
| `BATCH_SIZE` | 32 | Training batch size |
| `EPOCHS` | 20 | Number of training epochs |
| `d_model` | 64 | Transformer embedding dimension |
| `nhead` | 4 | Number of attention heads |
| `num_layers` | 2 | Number of Transformer encoder layers |

## Data Format

The input CSV should contain:
| Column | Type | Description |
|--------|------|-------------|
| date | datetime | Match date |
| home_team | string | Home team name |
| away_team | string | Away team name |
| home_score | int | Goals scored by home team |
| away_score | int | Goals scored by away team |
| tournament | string | Tournament name (non-Friendly matches used) |

## Docker

To run the model in a Docker container:

```bash
docker build -t world-cup-prediction .
docker run world-cup-prediction
```

## Model Performance

The model learns patterns from competitive international matches to make predictions. Performance depends on:
- Quality and recency of training data
- Team consistency over time
- External factors (injuries, tactics) not captured in historical scores

## Technical Details

### Poisson Distribution in Sports
Goal-scoring in football approximately follows a Poisson distribution. By predicting λ (expected goals), we can calculate probabilities for any scoreline:

```
P(goals = k) = (e^(-λ) * λ^k) / k!
```

### Transformer Advantages for This Task
- Captures temporal dependencies in team form
- Attention mechanism identifies which past matches are most relevant
- Handles variable sequence importance automatically
- Better than RNNs for capturing long-range patterns

## Future Improvements

- Include team rankings/ELO ratings as additional features
- Add home/away advantage factor
- Incorporate player availability and injuries
- Use ensemble methods with multiple models
- Add real-time feature updates (current season form)

## License

[Specify your license here]

## Data Source

Match data sourced from [Kaggle International Football Results Dataset](https://www.kaggle.com/martj42/international-football-results)

---

**Note**: This model is for educational and entertainment purposes. Actual match predictions involve many factors beyond historical scores.