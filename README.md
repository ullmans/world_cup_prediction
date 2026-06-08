🏆 World Cup Prediction AI
AI-powered football match predictor using Transformer neural networks and Poisson distribution analysis.

A machine learning system that predicts international football match outcomes with probabilistic scoreline predictions. Combines historical match data, ELO ratings, and advanced deep learning to estimate expected goals and generate accurate match forecasts.

📋 Overview
This project implements a sophisticated Transformer-based neural network trained on international football match results to predict match outcomes with quantified probabilities. The model analyzes team performance histories, accounts for ELO ratings, and uses Poisson statistics to generate realistic score predictions.

🎯 Key Features
Advanced Transformer Architecture: Multi-head self-attention mechanism captures temporal patterns in team performance
Positional Encoding: Chronological encoding preserves temporal context of match sequences
Poisson-Based Predictions: Converts predicted expected goals (λ) into probability distributions for all possible scorelines
ELO Rating System: Implements dynamic team strength estimation for more accurate predictions
Dual Dataset Support: Trains on both national team and club-level match data
Interactive GUI: Streamlit-based interface for easy match predictions
Live Prediction Engine: Real-time prediction capability with Hebrew/English team name support
Model Persistence: Pre-trained model weights for immediate predictions without retraining
📁 Project Structure
├── world_cup_model.py           # Core Transformer model & training pipeline
├── live_predictor.py             # Match prediction engine with Poisson calculations
├── gui.py                        # Streamlit web interface for predictions
├── download_club_data.py         # Club match data downloader
├── update_memory.py              # Memory update utilities
│
├── national_results.csv          # International match historical data
├── club_results.csv              # Club match historical data
│
├── world_cup_final_model.pth     # Pre-trained national team model weights
├── world_cup_model_weights.pth   # Alternative model weights checkpoint
├── pretrained_clubs.pth          # Pre-trained club-level model weights
├── tournament_state.pkl          # Serialized tournament state for live predictions
│
├── req.txt                       # Python dependencies
├── dockerfile                    # Docker containerization config
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
🧠 Model Architecture
Transformer-Based Design
Team A Match History              Team B Match History
        ↓                                  ↓
 Input Projection            Input Projection
        ↓                                  ↓
Positional Encoding         Positional Encoding
        ↓                                  ↓
Transformer Encoder         Transformer Encoder
        ↓                                  ↓
Extract Final State         Extract Final State
        ↓                                  ↓
    Concatenate & Merge
        ↓
  Fully Connected Head
        ↓
Output: λ_team_a, λ_team_b
        ↓
Poisson PMF → Score Probabilities
Key Components
Component	Purpose
Input Projection	Maps raw features to d_model dimensions
Positional Encoding	Adds temporal information to sequences
Transformer Encoder	Multi-head attention (4 heads) across match sequences
Output Head	Dense layers predicting expected goals (λ)
Softplus Activation	Ensures λ remains non-negative
🚀 Quick Start
Installation
Requirements:

Python 3.8+
PyTorch
Pandas, NumPy, SciPy
Streamlit (for GUI)
Setup:

pip install -r req.txt
Usage
1. Train the Model
python world_cup_model.py
Trains the Transformer on historical match data, saves weights to world_cup_final_model.pth

2. Web Interface (Recommended)
streamlit run gui.py
Opens an interactive web app for match predictions with bilingual team name support (Hebrew/English)

3. Live Predictions (Programmatic)
from live_predictor import main_live_predictor

predictions = main_live_predictor("France", "England")
# Returns top 3 scoreline predictions with probabilities
4. Docker Deployment
docker build -t world-cup-prediction .
docker run world-cup-prediction
💾 Data Format
Input CSV Structure
Column	Type	Description	Example
date	YYYY-MM-DD	Match date	2023-11-14
home_team	string	Home team name	France
away_team	string	Away team name	England
home_score	int	Goals scored by home team	2
away_score	int	Goals scored by away team	1
tournament	string	Match type (exclude "Friendly")	UEFA Euro
Data Processing Pipeline
Load: Parse CSV with temporal sorting
Filter: Remove friendly matches, keep competitive tournaments only
Validation: Ensure complete match records with valid scores
Sliding Window: Create sequences of last N matches per team (default: 5)
Normalization: Scale features for neural network input
🎮 Model Parameters
Hyperparameter	Value	Description
SEQ_LEN	5	Past matches per team in sequence
NUM_FEATURES	3	Input features (opponent strength, GF, GA)
BATCH_SIZE	32	Training batch size
EPOCHS	20	Training iterations
d_model	64	Transformer embedding dimension
nhead	4	Number of attention heads
num_layers	2	Transformer encoder layers
dropout	0.1	Regularization rate
dim_feedforward	128	Hidden layer size in FFN
📊 Example Prediction
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
📐 Technical Details
Poisson Distribution for Football Scoring
Goal scoring in football approximates a Poisson distribution. Given predicted expected goals (λ), we calculate exact score probabilities:

P(X = k) = (e^(-λ) × λ^k) / k!
For a match between Team A (λ_a) and Team B (λ_b):

P(Score = k₁-k₂) = P_A(k₁) × P_B(k₂)
Why Transformers for This Task?
✅ Temporal Dependencies: Self-attention captures which past matches influence predictions
✅ Variable Importance: Handles important recent matches vs. older patterns automatically
✅ Parallel Processing: Efficient computation on GPU with batch processing
✅ Long-Range Context: No vanishing gradient problems like RNNs
✅ Transfer Learning: Attention patterns interpretable and analyzable

ELO Rating System
The model implements dynamic team strength estimation:

Expected Score = 1 / (1 + 10^((opponent_rating - team_rating) / 400))

New Rating = Old Rating + K × (Actual - Expected)
🔧 Advanced Usage
Training on Custom Data
from world_cup_model import FootballDataset, WorldCupTransformer

dataset = FootballDataset('your_data.csv', seq_len=5, split_type='train')
model = WorldCupTransformer(num_features=3, d_model=64, nhead=4)
# Train your model...
Batch Predictions
from live_predictor import predict_match_outcome

matches = [
    (["France", "England", "Germany"], ["Spain", "Italy", "Netherlands"]),
]

for home_teams, away_teams in matches:
    predictions = predict_match_outcome(model, home_seq, away_seq, home_teams[0], away_teams[0])
    print(predictions)
📈 Model Performance & Limitations
Strengths
Captures historical team performance patterns accurately
Provides calibrated probability estimates
Handles sequences of varying importance effectively
Generalizes well to unseen matchups
Limitations
Historical data bias (past ≠ future performance)
Doesn't account for: injuries, tactical changes, player transfers, manager effects
Relies on match quality (friendly matches skewed)
External events (COVID, breaks) affect prediction accuracy
Performance Metrics
The model is evaluated using:

Poisson Negative Log-Likelihood (NLL): Primary loss function
Exact Score Accuracy: Percentage of correctly predicted exact scores
Outcome Accuracy: Win/Draw/Loss prediction accuracy
Calibration: How well predicted probabilities match actual frequencies
🎓 Future Enhancements
[ ] Player-Level Data: Incorporate squad information and player ratings
[ ] Injury Database: Account for key player absences
[ ] Home Advantage Factor: Dynamic home/away adjustment
[ ] Head-to-Head Analysis: Special handling for frequent opponents
[ ] Ensemble Methods: Combine multiple model architectures
[ ] Real-Time Updates: Live team form tracking during tournaments
[ ] Betting Odds Integration: Compare predictions vs. market odds
[ ] Weather & Venue: Environmental factors (altitude, weather patterns)
📦 Dependencies
See req.txt:

torch - Deep learning framework
pandas - Data manipulation and analysis
numpy - Numerical computing
scipy - Scientific computing (Poisson distribution)
streamlit - Web interface (optional, for GUI)
📄 License
Specify your license here (MIT, Apache 2.0, etc.)

📚 Data Source
Historical match data sourced from:

Kaggle: International Football Results Dataset
Club Data: Downloaded via internal pipeline
⚠️ Disclaimer
This model is for educational and entertainment purposes only.

Football match outcomes are influenced by countless factors beyond historical scores:

Player injuries and roster changes
Tactical adjustments and team morale
External events and unpredictability
Referee decisions and luck
Do not use for gambling or financial decisions without additional analysis and risk management.

🤝 Contributing
Contributions welcome! Areas for improvement:

Model architecture enhancements
Data pipeline optimization
Feature engineering
Documentation and examples
👤 Author
Developed as a machine learning project combining deep learning with sports analytics.

Questions or Issues? Open an issue on GitHub.