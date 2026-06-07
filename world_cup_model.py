import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import math
from scipy.stats import poisson

# ==========================================
# 1. Defining Transformer Architecture
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class WorldCupTransformer(nn.Module):
    def __init__(self, num_features, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(WorldCupTransformer, self).__init__()
        self.input_projection = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=128, dropout=dropout, 
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Network head - predicting expected goals (lambda) for each team
        self.fc_out = nn.Sequential(
            nn.Linear(d_model * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        self.softplus = nn.Softplus() # ensures lambda is always positive

    def forward(self, team_a_seq, team_b_seq):
        # Process team A sequence
        a_embedded = self.input_projection(team_a_seq)
        a_encoded = self.pos_encoder(a_embedded)
        a_out = self.transformer_encoder(a_encoded)
        a_final_state = a_out[:, -1, :] # extract the state of the last match in sequence

        # Process team B sequence
        b_embedded = self.input_projection(team_b_seq)
        b_encoded = self.pos_encoder(b_embedded)
        b_out = self.transformer_encoder(b_encoded)
        b_final_state = b_out[:, -1, :]

        # Concatenate states and predict
        combined_state = torch.cat((a_final_state, b_final_state), dim=1)
        raw_lambdas = self.fc_out(combined_state)
        predicted_lambdas = self.softplus(raw_lambdas)
        
        return predicted_lambdas

# ==========================================
# 2. Data Preparation (Sliding Window Dataset)
# ==========================================
class FootballDataset(Dataset):
    def __init__(self, csv_path='results.csv', seq_len=5):
        print("Loading data from Kaggle, filtering friendly matches and building time sequences...")
        
        # 1. Load the data
        df = pd.read_csv(csv_path)
        
        # Convert dates to proper format and sort chronologically
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # === Data Cleaning ===
        # Filter 1: Matches from 2015 onwards only
        df = df[df['date'].dt.year >= 2015]
        
        # Filter 2 (New!): Remove friendly matches
        # We only keep matches where the tournament column is not 'Friendly'
        df = df[df['tournament'] != 'Friendly']
        # =====================================

        # 2. Build history for each team
        team_histories = {}
        
        for index, row in df.iterrows():
            home = row['home_team']
            away = row['away_team']
            
            if home not in team_histories: team_histories[home] = []
            if away not in team_histories: team_histories[away] = []
                
            # [Opponent strength (currently 1.0), goals for, goals against]
            team_histories[home].append([1.0, row['home_score'], row['away_score']])
            team_histories[away].append([1.0, row['away_score'], row['home_score']])
            
        # 3. Create training examples (Sliding Windows)
        self.team_a_data = []
        self.team_b_data = []
        self.actual_scores = []
        
        team_current_idx = {team: 0 for team in team_histories.keys()}
        
        for index, row in df.iterrows():
            home = row['home_team']
            away = row['away_team']
            
            idx_home = team_current_idx[home]
            idx_away = team_current_idx[away]
            
            if idx_home >= seq_len and idx_away >= seq_len:
                home_seq = team_histories[home][idx_home - seq_len : idx_home]
                away_seq = team_histories[away][idx_away - seq_len : idx_away]
                
                self.team_a_data.append(home_seq)
                self.team_b_data.append(away_seq)
                self.actual_scores.append([row['home_score'], row['away_score']])
            
            team_current_idx[home] += 1
            team_current_idx[away] += 1

        # 4. Convert to PyTorch Tensors
        self.team_a_data = torch.tensor(self.team_a_data, dtype=torch.float32)
        self.team_b_data = torch.tensor(self.team_b_data, dtype=torch.float32)
        self.actual_scores = torch.tensor(self.actual_scores, dtype=torch.float32)
        
        print(f"Successfully created {len(self.actual_scores)} quality training examples (without friendly matches)!")

    def __len__(self):
        return len(self.actual_scores)

    def __getitem__(self, idx):
        return self.team_a_data[idx], self.team_b_data[idx], self.actual_scores[idx]
   
# ==========================================
# 3. Poisson Distribution Function (Generating Betting Form)
# ==========================================
def predict_match_outcome(model, team_a_history, team_b_history, team_a_name="Group A", team_b_name="Group B"):
    model.eval()
    with torch.no_grad():
        # Add batch dimension
        team_a_tensor = torch.tensor(team_a_history, dtype=torch.float32).unsqueeze(0)
        team_b_tensor = torch.tensor(team_b_history, dtype=torch.float32).unsqueeze(0)
        
        lambdas = model(team_a_tensor, team_b_tensor)
        lam_a = lambdas[0][0].item()
        lam_b = lambdas[0][1].item()
        
    print(f"\n[{team_a_name} vs {team_b_name}]")
    print(f"Calculated expected goals (xG): {team_a_name} ({lam_a:.2f}) - {team_b_name} ({lam_b:.2f})")
    
    probabilities = []
    # Calculate probabilities for all possible outcomes (0 to 5 goals)
    for goals_a in range(6):
        for goals_b in range(6):
            prob_a = poisson.pmf(goals_a, lam_a)
            prob_b = poisson.pmf(goals_b, lam_b)
            exact_prob = prob_a * prob_b * 100
            probabilities.append((goals_a, goals_b, exact_prob))
            
    probabilities.sort(key=lambda x: x[2], reverse=True)
    
    print("-" * 35)
    print("Most recommended outcomes:")
    for i in range(3):
        ga, gb, p = probabilities[i]
        print(f"Exact score: {ga}-{gb}  |  Probability: {p:.1f}%")
    print("-" * 35)

# ==========================================
# 4. Main Engine: Training and Execution
# ==========================================
def main():
    print("Initializing system...")
    
    # Hyperparameter settings
    SEQ_LEN = 5         # The model will look at the last 5 matches of each team
    NUM_FEATURES = 3    # Opponent strength, goals for, goals against
    BATCH_SIZE = 32
    EPOCHS = 20
    
    # Create model and data
    dataset = FootballDataset(csv_path='results.csv', seq_len=SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = WorldCupTransformer(num_features=NUM_FEATURES)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # PyTorch Poisson function - the heart of the model
    criterion = nn.PoissonNLLLoss(log_input=False)
    
    print("Starting Transformer training...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_a, batch_b, target_scores in dataloader:
            optimizer.zero_grad()
            
            # Predict lambdas
            pred_lambdas = model(batch_a, batch_b)
            
            # Calculate error against actual outcome (Poisson)
            loss = criterion(pred_lambdas, target_scores)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")
            
    print("Training completed successfully!\n")
    
    # ==========================================
    # 5. Live Prediction for World Cup
    # ==========================================
    print("Generating predictions for selected matches...")
    
    # Define virtual history (5 recent matches) for a strong team (like France)
    # Structure: [opponent strength, goals for, goals against]
    history_strong = [
        [0.8, 2, 0], [1.2, 3, 1], [0.9, 1, 0], [1.5, 2, 1], [0.7, 4, 0]
    ]
    
    # Define virtual history for a weak team (like Iraq)
    history_weak = [
        [0.5, 1, 1], [0.8, 0, 2], [1.1, 0, 3], [0.6, 1, 1], [0.9, 0, 2]
    ]
    
    # Define history for another strong team (like England) to predict a close match
    history_strong_2 = [
        [1.0, 1, 0], [1.3, 1, 1], [0.8, 2, 0], [1.4, 0, 0], [1.1, 2, 1]
    ]

    # Run predictions
    predict_match_outcome(model, history_strong, history_weak, "France", "Iraq")
    predict_match_outcome(model, history_strong, history_strong_2, "France", "England")

if __name__ == "__main__":
    main()