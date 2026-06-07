import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import math
from scipy.stats import poisson
import pickle

# ==========================================
# Helper Functions: Elo Rating System
# ==========================================
def calculate_expected_score(rating_a, rating_b):
    # חישוב תוחלת הניצחון לפי פערי הדירוג
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(rating_a, rating_b, score_a, score_b, k_factor=30):
    expected_a = calculate_expected_score(rating_a, rating_b)
    expected_b = calculate_expected_score(rating_b, rating_a)

    if score_a > score_b:
        actual_a, actual_b = 1.0, 0.0
    elif score_a < score_b:
        actual_a, actual_b = 0.0, 1.0
    else:
        actual_a, actual_b = 0.5, 0.5

    new_rating_a = rating_a + k_factor * (actual_a - expected_a)
    new_rating_b = rating_b + k_factor * (actual_b - expected_b)

    return new_rating_a, new_rating_b

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
    def __init__(self, csv_path='results.csv', seq_len=5, split_type='train', split_year=2022):
        print(f"Loading data for {split_type.upper()} split...")
        
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        df = df[df['date'].dt.year >= 2010]
        df = df[df['tournament'] != 'Friendly']

        # אתחול ה-Elo לכל נבחרת לציון בסיס של 1500
        all_teams = pd.concat([df['home_team'], df['away_team']]).unique()
        team_elos = {team: 1500.0 for team in all_teams}
        
        team_histories = {team: [] for team in all_teams}
        
        for index, row in df.iterrows():
            home = row['home_team']
            away = row['away_team']
            
            # בדיקה אם יש נתון על מגרש ניטרלי (בקובץ של Kaggle זה עמודת 'neutral')
            is_neutral = 1.0 if ('neutral' in df.columns and row['neutral']) else 0.0
            
            # שליפת הדירוג הנוכחי *לפני* המשחק
            current_home_elo = team_elos[home]
            current_away_elo = team_elos[away]
                
            # שמירת הנתונים להיסטוריה (נחלק את ה-Elo ב-1000 כדי שהרשת העצבית תעכל את זה טוב)
            # מבנה חדש (5 פיצ'רים): [Elo עצמי, Elo יריבה, זכות, חובה, האם ניטרלי]
            team_histories[home].append([current_home_elo/1000.0, current_away_elo/1000.0, row['home_score'], row['away_score'], is_neutral])
            team_histories[away].append([current_away_elo/1000.0, current_home_elo/1000.0, row['away_score'], row['home_score'], is_neutral])
            
            # עדכון ה-Elo *אחרי* המשחק (משפיע רק על המשחקים הבאים)
            new_home_elo, new_away_elo = update_elo(current_home_elo, current_away_elo, row['home_score'], row['away_score'])
            team_elos[home] = new_home_elo
            team_elos[away] = new_away_elo
            
        self.team_a_data = []
        self.team_b_data = []
        self.actual_scores = []
        
        team_current_idx = {team: 0 for team in all_teams}
        
        for index, row in df.iterrows():
            home = row['home_team']
            away = row['away_team']
            match_year = row['date'].year
            
            idx_home = team_current_idx[home]
            idx_away = team_current_idx[away]
            
            if idx_home >= seq_len and idx_away >= seq_len:
                is_train = match_year < split_year
                is_test = match_year >= split_year
                
                if (split_type == 'train' and is_train) or (split_type == 'test' and is_test):
                    home_seq = team_histories[home][idx_home - seq_len : idx_home]
                    away_seq = team_histories[away][idx_away - seq_len : idx_away]
                    
                    self.team_a_data.append(home_seq)
                    self.team_b_data.append(away_seq)
                    self.actual_scores.append([row['home_score'], row['away_score']])
            
            team_current_idx[home] += 1
            team_current_idx[away] += 1

        self.team_a_data = torch.tensor(self.team_a_data, dtype=torch.float32)
        self.team_b_data = torch.tensor(self.team_b_data, dtype=torch.float32)
        self.actual_scores = torch.tensor(self.actual_scores, dtype=torch.float32)
        
        print(f"Successfully created {len(self.actual_scores)} examples for {split_type.upper()}!")

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
    NUM_FEATURES = 5    # Opponent strength, goals for, goals against
    BATCH_SIZE = 32
    EPOCHS = 15         # Slightly reduced to prevent overfitting with train/test separation
    
    # Create distinct Train and Test datasets
    train_dataset = FootballDataset(csv_path='results.csv', seq_len=SEQ_LEN, split_type='train', split_year=2022)
    test_dataset = FootballDataset(csv_path='results.csv', seq_len=SEQ_LEN, split_type='test', split_year=2022)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) # Important: Don't shuffle test data
    
    model = WorldCupTransformer(num_features=NUM_FEATURES)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # PyTorch Poisson function - the heart of the model
    criterion = nn.PoissonNLLLoss(log_input=False)
    
    print("Starting Transformer training...")
    for epoch in range(EPOCHS):
        
        # === Training Phase ===
        model.train()
        total_train_loss = 0
        for batch_a, batch_b, target_scores in train_loader:
            optimizer.zero_grad()
            
            pred_lambdas = model(batch_a, batch_b)
            pred_lambdas = torch.clamp(pred_lambdas, min=1e-6, max=1e3)

            loss = criterion(pred_lambdas, target_scores)
            if torch.isnan(loss):
                continue
                        
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # === Testing Phase ===
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for batch_a, batch_b, target_scores in test_loader:
                pred_lambdas = model(batch_a, batch_b)
                pred_lambdas = torch.clamp(pred_lambdas, min=1e-6, max=1e3) # Keep clamping for evaluation too
                
                loss = criterion(pred_lambdas, target_scores)
                if not torch.isnan(loss):
                    total_test_loss += loss.item()
                    
        avg_test_loss = total_test_loss / len(test_loader)
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")
            
    print("\nTraining and Evaluation completed successfully!\n")
    
    # 1. שמירת המוח של הרשת (המשקולות)
    torch.save(model.state_dict(), 'world_cup_model_weights.pth')
    print("Model weights saved to 'world_cup_model_weights.pth'")

    # 2. שמירת ה"זיכרון" (הדירוגים העדכניים והיסטוריית המשחקים) מתוך קבוצת המבחן/אימון
    tournament_state = {
        'elos': test_dataset.team_elos, 
        'histories': test_dataset.team_histories
    }
    with open('tournament_state.pkl', 'wb') as f:
        pickle.dump(tournament_state, f)
    print("Tournament state saved to 'tournament_state.pkl'")
    # ==========================================
    # 5. Live Prediction for World Cup (Updated for 5 Features)
    # ==========================================
    print("Generating predictions for selected matches...")
    
    # היסטוריה לנבחרת עילית (למשל צרפת - Elo משוער 2000 -> 2.0)
    # מבנה: [Elo עצמי, Elo יריבה, זכות, חובה, האם ניטרלי]
    history_strong = [
        [2.0, 1.5, 2, 0, 1.0], # ניצחון על קבוצה ממוצעת במגרש ניטרלי
        [2.0, 1.8, 3, 1, 1.0], # ניצחון על קבוצה טובה במגרש ניטרלי
        [2.0, 1.4, 1, 0, 0.0], # ניצחון דחוק על קבוצה חלשה (בבית)
        [2.0, 1.9, 2, 1, 1.0], # ניצחון על נבחרת צמרת
        [2.0, 1.3, 4, 0, 1.0]  # תבוסה לקבוצה חלשה מאוד
    ]
    
    # היסטוריה לנבחרת חלשה (למשל עיראק - Elo משוער 1400 -> 1.4)
    history_weak = [
        [1.4, 1.5, 1, 1, 0.0], # תיקו מול קבוצה ממוצעת בבית
        [1.4, 1.7, 0, 2, 1.0], # הפסד לקבוצה סבירה במגרש ניטרלי
        [1.4, 1.9, 0, 3, 1.0], # תבוסה לנבחרת צמרת
        [1.4, 1.4, 1, 1, 1.0], # תיקו מול נבחרת שקולה
        [1.4, 1.6, 0, 2, 0.0]  # הפסד לקבוצה קצת יותר טובה בבית
    ]
    
    # היסטוריה לנבחרת עילית נוספת (למשל אנגליה - Elo משוער 1950 -> 1.95)
    history_strong_2 = [
        [1.95, 1.6, 1, 0, 1.0], # ניצחון מול קבוצה סבירה
        [1.95, 1.8, 1, 1, 1.0], # תיקו מול קבוצה טובה
        [1.95, 1.5, 2, 0, 0.0], # ניצחון בבית מול קבוצה ממוצעת
        [1.95, 2.0, 0, 0, 1.0], # תיקו מאופס מול נבחרת עילית
        [1.95, 1.7, 2, 1, 1.0]  # ניצחון מול קבוצה בינונית פלוס
    ]

    # הרצת התחזיות: המודל עכשיו מקבל את מלוא ההקשר!
    predict_match_outcome(model, history_strong, history_weak, "France", "Iraq")
    predict_match_outcome(model, history_strong, history_strong_2, "France", "England")
if __name__ == "__main__":
    main()