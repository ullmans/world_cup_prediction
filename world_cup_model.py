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
# פונקציות עזר (מועתקות מקוד האימון)
# ==========================================
def calculate_expected_score(rating_a, rating_b):
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
    def __init__(self, csv_path='results.csv', seq_len=5, split_type='train', test_size=0.1):
        print(f"Loading data for {split_type.upper()} split (Test size: {test_size*100}%)...")
        
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, format='mixed')
        df = df.sort_values('date')
        
        df = df[df['date'].dt.year >= 2010]
        df = df[df['tournament'] != 'Friendly']
        df = df.dropna(subset=['home_score', 'away_score']) # סינון משחקים ריקים

        all_teams = pd.concat([df['home_team'], df['away_team']]).unique()
        team_elos = {team: 1500.0 for team in all_teams}
        team_histories = {team: [] for team in all_teams}
        
        all_a_data = []
        all_b_data = []
        all_actual_scores = []
        
        team_current_idx = {team: 0 for team in all_teams}
        
        # מעבר על כל הנתונים ויצירת הרצפים הכרונולוגיים
        for index, row in df.iterrows():
            home = row['home_team']
            away = row['away_team']
            
            # שליפת הדירוג וחישוב הפיצ'רים החכמים
            current_home_elo = team_elos[home]
            current_away_elo = team_elos[away]
            elo_diff = (current_home_elo - current_away_elo) / 1000.0
            goal_diff = row['home_score'] - row['away_score']
            is_neutral = True if ('neutral' in df.columns and row['neutral']) else False
            home_adv = 0.0 if is_neutral else 1.0
            away_adv = 0.0 if is_neutral else -1.0
            
            idx_home = team_current_idx[home]
            idx_away = team_current_idx[away]
            
            # אם יש מספיק היסטוריה, מוסיפים לרשימה הכללית
            if idx_home >= seq_len and idx_away >= seq_len:
                home_seq = team_histories[home][idx_home - seq_len : idx_home]
                away_seq = team_histories[away][idx_away - seq_len : idx_away]
                
                all_a_data.append(home_seq)
                all_b_data.append(away_seq)
                all_actual_scores.append([row['home_score'], row['away_score']])
            
            # הוספת המשחק הנוכחי לזיכרון הקבוצות
            team_histories[home].append([elo_diff, row['home_score'], row['away_score'], goal_diff, home_adv])
            team_histories[away].append([-elo_diff, row['away_score'], row['home_score'], -goal_diff, away_adv])
            
            team_current_idx[home] += 1
            team_current_idx[away] += 1
            
            # עדכון דירוג ה-Elo לאחר המשחק
            new_home_elo, new_away_elo = update_elo(current_home_elo, current_away_elo, row['home_score'], row['away_score'])
            team_elos[home] = new_home_elo
            team_elos[away] = new_away_elo
            
        # === חיתוך אחוזים כרונולוגי ===
        total_samples = len(all_actual_scores)
        split_idx = int(total_samples * (1 - test_size)) # מציאת נקודת ה-90%
        
        if split_type == 'train':
            final_a = all_a_data[:split_idx]
            final_b = all_b_data[:split_idx]
            final_scores = all_actual_scores[:split_idx]
        else: # test
            final_a = all_a_data[split_idx:]
            final_b = all_b_data[split_idx:]
            final_scores = all_actual_scores[split_idx:]
            
        # המרה לטנזורים
        self.team_a_data = torch.tensor(final_a, dtype=torch.float32)
        self.team_b_data = torch.tensor(final_b, dtype=torch.float32)
        self.actual_scores = torch.tensor(final_scores, dtype=torch.float32)
        
        # שמירת הזיכרון
        self.team_elos = team_elos
        self.team_histories = team_histories
        
        print(f"Successfully created {len(self.actual_scores)} examples for {split_type.upper()}!")

    def __len__(self):
        return len(self.actual_scores)

    def __getitem__(self, idx):
        return self.team_a_data[idx], self.team_b_data[idx], self.actual_scores[idx]
    
# ==========================================
# 4. Main Engine: Training and Execution
# ==========================================
# ==========================================
# מנוע אימון גנרי (לשימוש כפול: Pre-training ו-Fine-tuning)
# ==========================================
def train_and_evaluate(model, train_loader, test_loader, epochs, lr, weight_decay, save_path):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.PoissonNLLLoss(log_input=False)
    
    for epoch in range(epochs):
        # === שלב האימון ===
        model.train()
        total_train_loss = 0
        for batch_a, batch_b, target_scores in train_loader:
            optimizer.zero_grad()
            
            pred_lambdas = model(batch_a, batch_b)
            pred_lambdas = torch.clamp(pred_lambdas, min=1e-4, max=50.0)

            loss = criterion(pred_lambdas, target_scores)
            if torch.isnan(loss):
                continue
                        
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # חגורת בטיחות
            optimizer.step()
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # === שלב המבחן ===
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for batch_a, batch_b, target_scores in test_loader:
                pred_lambdas = model(batch_a, batch_b)
                pred_lambdas = torch.clamp(pred_lambdas, min=1e-4, max=50.0)
                
                loss = criterion(pred_lambdas, target_scores)
                if not torch.isnan(loss):
                    total_test_loss += loss.item()
                    
        avg_test_loss = total_test_loss / len(test_loader)
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")
    
    # שמירת המשקולות בסוף תהליך האימון הזה
    torch.save(model.state_dict(), save_path)
    print(f"--> נשמר קובץ משקולות: {save_path}\n")
    return model


# ==========================================
# Main Engine: Transfer Learning Pipeline
# ==========================================
def main():
    print("מתחיל מערכת Transfer Learning...")
    
    SEQ_LEN = 5
    NUM_FEATURES = 5
    BATCH_SIZE = 32
    
    # אתחול הארכיטקטורה
    model = WorldCupTransformer(num_features=NUM_FEATURES)
    
    # =========================================================
    # Phase 1: Pre-training (למידת כדורגל כללית על מועדונים)
    # =========================================================
    print("==================================================")
    print("PHASE 1: Pre-training on Club Data")
    print("==================================================")
    
    # נטען את נתוני המועדונים ונחלק 90/10
    club_train = FootballDataset(csv_path='club_results.csv', seq_len=SEQ_LEN, split_type='train', test_size=0.1)
    club_test  = FootballDataset(csv_path='club_results.csv', seq_len=SEQ_LEN, split_type='test', test_size=0.1)
    
    club_train_loader = DataLoader(club_train, batch_size=BATCH_SIZE, shuffle=True)
    club_test_loader  = DataLoader(club_test, batch_size=BATCH_SIZE, shuffle=False)
    
    # נאמן את הרשת מאפס, עם קצב למידה סטנדרטי למשך 15 Epochs
    model = train_and_evaluate(
        model=model, 
        train_loader=club_train_loader, 
        test_loader=club_test_loader, 
        epochs=15, 
        lr=0.0001, 
        weight_decay=1e-4, 
        save_path='pretrained_clubs.pth'
    )
    
    # =========================================================
    # Phase 2: Fine-Tuning (התאמה עדינה לכדורגל נבחרות)
    # =========================================================
    print("==================================================")
    print("PHASE 2: Fine-Tuning on National Teams Data")
    print("==================================================")
    
    # נטען את נתוני הנבחרות (זה גם ישמור את ה-pkl המעודכן למונדיאל!)
    nat_train = FootballDataset(csv_path='national_results.csv', seq_len=SEQ_LEN, split_type='train', test_size=0.1)
    nat_test  = FootballDataset(csv_path='national_results.csv', seq_len=SEQ_LEN, split_type='test', test_size=0.1)
    
    nat_train_loader = DataLoader(nat_train, batch_size=BATCH_SIZE, shuffle=True)
    nat_test_loader  = DataLoader(nat_test, batch_size=BATCH_SIZE, shuffle=False)
    
    # טריק קריטי: אנחנו טוענים את ה"מוח" שהתאמן על המועדונים
    model.load_state_dict(torch.load('pretrained_clubs.pth'))
    
    # נאמן שוב, אבל הפעם: קצב למידה איטי בהרבה (כדי לא להרוס מה שנלמד) ורק 4 Epochs!
    model = train_and_evaluate(
        model=model, 
        train_loader=nat_train_loader, 
        test_loader=nat_test_loader, 
        epochs=4, 
        lr=0.00001,  # שמנו פה עוד אפס - אנחנו רק "מעדנים" את המשקולות
        weight_decay=1e-4, 
        save_path='world_cup_final_model.pth'
    )
    
    # שמירת הזיכרון (היסטוריות ו-Elo) מהנבחרות עבור סקריפט החיזוי החי
    import pickle
    tournament_state = {
        'elos': nat_test.team_elos, 
        'histories': nat_test.team_histories
    }
    with open('tournament_state.pkl', 'wb') as f:
        pickle.dump(tournament_state, f)
    print("Tournament state saved to 'tournament_state.pkl' - המערכת מוכנה למונדיאל!")

if __name__ == "__main__":
    main()