import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import math
from scipy.stats import poisson

# ==========================================
# 1. הגדרת ארכיטקטורת ה-Transformer
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
        
        # ראש הרשת - חיזוי תוחלת השערים (למבדא) לכל קבוצה
        self.fc_out = nn.Sequential(
            nn.Linear(d_model * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        self.softplus = nn.Softplus() # מבטיח שהלמבדא תמיד תהיה חיובית

    def forward(self, team_a_seq, team_b_seq):
        # עיבוד רצף קבוצה א'
        a_embedded = self.input_projection(team_a_seq)
        a_encoded = self.pos_encoder(a_embedded)
        a_out = self.transformer_encoder(a_encoded)
        a_final_state = a_out[:, -1, :] # לקיחת המצב של המשחק האחרון ברצף

        # עיבוד רצף קבוצה ב'
        b_embedded = self.input_projection(team_b_seq)
        b_encoded = self.pos_encoder(b_embedded)
        b_out = self.transformer_encoder(b_encoded)
        b_final_state = b_out[:, -1, :]

        # שרשור המצבים וחיזוי
        combined_state = torch.cat((a_final_state, b_final_state), dim=1)
        raw_lambdas = self.fc_out(combined_state)
        predicted_lambdas = self.softplus(raw_lambdas)
        
        return predicted_lambdas

# ==========================================
# 2. הכנת הנתונים (Sliding Window Dataset)
# ==========================================
class FootballDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=5, num_features=3):
        """
        כאן אנחנו מייצרים נתוני דמה שמדמים היסטוריה של משחקים.
        במציאות, הפונקציה הזו תקבל DataFrame שנוצר מקובץ ה-CSV שלך.
        """
        self.seq_len = seq_len
        
        # רצף המשחקים האחרונים של קבוצה א' [כוח יריבה, שערי זכות, שערי חובה]
        self.team_a_data = torch.rand(num_samples, seq_len, num_features) * 3 
        # רצף המשחקים האחרונים של קבוצה ב'
        self.team_b_data = torch.rand(num_samples, seq_len, num_features) * 3
        
        # התוצאה האמיתית שהתרחשה במשחק שביניהן (המטרה לחיזוי)
        # טור 1: שערים קבוצה א', טור 2: שערים קבוצה ב'
        self.actual_scores = torch.poisson(torch.rand(num_samples, 2) * 2)

    def __len__(self):
        return len(self.actual_scores)

    def __getitem__(self, idx):
        return self.team_a_data[idx], self.team_b_data[idx], self.actual_scores[idx]

# ==========================================
# 3. פונקציית תרגום פואסון (הפקת טופס ההימורים)
# ==========================================
def predict_match_outcome(model, team_a_history, team_b_history, team_a_name="Group A", team_b_name="Group B"):
    model.eval()
    with torch.no_grad():
        # הוספת ממד ה-Batch
        team_a_tensor = torch.tensor(team_a_history, dtype=torch.float32).unsqueeze(0)
        team_b_tensor = torch.tensor(team_b_history, dtype=torch.float32).unsqueeze(0)
        
        lambdas = model(team_a_tensor, team_b_tensor)
        lam_a = lambdas[0][0].item()
        lam_b = lambdas[0][1].item()
        
    print(f"\n[{team_a_name} מול {team_b_name}]")
    print(f"תוחלת שערים מחושבת (xG): {team_a_name} ({lam_a:.2f}) - {team_b_name} ({lam_b:.2f})")
    
    probabilities = []
    # חישוב הסתברויות לכל תוצאה אפשרית (מ-0 עד 5 שערים)
    for goals_a in range(6):
        for goals_b in range(6):
            prob_a = poisson.pmf(goals_a, lam_a)
            prob_b = poisson.pmf(goals_b, lam_b)
            exact_prob = prob_a * prob_b * 100
            probabilities.append((goals_a, goals_b, exact_prob))
            
    probabilities.sort(key=lambda x: x[2], reverse=True)
    
    print("-" * 35)
    print("התוצאות המומלצות ביותר לטופס:")
    for i in range(3):
        ga, gb, p = probabilities[i]
        print(f"תוצאה מדויקת: {ga}-{gb}  |  סיכוי: {p:.1f}%")
    print("-" * 35)

# ==========================================
# 4. המנוע הראשי: אימון והרצה
# ==========================================
def main():
    print("מאתחל מערכת...")
    
    # הגדרות היפר-פרמטרים
    SEQ_LEN = 5         # המודל יסתכל על 5 משחקים אחרונים של כל נבחרת
    NUM_FEATURES = 3    # עוצמת יריבה, שערי זכות, שערי חובה
    BATCH_SIZE = 32
    EPOCHS = 20
    
    # יצירת המודל והנתונים
    dataset = FootballDataset(num_samples=2000, seq_len=SEQ_LEN, num_features=NUM_FEATURES)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = WorldCupTransformer(num_features=NUM_FEATURES)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # פונקציית פואסון של PyTorch - הלב של המודל
    criterion = nn.PoissonNLLLoss(log_input=False)
    
    print("מתחיל אימון ה-Transformer...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_a, batch_b, target_scores in dataloader:
            optimizer.zero_grad()
            
            # חיזוי הלמבדאות
            pred_lambdas = model(batch_a, batch_b)
            
            # חישוב השגיאה לעומת התוצאה בפועל (פואסון)
            loss = criterion(pred_lambdas, target_scores)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")
            
    print("האימון הושלם בהצלחה!\n")
    
    # ==========================================
    # 5. תחזית חיה לקראת המונדיאל
    # ==========================================
    print("מייצר תחזית למשחקים נבחרים...")
    
    # נגדיר היסטוריה וירטואלית (5 משחקים אחרונים) לנבחרת חזקה (כמו צרפת)
    # מבנה: [כוח יריבה, כבש, ספג]
    history_strong = [
        [0.8, 2, 0], [1.2, 3, 1], [0.9, 1, 0], [1.5, 2, 1], [0.7, 4, 0]
    ]
    
    # נגדיר היסטוריה וירטואלית לנבחרת חלשה (כמו עיראק)
    history_weak = [
        [0.5, 1, 1], [0.8, 0, 2], [1.1, 0, 3], [0.6, 1, 1], [0.9, 0, 2]
    ]
    
    # נגדיר היסטוריה לנבחרת חזקה נוספת (כמו אנגליה) לחזות משחק צמוד
    history_strong_2 = [
        [1.0, 1, 0], [1.3, 1, 1], [0.8, 2, 0], [1.4, 0, 0], [1.1, 2, 1]
    ]

    # הרצת התחזיות
    predict_match_outcome(model, history_strong, history_weak, "צרפת", "עיראק")
    predict_match_outcome(model, history_strong, history_strong_2, "צרפת", "אנגליה")

if __name__ == "__main__":
    main()