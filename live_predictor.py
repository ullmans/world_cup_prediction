import torch
import pickle
from scipy.stats import poisson

from world_cup_model import WorldCupTransformer, PositionalEncoding, predict_match_outcome

def main_live_predictor():
    print("Loading World Cup AI Assistant...")
    
    # 1. הקמת שלד המודל וטעינת ה"מוח" ששמרנו
    NUM_FEATURES = 5
    model = WorldCupTransformer(num_features=NUM_FEATURES)
    
    # טעינת המשקולות 
    model.load_state_dict(torch.load('world_cup_model_weights.pth'))
    model.eval() # פקודה קריטית! נועלת את המודל למצב "תחזית" ומכבה Dropout
    
    # 2. טעינת ה"זיכרון" (Elo והיסטוריה)
    with open('tournament_state.pkl', 'rb') as f:
        state = pickle.load(f)
        
    team_elos = state['elos']
    team_histories = state['histories']
    
    print("System Ready!\n")
    print("-" * 30)
    
    # 3. כאן אתה מזין כל יום את הקבוצות שאתה רוצה לבדוק
    # שנה את השמות האלו בהתאם למשחקי היום (חובה שיהיו בדיוק באיות מה-CSV)
    team_a_name = "Argentina"
    team_b_name = "Mexico"
    
    # נוודא שהקבוצות אכן קיימות במסד הנתונים
    if team_a_name not in team_histories or team_b_name not in team_histories:
        print("שגיאה: אחת הקבוצות לא נמצאה במסד הנתונים. ודא איות באנגלית.")
        return
        
    # שולפים את 5 המשחקים האחרונים של כל קבוצה מהזיכרון
    history_a = team_histories[team_a_name][-5:]
    history_b = team_histories[team_b_name][-5:]
    
    # מדפיסים את דירוג ה-Elo הנוכחי שלהן שיהיה לך כרפרנס
    print(f"Current Elo Rankings:")
    print(f"{team_a_name}: {team_elos[team_a_name]:.1f}")
    print(f"{team_b_name}: {team_elos[team_b_name]:.1f}")
    
    # קוראים לפונקציה שתפיק את הטופס!
    predict_match_outcome(model, history_a, history_b, team_a_name, team_b_name)

if __name__ == "__main__":
    main_live_predictor()