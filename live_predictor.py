import torch
import pickle
from scipy.stats import poisson

from world_cup_model import WorldCupTransformer

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
    
    # predictions = []

    # print("-" * 35)
    # print("Most recommended outcomes:")
    # for i in range(3):
    #     ga, gb, p = probabilities[i]
    #     print(f"Exact score: {ga}-{gb}  |  Probability: {p:.1f}%")
    # print("-" * 35)
    # predictions.append([ga, gb, p])

    return probabilities[:3]  # Return top 3 predictions as a list of tuples (goals_a, goals_b, probability)


def main_live_predictor(team1_input, team2_input):
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
    # team_a_name = "United States"
    # team_b_name = "Paraguay"
    team_a_name = team1_input
    team_b_name = team2_input
    
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
    return predict_match_outcome(model, history_a, history_b, team_a_name, team_b_name)

# if __name__ == "__main__":
#     main_live_predictor()