import pickle

import torch
from scipy.stats import poisson

from world_cup_prediction.config import FINAL_MODEL_PATH, NUM_FEATURES, SEQ_LEN, TOURNAMENT_STATE_PATH
from world_cup_prediction.model import WorldCupTransformer


def predict_match_outcome(model, team_a_history, team_b_history, team_a_name="Group A", team_b_name="Group B"):
    model.eval()
    with torch.no_grad():
        team_a_tensor = torch.tensor(team_a_history, dtype=torch.float32).unsqueeze(0)
        team_b_tensor = torch.tensor(team_b_history, dtype=torch.float32).unsqueeze(0)

        lambdas = model(team_a_tensor, team_b_tensor)
        lam_a = lambdas[0][0].item()
        lam_b = lambdas[0][1].item()

    print(f"\n[{team_a_name} vs {team_b_name}]")
    print(f"Calculated expected goals (xG): {team_a_name} ({lam_a:.2f}) - {team_b_name} ({lam_b:.2f})")

    probabilities = []
    for goals_a in range(6):
        for goals_b in range(6):
            prob_a = poisson.pmf(goals_a, lam_a)
            prob_b = poisson.pmf(goals_b, lam_b)
            exact_prob = prob_a * prob_b * 100
            probabilities.append((goals_a, goals_b, exact_prob))

    probabilities.sort(key=lambda x: x[2], reverse=True)
    return probabilities[:3]


def main_live_predictor(team1_input, team2_input):
    print("Loading World Cup AI Assistant...")

    model = WorldCupTransformer(num_features=NUM_FEATURES)
    model.load_state_dict(torch.load(FINAL_MODEL_PATH))
    model.eval()

    with open(TOURNAMENT_STATE_PATH, 'rb') as f:
        state = pickle.load(f)

    team_elos = state['elos']
    team_histories = state['histories']

    print("System Ready!\n")
    print("-" * 30)

    team_a_name = team1_input
    team_b_name = team2_input

    if team_a_name not in team_histories or team_b_name not in team_histories:
        print("שגיאה: אחת הקבוצות לא נמצאה במסד הנתונים. ודא איות באנגלית.")
        return

    history_a = team_histories[team_a_name][-SEQ_LEN:]
    history_b = team_histories[team_b_name][-SEQ_LEN:]

    print("Current Elo Rankings:")
    print(f"{team_a_name}: {team_elos[team_a_name]:.1f}")
    print(f"{team_b_name}: {team_elos[team_b_name]:.1f}")

    return predict_match_outcome(model, history_a, history_b, team_a_name, team_b_name)
