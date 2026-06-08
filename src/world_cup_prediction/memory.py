import pickle

from world_cup_prediction.config import SEQ_LEN, TOURNAMENT_STATE_PATH
from world_cup_prediction.elo import update_elo
from world_cup_prediction.features import build_match_features


def update_tournament_memory(team_a, team_b, score_a, score_b, is_neutral=False):
    print(f"מעדכן נתונים עבור המשחק: {team_a} {score_a} - {score_b} {team_b}...")

    try:
        with open(TOURNAMENT_STATE_PATH, 'rb') as f:
            state = pickle.load(f)
    except FileNotFoundError:
        print(f"שגיאה: קובץ {TOURNAMENT_STATE_PATH} לא נמצא!")
        return

    team_elos = state['elos']
    team_histories = state['histories']

    if team_a not in team_elos or team_b not in team_elos:
        print("שגיאה: שם אחת הקבוצות לא קיים במסד הנתונים. בדוק איות.")
        return

    current_elo_a = team_elos[team_a]
    current_elo_b = team_elos[team_b]

    match_data_a, match_data_b = build_match_features(
        current_elo_a, current_elo_b, score_a, score_b, is_neutral=is_neutral
    )

    team_histories[team_a].append(match_data_a)
    team_histories[team_b].append(match_data_b)

    team_histories[team_a] = team_histories[team_a][-SEQ_LEN:]
    team_histories[team_b] = team_histories[team_b][-SEQ_LEN:]

    new_elo_a, new_elo_b = update_elo(current_elo_a, current_elo_b, score_a, score_b)
    team_elos[team_a] = new_elo_a
    team_elos[team_b] = new_elo_b

    updated_state = {'elos': team_elos, 'histories': team_histories}
    TOURNAMENT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TOURNAMENT_STATE_PATH, 'wb') as f:
        pickle.dump(updated_state, f)

    print("העדכון בוצע בהצלחה ונשמר!")
    print(f"דירוג מעודכן: {team_a} ({new_elo_a:.1f}) | {team_b} ({new_elo_b:.1f})\n")

    return True


if __name__ == "__main__":
    update_tournament_memory("England", "Iraq", 1, 0)
