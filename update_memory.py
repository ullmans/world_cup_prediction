import pickle

from world_cup_model import update_elo
# ==========================================
# מנוע העדכון היומי
# ==========================================
def update_tournament_memory(team_a, team_b, score_a, score_b):
    print(f"מעדכן נתונים עבור המשחק: {team_a} {score_a} - {score_b} {team_b}...")
    
    # 1. טעינת הזיכרון הנוכחי
    try:
        with open('tournament_state.pkl', 'rb') as f:
            state = pickle.load(f)
    except FileNotFoundError:
        print("שגיאה: קובץ tournament_state.pkl לא נמצא!")
        return
        
    team_elos = state['elos']
    team_histories = state['histories']
    
    if team_a not in team_elos or team_b not in team_elos:
        print("שגיאה: שם אחת הקבוצות לא קיים במסד הנתונים. בדוק איות.")
        return

    # 2. שליפת ה-Elo שהיה לקבוצות *לפני* המשחק
    current_elo_a = team_elos[team_a]
    current_elo_b = team_elos[team_b]
    
    # 3. עדכון חלון הנתונים (Sliding Window)
    # נזכור את המבנה: [Elo עצמי (מחולק ב-1000), Elo יריבה (מחולק ב-1000), זכות, חובה, מגרש ניטרלי (1.0)]
    match_data_a = [current_elo_a / 1000.0, current_elo_b / 1000.0, score_a, score_b, 1.0]
    match_data_b = [current_elo_b / 1000.0, current_elo_a / 1000.0, score_b, score_a, 1.0]
    
    team_histories[team_a].append(match_data_a)
    team_histories[team_b].append(match_data_b)
    
    # חיתוך הרשימה כדי לשמור רק את 5 המשחקים האחרונים
    team_histories[team_a] = team_histories[team_a][-5:]
    team_histories[team_b] = team_histories[team_b][-5:]
    
    # 4. חישוב ושמירת ה-Elo החדש *אחרי* המשחק
    new_elo_a, new_elo_b = update_elo(current_elo_a, current_elo_b, score_a, score_b)
    team_elos[team_a] = new_elo_a
    team_elos[team_b] = new_elo_b
    
    # 5. שמירת הזיכרון המעודכן בחזרה לקובץ
    updated_state = {'elos': team_elos, 'histories': team_histories}
    with open('tournament_state.pkl', 'wb') as f:
        pickle.dump(updated_state, f)
        
    print(f"העדכון בוצע בהצלחה ונשמר!")
    print(f"דירוג מעודכן: {team_a} ({new_elo_a:.1f}) | {team_b} ({new_elo_b:.1f})\n")

    return True

# ==========================================
# הפעלת הסקריפט
# ==========================================
if __name__ == "__main__":
    # כאן אתה מזין כל ערב את התוצאות האמיתיות שהסתיימו
    # לדוגמה: ארגנטינה ניצחה את מקסיקו 2-1
    update_tournament_memory("England", "Iraq", 1, 0)
    
    # אם היו כמה משחקים באותו יום, פשוט תוסיף עוד שורות:
    # update_tournament_memory("France", "Australia", 4, 1)