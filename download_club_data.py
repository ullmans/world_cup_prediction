import pandas as pd
import urllib.request
import ssl

def download_and_process_club_data():
    print("מתחיל בשאיבת נתונים מ-football-data.co.uk...")
    
    # עקיפת בעיות אבטחת SSL בהורדה ישירה בפייתון
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # מבנה ה-URL של האתר: https://www.football-data.co.uk/mmz4281/{season}/{league}.csv
    base_url = "https://www.football-data.co.uk/mmz4281/"
    
    # הליגות שאנחנו רוצים (הקודים כפי שהם מופיעים באתר)
    leagues = {
        'E0': 'Premier League',  # אנגליה
        'SP1': 'La Liga',        # ספרד
        'D1': 'Bundesliga',      # גרמניה
        'I1': 'Serie A',         # איטליה
        'F1': 'Ligue 1'          # צרפת
    }
    
    # יצירת רשימת העונות (מ-2010/2011 ועד 2023/2024)
    # הפורמט באתר הוא '1011', '1112', ..., '2324'
    seasons = [f"{str(year)[-2:]}{str(year+1)[-2:]}" for year in range(2010, 2024)]
    
    all_matches = []
    
    for season in seasons:
        for league_code, league_name in leagues.items():
            url = f"{base_url}{season}/{league_code}.csv"
            print(f"מוריד: עונת 20{season[:2]}/20{season[2:]} - {league_name}...")
            
            try:
                # קריאת ה-CSV ישירות מהאינטרנט
                # משתמשים ב-unicode_escape כי לפעמים יש שמות שחקנים/קבוצות עם תווים מיוחדים
                df = pd.read_csv(url, encoding='unicode_escape', on_bad_lines='skip')
                
                # העמודות באתר הן: Date, HomeTeam, AwayTeam, FTHG (Full Time Home Goals), FTAG (Full Time Away Goals)
                # נוודא שהעמודות קיימות (לפעמים יש קבצים פגומים באתר)
                required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
                if not set(required_cols).issubset(df.columns):
                    print(f"   -> דילוג: חסרות עמודות קריטיות בקובץ.")
                    continue
                
                # נשמור רק את העמודות שמעניינות אותנו
                df_clean = df[required_cols].copy()
                
                # נוסיף את שם הטורניר
                df_clean['tournament'] = league_name
                
                all_matches.append(df_clean)
                
            except urllib.error.HTTPError:
                print(f"   -> שגיאה: הקובץ לא נמצא בשרת ({url})")
            except Exception as e:
                print(f"   -> שגיאה כללית: {e}")

    print("\nמאחד את כל הנתונים...")
    final_df = pd.concat(all_matches, ignore_index=True)
    
    # שינוי שמות העמודות שיתאימו בדיוק למחלקה FootballDataset שלנו
    final_df.columns = ['date', 'home_team', 'away_team', 'home_score', 'away_score', 'tournament']
    
    # ניקוי נתונים: הסרת שורות ריקות לחלוטין שאולי השתרבבו
    final_df.dropna(subset=['home_score', 'away_score', 'date'], inplace=True)
    
    # שמירה לקובץ
    output_filename = 'club_results.csv'
    final_df.to_csv(output_filename, index=False)
    
    print("-" * 50)
    print(f"התהליך הושלם בהצלחה!")
    print(f"נוצר הקובץ '{output_filename}' המכיל {len(final_df)} משחקים רשמיים של מועדונים.")
    print("הקובץ מוכן לשלב ה-Pre-training של הרשת שלך!")

if __name__ == "__main__":
    download_and_process_club_data()