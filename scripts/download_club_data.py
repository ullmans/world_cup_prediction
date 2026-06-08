import ssl
import urllib.request

import pandas as pd

from world_cup_prediction.config import CLUB_RESULTS_CSV


def download_and_process_club_data():
    print("מתחיל בשאיבת נתונים מ-football-data.co.uk...")

    ssl._create_default_https_context = ssl._create_unverified_context

    base_url = "https://www.football-data.co.uk/mmz4281/"

    leagues = {
        'E0': 'Premier League',
        'SP1': 'La Liga',
        'D1': 'Bundesliga',
        'I1': 'Serie A',
        'F1': 'Ligue 1',
    }

    seasons = [f"{str(year)[-2:]}{str(year + 1)[-2:]}" for year in range(2010, 2024)]

    all_matches = []

    for season in seasons:
        for league_code, league_name in leagues.items():
            url = f"{base_url}{season}/{league_code}.csv"
            print(f"מוריד: עונת 20{season[:2]}/20{season[2:]} - {league_name}...")

            try:
                df = pd.read_csv(url, encoding='unicode_escape', on_bad_lines='skip')

                required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
                if not set(required_cols).issubset(df.columns):
                    print("   -> דילוג: חסרות עמודות קריטיות בקובץ.")
                    continue

                df_clean = df[required_cols].copy()
                df_clean['tournament'] = league_name

                all_matches.append(df_clean)

            except urllib.error.HTTPError:
                print(f"   -> שגיאה: הקובץ לא נמצא בשרת ({url})")
            except Exception as e:
                print(f"   -> שגיאה כללית: {e}")

    print("\nמאחד את כל הנתונים...")
    final_df = pd.concat(all_matches, ignore_index=True)

    final_df.columns = ['date', 'home_team', 'away_team', 'home_score', 'away_score', 'tournament']
    final_df.dropna(subset=['home_score', 'away_score', 'date'], inplace=True)

    CLUB_RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(CLUB_RESULTS_CSV, index=False)

    print("-" * 50)
    print("התהליך הושלם בהצלחה!")
    print(f"נוצר הקובץ '{CLUB_RESULTS_CSV}' המכיל {len(final_df)} משחקים רשמיים של מועדונים.")
    print("הקובץ מוכן לשלב ה-Pre-training של הרשת שלך!")


if __name__ == "__main__":
    download_and_process_club_data()
