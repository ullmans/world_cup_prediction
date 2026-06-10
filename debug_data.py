import pandas as pd
from world_cup_prediction.config import NATIONAL_RESULTS_CSV, FIFA_PLAYERS_CSV

print("Loading files for inspection...")
matches_df = pd.read_csv(NATIONAL_RESULTS_CSV)
matches_df['date'] = pd.to_datetime(matches_df['date'])
matches_df['match_year'] = matches_df['date'].dt.year

players_df = pd.read_csv(FIFA_PLAYERS_CSV)

print("\n=== YEAR FORMAT ===")
print(f"Matches file years (Sample): {matches_df['match_year'].dropna().unique()[-5:]}")
print(f"FIFA file years (Sample): {players_df['year'].dropna().unique()[:5]}")

print("\n=== COUNTRY FORMAT ===")
print(f"Matches file countries (Sample): {matches_df['home_team'].dropna().unique()[:5]}")
print(f"FIFA file countries (Sample): {players_df['nationality'].dropna().unique()[:5]}")