import pandas as pd
import numpy as np

from world_cup_prediction.config import NATIONAL_RESULTS_CSV, FIFA_PLAYERS_CSV

def create_squad_features(matches_csv=NATIONAL_RESULTS_CSV, fifa_csv=FIFA_PLAYERS_CSV):
    print("1. Reading FIFA players dataset...")
    try:
        players_df = pd.read_csv(fifa_csv)
    except FileNotFoundError:
        print(f"Error: File {fifa_csv} not found. Please ensure it is in the directory.")
        return

    # Ensure critical columns exist
    required_cols = ['year', 'nationality', 'overall']
    if not all(col in players_df.columns for col in required_cols):
        print(f"Error: Players file must contain the columns: {required_cols}")
        return

    print("2. Calculating average of top 16 players per nation per year...")
    # Group by year and nation, sort by rating, and take top 16
    top_players = players_df.sort_values(['year', 'nationality', 'overall'], ascending=[True, True, False])
    top_16_per_nation = top_players.groupby(['year', 'nationality']).head(16)
    
    # Calculate the mean of those 16 players
    squad_ratings = top_16_per_nation.groupby(['year', 'nationality'])['overall'].mean().reset_index()
    squad_ratings.rename(columns={'overall': 'squad_rating'}, inplace=True)

    print("3. Loading national team match results...")
    matches_df = pd.read_csv(matches_csv)
    matches_df['date'] = pd.to_datetime(matches_df['date'])
    matches_df['match_year'] = matches_df['date'].dt.year

    # ==========================================
    # Sanitization Layer
    # ==========================================
    print("-> Cleaning whitespace and fixing data types for merging...")
    
    # Convert matches year to int
    matches_df['match_year'] = matches_df['match_year'].astype(int)
    
    # FIX: Convert FIFA year to int, and fix 2-digit years (e.g., 15 -> 2015)
    squad_ratings['year'] = squad_ratings['year'].astype(int)
    squad_ratings['year'] = squad_ratings['year'].apply(lambda y: y + 2000 if y < 100 else y)
    
    # Clean extra spaces and lowercase (for joining purposes only)
    squad_ratings['join_nation'] = squad_ratings['nationality'].astype(str).str.strip().str.lower()
    matches_df['join_home'] = matches_df['home_team'].astype(str).str.strip().str.lower()
    matches_df['join_away'] = matches_df['away_team'].astype(str).str.strip().str.lower()

    # Handle historical naming differences between FIFA and matches dataset
    mapping_fixes = {
        'usa': 'united states',
        'korea republic': 'south korea',
        'côte d\'ivoire': 'ivory coast',
        'republic of ireland': 'ireland'
    }
    squad_ratings['join_nation'] = squad_ratings['join_nation'].replace(mapping_fixes)

    print("4. Merging squad ratings into national matches...")
    # Merge for Home team
    matches_df = pd.merge(
        matches_df, 
        squad_ratings[['year', 'join_nation', 'squad_rating']], 
        left_on=['match_year', 'join_home'], 
        right_on=['year', 'join_nation'], 
        how='left'
    )
    matches_df.rename(columns={'squad_rating': 'home_squad_rating'}, inplace=True)
    matches_df.drop(columns=['year', 'join_nation'], inplace=True)

    # Merge for Away team
    matches_df = pd.merge(
        matches_df, 
        squad_ratings[['year', 'join_nation', 'squad_rating']], 
        left_on=['match_year', 'join_away'], 
        right_on=['year', 'join_nation'], 
        how='left'
    )
    matches_df.rename(columns={'squad_rating': 'away_squad_rating'}, inplace=True)
    matches_df.drop(columns=['year', 'join_nation', 'join_home', 'join_away'], inplace=True)

    print("5. Handling missing national teams (Fallback)...")
    DEFAULT_WEAK_RATING = 65.0
    missing_home = matches_df['home_squad_rating'].isna().sum()
    missing_away = matches_df['away_squad_rating'].isna().sum()
    
    total_rows = len(matches_df)
    print(f"   -> Missing home data for {missing_home}/{total_rows} matches.")
    print(f"   -> Missing away data for {missing_away}/{total_rows} matches.")
    
    matches_df['home_squad_rating'] = matches_df['home_squad_rating'].fillna(DEFAULT_WEAK_RATING)
    matches_df['away_squad_rating'] = matches_df['away_squad_rating'].fillna(DEFAULT_WEAK_RATING)

    # Save the updated file
    output_filename = 'national_results_with_squads.csv'
    matches_df.to_csv(output_filename, index=False)
    print("-" * 40)
    print(f"Process complete! Created new file: {output_filename}")
    print("File now contains 'home_squad_rating' and 'away_squad_rating' columns.")

if __name__ == "__main__":
    create_squad_features()