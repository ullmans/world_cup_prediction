def build_match_features(home_elo, away_elo, home_score, away_score, is_neutral=False):
    """Build the 5-feature history vectors used by training and live memory updates."""
    elo_diff = (home_elo - away_elo) / 1000.0
    goal_diff = home_score - away_score
    home_adv = 0.0 if is_neutral else 1.0
    away_adv = 0.0 if is_neutral else -1.0

    home_features = [elo_diff, home_score, away_score, goal_diff, home_adv]
    away_features = [-elo_diff, away_score, home_score, -goal_diff, away_adv]

    return home_features, away_features
