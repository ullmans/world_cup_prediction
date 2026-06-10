def build_match_features(home_elo, away_elo, home_score, away_score, is_neutral, squad_diff=0.0):
    # 1. פער ה-Elo
    elo_diff = (home_elo - away_elo) / 1000.0
    
    # 2. הפרש שערים
    goal_diff = home_score - away_score
    
    # 3. יתרון ביתיות
    home_adv = 0.0 if is_neutral else 1.0
    away_adv = 0.0 if is_neutral else -1.0
    
    # בניית מערך הפיצ'רים (6 איברים)
    # [פער דירוג, זכות, חובה, הפרש, ביתיות, פער סגל]
    # home_features = [elo_diff, home_score, away_score, goal_diff, home_adv, squad_diff]    
    home_features = [elo_diff, home_score, away_score, goal_diff, home_adv]

    # לקבוצת החוץ הפערים תמיד הפוכים
    # away_features = [-elo_diff, away_score, home_score, -goal_diff, away_adv, -squad_diff]
    away_features = [-elo_diff, away_score, home_score, -goal_diff, away_adv]

    return home_features, away_features