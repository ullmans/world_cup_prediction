def calculate_expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(rating_a, rating_b, score_a, score_b, k_factor=30):
    expected_a = calculate_expected_score(rating_a, rating_b)
    expected_b = calculate_expected_score(rating_b, rating_a)

    if score_a > score_b:
        actual_a, actual_b = 1.0, 0.0
    elif score_a < score_b:
        actual_a, actual_b = 0.0, 1.0
    else:
        actual_a, actual_b = 0.5, 0.5

    new_rating_a = rating_a + k_factor * (actual_a - expected_a)
    new_rating_b = rating_b + k_factor * (actual_b - expected_b)

    return new_rating_a, new_rating_b
