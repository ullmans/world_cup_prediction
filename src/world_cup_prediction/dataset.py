import pandas as pd
import torch
from torch.utils.data import Dataset

from world_cup_prediction.config import SEQ_LEN
from world_cup_prediction.elo import update_elo
from world_cup_prediction.features import build_match_features


class FootballDataset(Dataset):
    def __init__(self, csv_path, seq_len=SEQ_LEN, split_type='train', train_ratio = 0.9):
        if split_type =='train':
            print(f"Loading data for {split_type.upper()} split (Train size: {train_ratio * 100}%)...")
        else:
            print(f"Loading data for {split_type.upper()} split (Test size: {100 - train_ratio * 100}%)...")

        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, format='mixed')
        df = df.sort_values('date')

        df = df[df['date'].dt.year >= 2010]
        df = df[df['tournament'] != 'Friendly']
        df = df.dropna(subset=['home_score', 'away_score'])

        all_teams = pd.concat([df['home_team'], df['away_team']]).unique()
        team_elos = {team: 1500.0 for team in all_teams}
        team_histories = {team: [] for team in all_teams}

        all_a_data = []
        all_b_data = []
        all_actual_scores = []

        team_current_idx = {team: 0 for team in all_teams}

        for _, row in df.iterrows():
            home = row['home_team']
            away = row['away_team']

            current_home_elo = team_elos[home]
            current_away_elo = team_elos[away]
            is_neutral = bool('neutral' in df.columns and row['neutral'])

            idx_home = team_current_idx[home]
            idx_away = team_current_idx[away]

            if idx_home >= seq_len and idx_away >= seq_len:
                home_seq = team_histories[home][idx_home - seq_len : idx_home]
                away_seq = team_histories[away][idx_away - seq_len : idx_away]

                all_a_data.append(home_seq)
                all_b_data.append(away_seq)
                all_actual_scores.append([row['home_score'], row['away_score']])

            home_features, away_features = build_match_features(
                current_home_elo,
                current_away_elo,
                row['home_score'],
                row['away_score'],
                is_neutral=is_neutral,
            )
            team_histories[home].append(home_features)
            team_histories[away].append(away_features)

            team_current_idx[home] += 1
            team_current_idx[away] += 1

            new_home_elo, new_away_elo = update_elo(
                current_home_elo, current_away_elo, row['home_score'], row['away_score']
            )
            team_elos[home] = new_home_elo
            team_elos[away] = new_away_elo

        total_samples = len(all_actual_scores)
        split_idx = int(total_samples * train_ratio)
        if split_type == 'train':
            final_a = all_a_data[:split_idx]
            final_b = all_b_data[:split_idx]
            final_scores = all_actual_scores[:split_idx]
        else:
            final_a = all_a_data[split_idx:]
            final_b = all_b_data[split_idx:]
            final_scores = all_actual_scores[split_idx:]

        self.team_a_data = torch.tensor(final_a, dtype=torch.float32)
        self.team_b_data = torch.tensor(final_b, dtype=torch.float32)
        self.actual_scores = torch.tensor(final_scores, dtype=torch.float32)

        self.team_elos = team_elos
        self.team_histories = team_histories

        print(f"Successfully created {len(self.actual_scores)} examples for {split_type.upper()}!")

    def __len__(self):
        return len(self.actual_scores)

    def __getitem__(self, idx):
        return self.team_a_data[idx], self.team_b_data[idx], self.actual_scores[idx]
