import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from world_cup_prediction.config import (
    BATCH_SIZE,
    CLUB_RESULTS_CSV,
    FINAL_MODEL_PATH,
    NATIONAL_RESULTS_CSV,
    NUM_FEATURES,
    PRETRAINED_CLUBS_PATH,
    SEQ_LEN,
    TOURNAMENT_STATE_PATH,
)
from world_cup_prediction.dataset import FootballDataset
from world_cup_prediction.model import WorldCupTransformer


def train_and_evaluate(model, train_loader, test_loader, epochs, lr, weight_decay, save_path):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.PoissonNLLLoss(log_input=False)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch_a, batch_b, target_scores in train_loader:
            optimizer.zero_grad()

            pred_lambdas = model(batch_a, batch_b)
            pred_lambdas = torch.clamp(pred_lambdas, min=1e-4, max=50.0)

            loss = criterion(pred_lambdas, target_scores)
            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for batch_a, batch_b, target_scores in test_loader:
                pred_lambdas = model(batch_a, batch_b)
                pred_lambdas = torch.clamp(pred_lambdas, min=1e-4, max=50.0)

                loss = criterion(pred_lambdas, target_scores)
                if not torch.isnan(loss):
                    total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}"
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"--> נשמר קובץ משקולות: {save_path}\n")
    return model


def main():
    print("מתחיל מערכת Transfer Learning...")

    ARTIFACTS_DIR = FINAL_MODEL_PATH.parent
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model = WorldCupTransformer(num_features=NUM_FEATURES)

    print("==================================================")
    print("PHASE 1: Pre-training on Club Data")
    print("==================================================")

    club_train = FootballDataset(
        csv_path=CLUB_RESULTS_CSV, seq_len=SEQ_LEN, split_type='train', test_size=0.1
    )
    club_test = FootballDataset(
        csv_path=CLUB_RESULTS_CSV, seq_len=SEQ_LEN, split_type='test', test_size=0.1
    )

    club_train_loader = DataLoader(club_train, batch_size=BATCH_SIZE, shuffle=True)
    club_test_loader = DataLoader(club_test, batch_size=BATCH_SIZE, shuffle=False)

    model = train_and_evaluate(
        model=model,
        train_loader=club_train_loader,
        test_loader=club_test_loader,
        epochs=15,
        lr=0.0001,
        weight_decay=1e-4,
        save_path=PRETRAINED_CLUBS_PATH,
    )

    print("==================================================")
    print("PHASE 2: Fine-Tuning on National Teams Data")
    print("==================================================")

    nat_train = FootballDataset(
        csv_path=NATIONAL_RESULTS_CSV, seq_len=SEQ_LEN, split_type='train', test_size=0.1
    )
    nat_test = FootballDataset(
        csv_path=NATIONAL_RESULTS_CSV, seq_len=SEQ_LEN, split_type='test', test_size=0.1
    )

    nat_train_loader = DataLoader(nat_train, batch_size=BATCH_SIZE, shuffle=True)
    nat_test_loader = DataLoader(nat_test, batch_size=BATCH_SIZE, shuffle=False)

    model.load_state_dict(torch.load(PRETRAINED_CLUBS_PATH))

    model = train_and_evaluate(
        model=model,
        train_loader=nat_train_loader,
        test_loader=nat_test_loader,
        epochs=4,
        lr=0.00001,
        weight_decay=1e-4,
        save_path=FINAL_MODEL_PATH,
    )

    tournament_state = {
        'elos': nat_test.team_elos,
        'histories': nat_test.team_histories,
    }
    with open(TOURNAMENT_STATE_PATH, 'wb') as f:
        pickle.dump(tournament_state, f)
    print(f"Tournament state saved to '{TOURNAMENT_STATE_PATH}' - המערכת מוכנה למונדיאל!")


if __name__ == "__main__":
    main()
