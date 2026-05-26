"""
1D CNN Annotator (PyTorch)
Train CNN สำหรับ NMR annotation
Input: raw spectrum (20,000 points) [1]
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os


class NMR_CNN(nn.Module):
    """
    1D CNN สำหรับ classify NMR spectrum
    Input : (batch, 1, 20000) [1]
    Output: (batch, num_compounds) — multi-label
    """

    def __init__(self, spectrum_length, num_compounds):
        super(NMR_CNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),

            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),

            # Block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),

            # Block 4
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 32, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_compounds),
            nn.Sigmoid()  # Multi-label
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class CNNAnnotator:
    """
    CNN-based NMR Annotator
    """

    def __init__(self, compound_names, spectrum_length=20000):
        self.compound_names = compound_names
        self.spectrum_length = spectrum_length
        self.num_compounds = len(compound_names)
        self.threshold = 0.3
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"CNN using device: {self.device}")

        self.model = NMR_CNN(
            spectrum_length, self.num_compounds
        ).to(self.device)

    def train(self, X, y,
              epochs=30, batch_size=32,
              learning_rate=0.001, test_size=0.2):
        """
        Train CNN model

        Parameters:
            X     : spectra (n_samples, spectrum_length)
            y     : labels (n_samples, num_compounds)
            epochs: จำนวน epochs
        """
        print("\nTraining 1D CNN...")
        print("=" * 50)

        # Split data
        split = int(len(X) * (1 - test_size))
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]

        print(f"  Train: {len(X_tr)} samples")
        print(f"  Val  : {len(X_val)} samples")
        print(f"  Epochs: {epochs}")

        # DataLoader
        X_tr_t = torch.FloatTensor(X_tr).unsqueeze(1)
        y_tr_t = torch.FloatTensor(y_tr)
        X_val_t = torch.FloatTensor(X_val).unsqueeze(1)
        y_val_t = torch.FloatTensor(y_val)

        train_ds = TensorDataset(X_tr_t, y_tr_t)
        val_ds = TensorDataset(X_val_t, y_val_t)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size
        )

        # Optimizer + Loss
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.5
        )
        criterion = nn.BCELoss()

        best_val_loss = float('inf')
        best_state = None

        # Training loop
        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validate
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    output = self.model(X_batch)
                    loss = criterion(output, y_batch)
                    val_loss += loss.item()

            avg_train = train_loss / len(train_loader)
            avg_val = val_loss / len(val_loader)

            # Save best model
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_state = {
                    k: v.clone()
                    for k, v in self.model.state_dict().items()
                }

            scheduler.step()

            if (epoch + 1) % 5 == 0:
                print(
                    f"  Epoch {epoch+1:3d}/{epochs}"
                    f"  Train: {avg_train:.4f}"
                    f"  Val: {avg_val:.4f}"
                )

        # Load best model
        if best_state:
            self.model.load_state_dict(best_state)

        print(f"\n  Best val loss: {best_val_loss:.4f}")
        print("=" * 50)

    def predict(self, spectrum):
        """
        ทำนาย compounds จาก spectrum

        Parameters:
            spectrum: np.array (spectrum_length,)

        Returns:
            list: [{'compound': name, 'cnn_score': prob}, ...]
        """
        self.model.eval()

        X_tensor = torch.FloatTensor(spectrum)\
                        .unsqueeze(0)\
                        .unsqueeze(0)\
                        .to(self.device)

        with torch.no_grad():
            probs = self.model(X_tensor)
            probs = probs.cpu().numpy()[0]

        results = []
        for i, compound in enumerate(self.compound_names):
            prob = float(probs[i])
            if prob >= self.threshold:
                results.append({
                    'compound': compound,
                    'cnn_score': round(prob, 4)
                })

        return sorted(
            results,
            key=lambda x: x['cnn_score'],
            reverse=True
        )

    def save(self, path='data/trained_models/cnn_model.pt'):
        """บันทึก CNN model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'compound_names': self.compound_names,
            'spectrum_length': self.spectrum_length,
            'threshold': self.threshold
        }, path)
        print(f"CNN model saved: {path}")

    def load(self, path='data/trained_models/cnn_model.pt'):
        """โหลด CNN model"""
        checkpoint = torch.load(
            path, map_location=self.device
        )
        self.model.load_state_dict(
            checkpoint['model_state']
        )
        self.compound_names = checkpoint['compound_names']
        self.threshold = checkpoint['threshold']
        print(f"CNN model loaded: {path}")