import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict

class SatisfactionClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.model = self._build_model(input_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def _build_model(self, input_dim: int) -> nn.Sequential:
        """
        Build the DNN model architecture
        
        Args:
            input_dim: Number of input features
            
        Returns:
            PyTorch Sequential model
        """
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def train(self, 
             X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: np.ndarray,
             y_val: np.ndarray,
             batch_size: int = 32,
             epochs: int = 50) -> Dict:
        """
        Train the satisfaction classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            batch_size: Batch size for training
            epochs: Number of training epochs
            
        Returns:
            Training history
        """
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = nn.BCELoss()
        
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)
        
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val.unsqueeze(1))
            
            history['train_loss'].append(loss.item())
            history['val_loss'].append(val_loss.item())
            
            # Early stopping
            if epoch > 5 and val_loss > min(history['val_loss'][:-1]):
                break
                
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            X: Input features
            
        Returns:
            Predicted satisfaction scores
        """
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X)
            return predictions.cpu().numpy()
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        with torch.no_grad():
            X_test = torch.FloatTensor(X_test).to(self.device)
            y_test = torch.FloatTensor(y_test).to(self.device)
            
            outputs = self.model(X_test)
            predictions = (outputs > 0.5).float()
            accuracy = (predictions == y_test.unsqueeze(1)).float().mean()
            
            criterion = nn.BCELoss()
            loss = criterion(outputs, y_test.unsqueeze(1))
            
            return loss.item(), accuracy.item() 