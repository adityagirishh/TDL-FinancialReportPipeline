import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionRiskModel(nn.Module):
    def __init__(self):
        super(FusionRiskModel, self).__init__()

        # Sentiment (3,)
        self.sentiment_fc = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Image features (128,)
        self.image_fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Time series (64,)
        self.timeseries_fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Financials (32,)
        self.financials_fc = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Fusion layer
        self.fusion_fc = nn.Sequential(
            nn.Linear(16 + 64 + 32 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),  # Final risk score output
            nn.Sigmoid()  # Risk score between 0 and 1
        )

    def forward(self, sentiment, image_feat, timeseries_feat, financial_feat):
        s = self.sentiment_fc(sentiment)
        i = self.image_fc(image_feat)
        t = self.timeseries_fc(timeseries_feat)
        f = self.financials_fc(financial_feat)

        x = torch.cat([s, i, t, f], dim=1)
        out = self.fusion_fc(x)
        return out
