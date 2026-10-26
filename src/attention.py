
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, input_dim=8, attention_dim=8):
        super().__init__()

        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = torch.matmul(
            q, k.transpose(-2, -1)
        ) / (q.size(-1) ** 0.5)

        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, v)

        return output, weights


if __name__ == "__main__":
    torch.manual_seed(42)

    # Five synthetic feature tokens, each with eight values.
    # These are demonstration inputs, not patient data.
    x = torch.randn(1, 5, 8)

    model = SelfAttention()
    output, weights = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("\nAttention weights:")
    print(weights.detach().squeeze(0))

    print("\nSelf-attention module ran successfully!")
