
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------

st.set_page_config(
    page_title="Self-Attention Visualization",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Self-Attention Visualization")

st.write(
    "This page demonstrates how self-attention combines "
    "information across input tokens."
)

st.warning(
    "This is a synthetic educational demonstration. "
    "It does not analyze real patient records or make "
    "medical safety or dosage recommendations."
)


# -----------------------------------------
# 2. SELF-ATTENTION MODEL
# -----------------------------------------

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


# -----------------------------------------
# 3. GENERATE DEMONSTRATION INPUT
# -----------------------------------------

st.subheader("Step 1: Generate synthetic input")

num_tokens = st.slider(
    "Number of input tokens",
    min_value=3,
    max_value=8,
    value=5
)

if st.button("Run Self-Attention"):

    torch.manual_seed(42)

    # Synthetic values, not patient data
    x = torch.randn(1, num_tokens, 8)

    model = SelfAttention()
    model.eval()

    with torch.no_grad():
        output, weights = model(x)

    attention_matrix = weights[0].numpy()

    st.success("Self-attention completed successfully!")

    # -----------------------------------------
    # 4. DISPLAY OUTPUT SHAPES
    # -----------------------------------------

    st.subheader("Step 2: Model output")

    col1, col2 = st.columns(2)

    col1.metric(
        "Input tokens",
        num_tokens
    )

    col2.metric(
        "Output tokens",
        output.shape[1]
    )

    st.write("Input shape:", tuple(x.shape))
    st.write("Output shape:", tuple(output.shape))

    # -----------------------------------------
    # 5. ATTENTION HEATMAP
    # -----------------------------------------

    st.subheader("Step 3: Attention Weight Heatmap")

    fig, ax = plt.subplots(figsize=(8, 6))

    image = ax.imshow(
        attention_matrix,
        cmap="Blues",
        vmin=0,
        vmax=1
    )

    ax.set_xlabel("Token being attended to")
    ax.set_ylabel("Token calculating attention")
    ax.set_title("Self-Attention Weights")

    ax.set_xticks(range(num_tokens))
    ax.set_yticks(range(num_tokens))

    ax.set_xticklabels([
        f"Token {i + 1}" for i in range(num_tokens)
    ])

    ax.set_yticklabels([
        f"Token {i + 1}" for i in range(num_tokens)
    ])

    for i in range(num_tokens):
        for j in range(num_tokens):
            ax.text(
                j,
                i,
                f"{attention_matrix[i, j]:.2f}",
                ha="center",
                va="center"
            )

    fig.colorbar(image, ax=ax, label="Attention weight")

    st.pyplot(fig)
    plt.close(fig)

    # -----------------------------------------
    # 6. EXPLANATION
    # -----------------------------------------

    st.subheader("Step 4: Understand the heatmap")

    st.write(
        "- Each row represents one input token."
    )

    st.write(
        "- Each column represents a token it can attend to."
    )

    st.write(
        "- Larger weights indicate greater attention "
        "within this demonstration."
    )

    st.write(
        "- Each row's attention weights sum to approximately 1."
    )

    st.write(
        "These weights describe the behavior of this small "
        "randomly initialized model. They are not medical "
        "risk scores or explanations of clinical outcomes."
    )

else:
    st.info(
        "Choose the number of tokens and click "
        "'Run Self-Attention' to generate the heatmap."
    )
