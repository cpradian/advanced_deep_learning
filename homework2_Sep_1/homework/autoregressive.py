import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.
        """

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Decoder-only style autoregressive transformer over image tokens.
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        self.d_latent = d_latent
        self.n_tokens = n_tokens

        self.token_embedding = torch.nn.Embedding(n_tokens, d_latent)
        self.pos_embedding = torch.nn.Embedding(1024, d_latent)

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_latent,
            nhead=8,
            dim_feedforward=4 * d_latent,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.ln = torch.nn.LayerNorm(d_latent)
        self.head = torch.nn.Linear(d_latent, n_tokens)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        x: (B, h, w) integer token grid
        returns logits: (B, h, w, n_tokens)
        """
        B, h, w = x.shape
        T = h * w

        x_flat = x.view(B, T)  # (B, T)

        # Embed tokens
        tok = self.token_embedding(x_flat)  # (B, T, d)

        # Shift right by one position so token t predicts x_t using only < t
        start = torch.zeros(B, 1, self.d_latent, device=x.device, dtype=tok.dtype)
        tok_shifted = torch.cat([start, tok[:, :-1]], dim=1)  # (B, T, d)

        # Positional embedding
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        z = tok_shifted + self.pos_embedding(pos_ids)  # (B, T, d)

        # Causal mask: token t cannot see future tokens
        mask = torch.nn.Transformer.generate_square_subsequent_mask(T, device=x.device)

        z = self.transformer(z, mask=mask)
        z = self.ln(z)
        logits = self.head(z)  # (B, T, n_tokens)

        return logits.view(B, h, w, self.n_tokens), {}

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        if device is None:
            device = next(self.parameters()).device

        T = h * w
        out = torch.zeros(B, T, dtype=torch.long, device=device)

        self.eval()
        with torch.no_grad():
            for t in range(T):
                logits, _ = self.forward(out.view(B, h, w))
                logits_t = logits.view(B, T, self.n_tokens)[:, t, :]  # (B, n_tokens)

                probs = torch.softmax(logits_t, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)

                out[:, t] = next_token

        return out.view(B, h, w)