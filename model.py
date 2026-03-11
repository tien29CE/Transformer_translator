import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: str):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, drop_out: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.drop_out = nn.Dropout(drop_out)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply the sin to even position, cos to odd position
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.drop_out(x)

class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multipied
        self.bias = nn.Parameter(torch.ones(1)) # Added

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, drop_out: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.drop_out = nn.Dropout(drop_out)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2

    def forward(self, x: torch.Tensor):
        # (Batch, Seq_len, d_model) ---> RELU(Batch, seq_len, d_ff) --> (Batch, seq_len, d_model)
        return self.linear_2(self.drop_out(torch.relu(self.linear_1(x))))

class MultiheadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, drop_out: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) # W_q
        self.w_k = nn.Linear(d_model, d_model) # W_k
        self.w_v = nn.Linear(d_model, d_model) # W_v

        self.w_o = nn.Linear(d_model, d_model) # W_o
        self.drop_out = nn.Dropout(drop_out)

    @staticmethod
    def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask, drop_out: nn.Dropout) -> tuple[torch.Tensor, torch.Tensor]:
        d_k = query.shape[-1]

        # (Batch, h, Seq_len, d_k) ---> (Batch, h, seq_len, seq_len)
        attention_scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1) # (Batch, h, seq_len, seq_len)

        if drop_out is None:
            attention_scores = drop_out(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask):
        query = self.w_q(q) # (Batch, Seq_len, d_module) ---> (Batch, Seq_len, d_module)
        key = self.w_k(k)   # (Batch, Seq_len, d_module) ---> (Batch, Seq_len, d_module)
        value = self.w_v(v) # (Batch, Seq_len, d_module) ---> (Batch, Seq_len, d_module)

        # (Batch, Seq_len, d_model) ---> (Batch, seq_len, h, d_k) ---> (Batch, h, Seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiheadAttentionBlock.attention(query, key, value, mask, self.drop_out)

        # (Batch, h, seq_len, d_k) ---> (Batch, Seq_len, h, d_k) ---> (Batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (Batch, seq_len, d_model) ---> (Batch, seq_len, d_model)
        return self.w_o(x)

class ResidualConnection(nn.Module):

    def __init__(self, drop_out: float) -> None:
        super().__init__()
        self.drop_out = nn.Dropout(drop_out)
        self.norm = LayerNormalization()

    def forward(self, x: torch.Tensor, sublayer):
        return x + self.drop_out(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiheadAttentionBlock, feed_forward_block: FeedForwardBlock, drop_out: float) -> float:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(drop_out) for _ in range(2)])

    def forward(self, x: torch.Tensor, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)

        return x

class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)

class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiheadAttentionBlock, cross_attention_block: MultiheadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, drop_out: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(drop_out) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, target_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, target_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)

        return x

class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x: torch.Tensor, encoder_ouput, src_mask, target_mask) -> str:
        for layer in self.layers:
            x = layer.forward(x, encoder_ouput, src_mask, target_mask)

        return self.norm(x)

class ProjectionLayer(nn.Module):

    def __init__(self, d_module: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_module, vocab_size)

    def forward(self, x: torch.Tensor):
        # (Batch, seq_len, d_model) ---> (Batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)

class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, target_embed: InputEmbedding,
                 src_pos: PositionEncoding, target_pos: PositionEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.target_embed = target_embed
        self.src_pos = src_pos
        self.target_pos = target_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)

        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, target, target_mask):
        target = self.target_embed(target)
        target = self.target_pos(target)

        return self.decoder(target, encoder_output, src_mask, target_mask)
    
    def project(self, x):
        return self.projection_layer(x)

def build_tranformer(src_vocab_size: int, target_vocab_size: int, src_seq_len: int, target_seq_len: int,
                     d_model: int = 512, N: int = 6, h: int = 8, drop_out: float = 0.1, d_ff: int = 2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbedding(d_model, src_vocab_size)
    target_embed = InputEmbedding(d_model, target_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionEncoding(d_model, src_seq_len, drop_out)
    target_pos = PositionEncoding(d_model, target_seq_len, drop_out)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiheadAttentionBlock(d_model, h, drop_out)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, drop_out)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, drop_out)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks]
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiheadAttentionBlock(d_model, h, drop_out)
        decoder_cross_attention_block = MultiheadAttentionBlock(d_model, h, drop_out)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, drop_out)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, drop_out)
        decoder_blocks.append(decoder_block)

    # Create the encoder and the decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create projection layer
    projection_layer = ProjectionLayer(d_model, target_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, target_embed, src_pos, target_pos, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer