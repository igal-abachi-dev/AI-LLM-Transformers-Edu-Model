import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. ARCHITECTURE CONFIGURATION
# =====================================================================
class MiniLLMConfig:
    vocab_size = 64      # Number of unique characters/tokens
    block_size = 128     # Maximum context window length (Max sequence length)
    n_embd = 128         # Embedding dimension size
    n_head = 4           # Number of attention heads (128 / 4 = 32 dim per head)
    n_layer = 3          # Number of transformer blocks
    bias = False         # True: bias in Linears, False: Llama-style cleaner math

# =====================================================================
# 2. CORE ATTENTION MECHANISM
# =====================================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Key, Query, Value projections combined into a single matrix
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection back to the network
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        # Causal mask blueprint to prevent looking ahead into the future
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # Batch size, Sequence length, Embedding channels

        # Calculate query, key, values for all heads in batch
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Causal self-attention math: (Q @ K^T) / sqrt(d_k)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # Fill future tokens with -infinity so their softmax probability drops to 0
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        # Reassemble all head outputs back into the standard shape
        y = y.transpose(1, 2).contiguous().view(B, T, C) 
        return self.c_proj(y)

# =====================================================================
# 3. FEED-FORWARD NETWORK & TRANSFORMATION BLOCK
# =====================================================================
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Standard transformer expansion layer (swaps out for SwiGLU in newer models)
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # Clean residual stream pipeline with pre-layer normalization
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# =====================================================================
# 4. MASTER MODEL CONTAINER
# =====================================================================
class PurePyTorchLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying (GPT style optimization)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is {self.config.block_size}"
        
        pos = torch.arange(0, t, dtype=torch.long, device=device) # Shape (t)

        # Forward pass tracking through the embeddings
        tok_emb = self.transformer.wte(idx) # Token embeddings shape (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos) # Position embeddings shape (T, n_embd)
        x = tok_emb + pos_emb
        
        # Cycle through every transformer hidden layer block
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # If targets are passed, compute evaluation cross-entropy loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        else:
            # Inference evaluation mode optimization
            logits = self.lm_head(x[:, [-1], :]) # Focus exclusively on the very last predicted token
            return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Generates raw text tokens autoregressively."""
        for _ in range(max_new_tokens):
            # Crop current context window if it overflows block size limits
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            # Apply temperature scaling to logits
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            # Sample next token based on probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# =====================================================================
# 5. EXECUTION & TEST DRIVE RUN
# =====================================================================
if __name__ == "__main__":
    # Choose execution engine environment (GPU hardware preferred if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device execution path: {device}")

    # Micro toy training dataset corpus
    data_text = "the quick brown fox jumps over the lazy dog. the lazy dog barked at the quick brown fox."
    chars = sorted(list(set(data_text)))
    vocab_size = len(chars)
    
    # Primitive character-level mapping tokenizer
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    # Convert text to tensor structure
    data_tensor = torch.tensor(encode(data_text), dtype=torch.long)
    
    # Configure custom overrides matching dataset size boundaries
    config = MiniLLMConfig()
    config.vocab_size = vocab_size
    config.block_size = 32  # Small context fit for toy dataset
    
    model = PurePyTorchLLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Super simple dummy batch creation helper
    def get_batch():
        ix = torch.randint(len(data_tensor) - config.block_size, (4,))
        x = torch.stack([data_tensor[i:i+config.block_size] for i in ix])
        y = torch.stack([data_tensor[i+1:i+config.block_size+1] for i in ix])
        return x.to(device), y.to(device)

    print("Training model initialized. Starting structural loop optimization steps...")
    model.train()
    for step in range(300): # Quick overfit training optimization cycles
        xb, yb = get_batch()
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"Step {step:03d} | Batch Optimization Loss Matrix Calculation: {loss.item():.4f}")

    # Run Generation Test
    model.eval()
    prompt = "the quick "
    context_tokens = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    generated_indices = model.generate(context_tokens, max_new_tokens=40, temperature=0.8)
    
    print("\n--- Model Output Generation Result ---")
    print(decode(generated_indices[0].tolist()))