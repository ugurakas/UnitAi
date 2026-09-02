import torch
import torch.nn as nn
import math

# ==========================================
# 1. MODEL HİPERPARAMETRELERİ (KENDİNİZE GÖRE AYARLAYIN)
# ==========================================
class UnitAi-Config:
    def __init__(self):
        self.vocab_size = 100352      # Kelime dağarcığı boyutu (Orijinal Phi-4 standardı)
        self.hidden_size = 512        # Modelin anlamsal vektör genişliği (Örn: 256, 512, 1024, 5120)
        self.num_layers = 6           # Üst üste binecek toplam derin katman sayısı (Örn: 4, 6, 12, 40)
        self.num_heads = 8            # Query (Soru) dikkat kafa sayısı (Hidden size'a tam bölünmelidir)
        self.num_kv_heads = 2         # Key-Value (Hafıza) kafa sayısı (GQA için num_heads'den küçük olmalı)
        self.intermediate_size = 1536 # MLP katmanının genişliği (Genelde hidden_size * 3 veya * 4 yapılır)
        self.max_position_embeddings = 2048 # Modelin bir seferde okuyabileceği maksimum kelime sayısı
        self.rms_norm_eps = 1e-5      # Matematiksel kararlılık için küçük bir sayı

# ==========================================
# 2. RMS NORM (Gelişmiş Katman Normalizasyonu)
# ==========================================
class UnitAi-RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        # Bu ağırlıklar model eğitildikçe optimize olan parametrelerdir
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

# ==========================================
# 3. RoPE (Rotary Position Embedding - Konum Algısı)
# ==========================================
class UnitAi-RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())

    def forward(self, x, seq_len):
        return self.cos[:seq_len, :], self.sin[:seq_len, :]

def apply_rope(x, cos, sin):
    d = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated * sin)

# ==========================================
# 4. GROUPED-QUERY ATTENTION (GQA - Dikkat Mekanizması)
# ==========================================
class UnitAi-Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.num_heads
        
        # Öğrenilebilir Parametre Matrisleri
        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.rotary_emb = UnitAi-RotaryEmbedding(self.head_dim)

    def forward(self, x):
        B, S, C = x.shape
        q = self.q_proj(x).view(B, S, self.config.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.config.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.config.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(q, S)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # GQA için Key ve Value kafalarını Query kafa sayısına eşleme
        num_queries_per_kv = self.config.num_heads // self.config.num_kv_heads
        k = k.repeat_interleave(num_queries_per_kv, dim=1)
        v = v.repeat_interleave(num_queries_per_kv, dim=1)

        # Kelimelerin birbiriyle ilişkisini hesaplama (Matris Çarpımı)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Causal Mask (Modelin gelecekteki kelimeleri kopya çekmesini engeller)
        mask = torch.full((S, S), float("-inf"), device=x.device).triu(1)
        scores = scores + mask

        attention_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, v)
        
        context = context.transpose(1, 2).contiguous().view(B, S, C)
        return self.o_proj(context)

# ==========================================
# 5. Phi-4 MLP (SwiGLU Aktivasyonlu Mantık Katmanı)
# ==========================================
class UnitAi-MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_up_proj = nn.Linear(config.hidden_size, config.intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        # SwiGLU aktivasyonu: Orijinal Phi-4 ve Llama mimarilerinin temel taşıdır
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(torch.nn.functional.silu(gate) * up)

# ==========================================
# 6. DECODER LAYER (Katman Birleşimi)
# ==========================================
class UnitAi-DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = UnitAi-RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = UnitAi-Attention(config)
        self.post_attention_layernorm = UnitAi-RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = UnitAi-MLP(config)

    def forward(self, x):
        # Residual bağlantılar ile gradyan akışını koruma
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

# ==========================================
# 7. ANA MODEL (Giriş ve Çıkış Yönetimi)
# ==========================================
class CustomUnitAi-Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([UnitAi-DecoderLayer(config) for _ in range(config.num_layers)])
        self.norm = UnitAi-RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, tokens):
        x = self.embed_tokens(tokens)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)

# ==========================================
# TEST VE KONTROL ALANI
# ==========================================
config = UnitAi-Config()
model = CustomUnitAi-Model(config)

# Ayarladığınız hiperparametrelere göre oluşan toplam parametre hesabı
toplam_parametre = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Kendi Ayarladığınız Modelin Toplam Parametre Sayısı: {toplam_parametre:,}")

# Test Girdisi (Örnek: 4 kelimelik bir cümle token ID'leri)
test_input = torch.tensor([[101, 4582, 2013, 999]]) # Batch size = 1, Sequence length = 4
output_logits = model(test_input)

print("Çıktı Başarıyla Üretildi! Matris Boyutu (Batch, Kelime, Vocab Size):", output_logits.shape)
