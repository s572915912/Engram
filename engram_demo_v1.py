"""
================================================================================
[Engram Architecture Demo Implementation]

DISCLAIMER:
1. Demo Purpose Only: 
   This code is a demonstration version intended to illustrate the core logic and 
   data flow of the Engram module.

2. Production Readiness: 
   This implementation requires further optimization for actual production use 
   (e.g., custom CUDA kernels, distributed training support).

3. Simplifications: 
   Standard components (Normalization, Attention, MoE) and complex Hyper-connection 
   mechanisms are omitted or mocked in this version to focus exclusively on the 
   Engram module implementation.
================================================================================
"""

"""
pip install torch numpy transformers sympy
"""

## built-in
from typing import List
from dataclasses import dataclass, field
import math

## third-party
from sympy import isprime
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from tokenizers import normalizers, Regex 

@dataclass
class EngramConfig:
    tokenizer_name_or_path: str = "deepseek-ai/DeepSeek-V3"
    engram_vocab_size: List[int] = field(default_factory=lambda: [129280*5, 129280*5])
    max_ngram_size: int = 3
    n_embed_per_ngram: int = 512
    n_head_per_ngram: int = 8
    layer_ids: List[int] = field(default_factory=lambda: [1, 15])
    pad_id: int = 2
    seed: int = 0
    kernel_size: int = 4
    
@dataclass
class BackBoneConfig:
    hidden_size: int = 1024
    hc_mult: int = 4
    vocab_size: int = 129280
    num_layers: int = 30
    
@dataclass
class DemoConfig:
    text: str = "Only Alexander the Great could tame the horse Bucephalus."
    tokenizer_name_or_path: str = "deepseek-ai/DeepSeek-V3"
    use_tiny_config: bool = True
    token_preview: int = 12
    seed: int = 0


def print_section(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{title}\n{bar}")


def describe_tensor(name: str, tensor: torch.Tensor) -> None:
    print(f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype}")


def preview_tokens(tokenizer: AutoTokenizer, input_ids: torch.Tensor, limit: int) -> None:
    ids = input_ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)
    preview = list(zip(ids[:limit], tokens[:limit]))
    print(f"Token preview (id, token)[:{limit}]: {preview}")


def build_demo_configs(
    tokenizer: AutoTokenizer,
    demo_cfg: DemoConfig,
) -> tuple[EngramConfig, BackBoneConfig]:
    if demo_cfg.use_tiny_config:
        engram_cfg = EngramConfig(
            tokenizer_name_or_path=demo_cfg.tokenizer_name_or_path,
            engram_vocab_size=[101, 103],
            max_ngram_size=3,
            n_embed_per_ngram=64,
            n_head_per_ngram=2,
            layer_ids=[1],
            pad_id=2,
            seed=demo_cfg.seed,
            kernel_size=2,
        )
        backbone_cfg = BackBoneConfig(
            hidden_size=64,
            hc_mult=2,
            vocab_size=len(tokenizer),
            num_layers=4,
        )
        return engram_cfg, backbone_cfg

    engram_cfg = EngramConfig(
        tokenizer_name_or_path=demo_cfg.tokenizer_name_or_path,
        seed=demo_cfg.seed,
    )
    backbone_cfg = BackBoneConfig(vocab_size=len(tokenizer))
    return engram_cfg, backbone_cfg

class CompressedTokenizer:
    def __init__(
        self,
        tokenizer_name_or_path,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
        
        SENTINEL = "\uE000"
        self.normalizer = normalizers.Sequence([
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), SENTINEL),
            normalizers.Strip(),
            normalizers.Replace(SENTINEL, " "),
        ])
        
        self.lookup_table, self.num_new_token = self._build_lookup_table()
    
    def __len__(self):
        return self.num_new_token
    
    def _build_lookup_table(self):
        old2new = {}
        key2new = {}          
        new_tokens = []

        vocab_size = len(self.tokenizer)
        for tid in range(vocab_size):
            text = self.tokenizer.decode([tid], skip_special_tokens=False)
            
            if "�" in text:
                key = self.tokenizer.convert_ids_to_tokens(tid)
            else:
                norm = self.normalizer.normalize_str(text)
                key = norm if norm else text

            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)
            old2new[tid] = nid
        
        lookup = np.empty(vocab_size, dtype=np.int64)
        for tid in range(vocab_size):
            lookup[tid] = old2new[tid]

        return lookup, len(new_tokens)
    
    def _compress(self, input_ids):
        if isinstance(input_ids, torch.Tensor):
            arr = input_ids.detach().cpu().numpy().astype(np.int64, copy=False)
        else:
            arr = np.asarray(input_ids, dtype=np.int64)
        pos_mask = arr >= 0
        out = arr.copy()
        valid_ids = arr[pos_mask]
        out[pos_mask] = self.lookup_table[valid_ids]
        return out   
    
    def __call__(self, input_ids):
        return self._compress(input_ids)
            
class ShortConv(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        kernel_size: int = 4, 
        dilation: int = 1, 
        norm_eps: float = 1e-5,
        hc_mult: int = 4,
        activation: bool = True,
    ):
        super().__init__()
        self.hc_mult = hc_mult
        self.activation = activation
        
        total_channels = hidden_size * hc_mult
        self.conv = nn.Conv1d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=kernel_size,
            groups=total_channels,
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )

        self.norms = nn.ModuleList([
            nn.RMSNorm(hidden_size, eps=norm_eps) 
            for _ in range(hc_mult)
        ])
        
        if self.activation:
            self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B,L,HC_MULT,D)
        Output: (B,L,HC_MULT,D)
        """
        B, T, G, C = x.shape
        
        assert G == self.hc_mult, f"Input groups {G} != hc_mult {self.hc_mult}"

        normed_chunks = []
        for i in range(G):
            chunk = x[:, :, i, :]
            normed_chunks.append(self.norms[i](chunk))
        
        x_norm = torch.cat(normed_chunks, dim=-1)
        x_bct = x_norm.transpose(1, 2)
        y_bct = self.conv(x_bct)
        y_bct = y_bct[..., :T]

        if self.activation:
            y_bct = self.act_fn(y_bct)
        y = y_bct.transpose(1, 2).view(B, T, G, C).contiguous()
        
        return y
    
def find_next_prime(start, seen_primes):
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1

class NgramHashMapping:
    def __init__(
        self, 
        engram_vocab_size,
        max_ngram_size,
        n_embed_per_ngram,
        n_head_per_ngram,
        layer_ids,
        tokenizer_name_or_path,
        pad_id,
        seed,  
    ):
        self.vocab_size_per_ngram = engram_vocab_size
        self.max_ngram_size = max_ngram_size
        self.n_embed_per_ngram = n_embed_per_ngram
        self.n_head_per_ngram = n_head_per_ngram
        self.pad_id = pad_id
        self.layer_ids = layer_ids

        self.compressed_tokenizer = CompressedTokenizer(
            tokenizer_name_or_path=tokenizer_name_or_path
        )            
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)
        if self.pad_id is not None:
            self.pad_id = int(self.compressed_tokenizer.lookup_table[self.pad_id])

        max_long = np.iinfo(np.int64).max
        M_max = int(max_long // self.tokenizer_vocab_size)
        half_bound = max(1, M_max // 2)
        PRIME_1 = 10007
        
        self.layer_multipliers = {}

        for layer_id in self.layer_ids:
            base_seed = int(seed + PRIME_1 * int(layer_id))
            g = np.random.default_rng(base_seed)
            r = g.integers(
                low=0,
                high=half_bound,
                size=(self.max_ngram_size,),
                dtype=np.int64
            )
            multipliers = r * 2 + 1
            self.layer_multipliers[layer_id] = multipliers

        self.vocab_size_across_layers = self.calculate_vocab_size_across_layers()

    def calculate_vocab_size_across_layers(self):
        seen_primes = set()
        vocab_size_across_layers = {}
        
        for layer_id in self.layer_ids:
            all_ngram_vocab_sizes = []
            for ngram in range(2, self.max_ngram_size + 1):
                current_ngram_heads_sizes = []
                
                vocab_size = self.vocab_size_per_ngram[ngram - 2]
                num_head = self.n_head_per_ngram
                current_prime_search_start = vocab_size - 1
                
                for _ in range(num_head):
                    found_prime = find_next_prime(
                        current_prime_search_start, 
                        seen_primes
                    )
                    seen_primes.add(found_prime)
                    current_ngram_heads_sizes.append(found_prime)
                    current_prime_search_start = found_prime
                
                all_ngram_vocab_sizes.append(current_ngram_heads_sizes)
            vocab_size_across_layers[layer_id] = all_ngram_vocab_sizes
            
        return vocab_size_across_layers

    def _get_ngram_hashes(
        self,
        input_ids: np.ndarray,
        layer_id: int,
    ) -> np.ndarray:
        """
        Build hash buckets for n-grams.

        Output shape: (B, T, num_heads_across_all_ngrams)
        Example when max_ngram_size=3, n_head_per_ngram=2:
            2-gram -> 2 heads
            3-gram -> 2 heads
            total heads = 4
        """
        x = np.asarray(input_ids, dtype=np.int64)
        B, T = x.shape

        multipliers = self.layer_multipliers[layer_id]

        def shift_k(k: int) -> np.ndarray:
            if k == 0: return x
            shifted = np.pad(x, ((0, 0), (k, 0)),
                                mode='constant', constant_values=self.pad_id)[:, :T]
            return shifted

        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]

        all_hashes = []
        
        for n in range(2, self.max_ngram_size + 1):
            n_gram_index = n - 2
            tokens = base_shifts[:n]
            mix = (tokens[0] * multipliers[0])
            for k in range(1, n):
                mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])
            num_heads_for_this_ngram = self.n_head_per_ngram
            head_vocab_sizes = self.vocab_size_across_layers[layer_id][n_gram_index]
            
            for j in range(num_heads_for_this_ngram):
                mod = int(head_vocab_sizes[j])
                head_hash = mix % mod
                all_hashes.append(head_hash.astype(np.int64, copy=False))
        
        return np.stack(all_hashes, axis=2)

    def hash(self, input_ids):
        input_ids = self.compressed_tokenizer(input_ids)
        hash_ids_for_all_layers = {}
        for layer_id in self.layer_ids:
            hash_ids_for_all_layers[layer_id] = self._get_ngram_hashes(input_ids, layer_id=layer_id)
        return hash_ids_for_all_layers

class MultiHeadEmbedding(nn.Module):
    def __init__(self, list_of_N: List[int], D: int):
        super().__init__()
        self.num_heads = len(list_of_N)
        self.embedding_dim = D
        
        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)
        
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        
        total_N = sum(list_of_N)
        self.embedding = nn.Embedding(num_embeddings=total_N, embedding_dim=D)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        shifted_input_ids = input_ids + self.offsets
        output = self.embedding(shifted_input_ids)
        
        return output
    
class Engram(nn.Module):
    def __init__(self, layer_id, engram_cfg: EngramConfig, backbone_cfg: BackBoneConfig):
        super().__init__()
        self.layer_id = layer_id
        self.engram_cfg = engram_cfg
        self.backbone_cfg = backbone_cfg
        self.hash_mapping = NgramHashMapping(
            engram_vocab_size=engram_cfg.engram_vocab_size,
            max_ngram_size=engram_cfg.max_ngram_size,
            n_embed_per_ngram=engram_cfg.n_embed_per_ngram,
            n_head_per_ngram=engram_cfg.n_head_per_ngram,
            layer_ids=engram_cfg.layer_ids,
            tokenizer_name_or_path=engram_cfg.tokenizer_name_or_path,
            pad_id=engram_cfg.pad_id,
            seed=engram_cfg.seed,
        )
        self.multi_head_embedding = MultiHeadEmbedding(
            list_of_N = [x for y in self.hash_mapping.vocab_size_across_layers[self.layer_id] for x in y],
            D = engram_cfg.n_embed_per_ngram // engram_cfg.n_head_per_ngram,
        )
        self.short_conv = ShortConv(
            hidden_size=backbone_cfg.hidden_size,
            kernel_size=engram_cfg.kernel_size,
            dilation=engram_cfg.max_ngram_size,
            hc_mult=backbone_cfg.hc_mult,
        )
        engram_hidden_size = (engram_cfg.max_ngram_size - 1) * engram_cfg.n_embed_per_ngram
        self.value_proj = nn.Linear(engram_hidden_size, backbone_cfg.hidden_size)
        self.key_projs = nn.ModuleList(
            [nn.Linear(engram_hidden_size, backbone_cfg.hidden_size) for _ in range(backbone_cfg.hc_mult)]
        )
        self.norm1 = nn.ModuleList([nn.RMSNorm(backbone_cfg.hidden_size) for _ in range(backbone_cfg.hc_mult)])
        self.norm2 = nn.ModuleList([nn.RMSNorm(backbone_cfg.hidden_size) for _ in range(backbone_cfg.hc_mult)])
    
    def forward(self,hidden_states,input_ids):
        """
        hidden_states: [B, L, HC_MULT, D]
        input_ids: [B, L]
        """
        hash_input_ids = torch.from_numpy(self.hash_mapping.hash(input_ids)[self.layer_id])
        embeddings = self.multi_head_embedding(hash_input_ids).flatten(start_dim=-2)
        gates = []
        for hc_idx in range(self.backbone_cfg.hc_mult):
            key = self.key_projs[hc_idx](embeddings)
            normed_key = self.norm1[hc_idx](key)
            query = hidden_states[:,:,hc_idx,:]
            normed_query = self.norm2[hc_idx](query)
            gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(self.backbone_cfg.hidden_size)
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gate = gate.sigmoid().unsqueeze(-1)
            gates.append(gate)
        gates = torch.stack(gates,dim=2)
        value = gates * self.value_proj(embeddings).unsqueeze(2)
        output = value + self.short_conv(value)
        return output 

class TransformerBlock(nn.Module):
    def __init__(self, layer_id, engram_cfg: EngramConfig, backbone_cfg: BackBoneConfig):
        super().__init__()
        self.attn = lambda x:x
        self.moe  = lambda x:x
        self.engram = None
        if layer_id in engram_cfg.layer_ids:
            self.engram = Engram(
                layer_id=layer_id,
                engram_cfg=engram_cfg,
                backbone_cfg=backbone_cfg,
            )
    
    def forward(self,input_ids,hidden_states):
        if self.engram is not None:
            hidden_states = self.engram(hidden_states=hidden_states,input_ids=input_ids) + hidden_states
        hidden_states = self.attn(hidden_states) + hidden_states
        hidden_states = self.moe(hidden_states) + hidden_states
        return hidden_states

if __name__ == '__main__':
    demo_cfg = DemoConfig()
    torch.manual_seed(demo_cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        demo_cfg.tokenizer_name_or_path,
        trust_remote_code=True,
    )
    engram_cfg, backbone_cfg = build_demo_configs(tokenizer, demo_cfg)

    print_section("1) Tokenization")
    input_ids = tokenizer(demo_cfg.text, return_tensors="pt").input_ids
    describe_tensor("input_ids", input_ids)
    preview_tokens(tokenizer, input_ids, demo_cfg.token_preview)

    print_section("2) Compressed token ids")
    compressed_tokenizer = CompressedTokenizer(demo_cfg.tokenizer_name_or_path)
    compressed_ids = compressed_tokenizer(input_ids)
    print(f"Compressed vocab size: {len(compressed_tokenizer)}")
    print(f"Compressed ids preview: {compressed_ids[0][:demo_cfg.token_preview].tolist()}")

    print_section("3) N-gram hash buckets")
    hash_mapping = NgramHashMapping(
        engram_vocab_size=engram_cfg.engram_vocab_size,
        max_ngram_size=engram_cfg.max_ngram_size,
        n_embed_per_ngram=engram_cfg.n_embed_per_ngram,
        n_head_per_ngram=engram_cfg.n_head_per_ngram,
        layer_ids=engram_cfg.layer_ids,
        tokenizer_name_or_path=engram_cfg.tokenizer_name_or_path,
        pad_id=engram_cfg.pad_id,
        seed=engram_cfg.seed,
    )
    hash_ids = hash_mapping.hash(input_ids)[engram_cfg.layer_ids[0]]
    print(f"Hash ids shape: {hash_ids.shape} (B, T, total_heads)")

    print_section("4) Engram module forward")
    LLM = [
        nn.Embedding(backbone_cfg.vocab_size, backbone_cfg.hidden_size),
        *[
            TransformerBlock(
                layer_id=layer_id,
                engram_cfg=engram_cfg,
                backbone_cfg=backbone_cfg,
            )
            for layer_id in range(backbone_cfg.num_layers)
        ],
        nn.Linear(backbone_cfg.hidden_size, backbone_cfg.vocab_size),
    ]

    for idx, layer in enumerate(LLM):
        if idx == 0:
            hidden_states = LLM[0](input_ids)
            describe_tensor("embedding output", hidden_states)
            # mock hyper-connection (expand to HC_MULT groups)
            hidden_states = hidden_states.unsqueeze(2).expand(-1, -1, backbone_cfg.hc_mult, -1)
            describe_tensor("hyper-connection states", hidden_states)
        elif idx == len(LLM) - 1:
            # mock hyper-connection (collapse groups)
            hidden_states = hidden_states[:, :, 0, :]
            output = layer(hidden_states)
            describe_tensor("final logits", output)
        else:
            hidden_states = layer(input_ids=input_ids, hidden_states=hidden_states)

    print_section("5) Done")
    print("✅ Forward Complete!")
            
