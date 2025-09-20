# sampling.py
import torch
from llm_io import SPTokenizer

@torch.no_grad()
def reset_caches(model):
    for m in model.modules():
        if hasattr(m, "k_cache"):   m.k_cache.zero_()
        if hasattr(m, "v_cache"):   m.v_cache.zero_()
        if hasattr(m, "kv_cache"):  m.kv_cache.zero_()
        if hasattr(m, "pe_cache"):  m.pe_cache.zero_()

def sample_logits(logits, temperature=1.0, top_p=0.9, repetition_penalty=1.1, recent_ids=None):
    if recent_ids is not None and repetition_penalty != 1.0:
        # penaliza tokens recientes (simple)
        logits.scatter_(0, recent_ids, logits[recent_ids] / repetition_penalty)
    if temperature != 1.0:
        logits = logits / max(1e-6, temperature)
    probs = torch.softmax(logits.float(), dim=-1)

    # nucleus top-p
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cdf = torch.cumsum(sorted_probs, dim=-1)
    mask = cdf > top_p
    # mantiene al menos 1
    mask[..., 0] = False
    sorted_probs[mask] = 0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    idx = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_idx.gather(-1, idx).squeeze(-1)

@torch.inference_mode()
def generate(model, tok: SPTokenizer, prompt: str, max_new_tokens=200,
             temperature=0.8, top_p=0.9, repetition_penalty=1.1, stop_on_eos=True):
    device = "cuda"
    model.eval()
    reset_caches(model)

    ids = tok.encode(prompt, add_bos=True)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    # prefill todos los tokens del prompt
    start_pos = 0
    for t in range(x.size(1)):
        _ = model(x[:, t:t+1], start_pos=start_pos)
        start_pos += 1

    recent = torch.tensor([], dtype=torch.long, device=device)
    out_ids = []

    for _ in range(max_new_tokens):
        logits = model(x[:, -1:], start_pos=start_pos)[0]  # (V,)
        start_pos += 1
        next_id = sample_logits(
            logits, temperature=temperature, top_p=top_p,
            repetition_penalty=repetition_penalty, recent_ids=recent.unique()
        ).item()
        out_ids.append(next_id)
        # actualiza contexto rolling
        x = torch.tensor([[next_id]], dtype=torch.long, device=device)
        recent = torch.cat([recent, torch.tensor([next_id], device=device)])
        if stop_on_eos and tok.eos_id != -1 and next_id == tok.eos_id:
            break

    return tok.decode(out_ids)
