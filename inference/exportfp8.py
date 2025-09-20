# export_fp8.py
import torch
from core_model import ModelArgs, Transformer, block_size

def quantize_blockwise_to_fp8(W: torch.Tensor, out_blocks: int, in_blocks: int):
    out_dim, in_dim = W.shape
    W_fp8 = torch.empty((out_dim, in_dim), dtype=torch.float8_e4m3fn, device=W.device)
    S = torch.empty((out_blocks, in_blocks), dtype=torch.float32, device=W.device)
    for ob in range(out_blocks):
        o0 = ob*block_size; o1 = min(o0+block_size, out_dim)
        for ib in range(in_blocks):
            i0 = ib*block_size; i1 = min(i0+block_size, in_dim)
            blk = W[o0:o1, i0:i1]
            s = max(blk.abs().max().item(), 1e-8)
            S[ob, ib] = s
            W_fp8[o0:o1, i0:i1] = (blk / s).to(dtype=torch.float8_e4m3fn)
    return W_fp8, S

def export_bf16_to_fp8(bf16_ckpt: str, out_fp8_ckpt: str):
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")

    # fuente BF16
    args_bf16 = ModelArgs(dtype="bf16")
    src = Transformer(args_bf16).cuda()
    sd = torch.load(bf16_ckpt, map_location="cuda")
    src.load_state_dict(sd["model"] if "model" in sd else sd, strict=False)

    # destino FP8 (crea .scale)
    args_fp8 = ModelArgs(dtype="fp8")
    dst = Transformer(args_fp8).cuda()

    src_sd = dict(src.named_parameters())
    dst_sd = dict(dst.named_parameters())
    with torch.no_grad():
        for name, p_dst in dst_sd.items():
            if "scale" in name:  # se setean aparte
                continue
            if name not in src_sd:
                continue
            p_src = src_sd[name]
            if p_dst.element_size() == 1 and p_src.element_size() > 1:
                # cuantiza lineales
                out_dim, in_dim = p_src.shape
                ob = (out_dim + block_size - 1)//block_size
                ib = (in_dim  + block_size - 1)//block_size
                W_fp8, S = quantize_blockwise_to_fp8(p_src.data, ob, ib)
                p_dst.copy_(W_fp8)
                scale_name = name.replace("weight", "scale")
                dst.state_dict()[scale_name].copy_(S)
            else:
                p_dst.copy_(p_src.data.to(dtype=p_dst.dtype))

    torch.save(dst.state_dict(), out_fp8_ckpt)
    print("Exportado FP8:", out_fp8_ckpt)

if __name__ == "__main__":
    export_bf16_to_fp8("pretrain_step_010000.pt", "pretrain_step_010000_fp8.pt")
