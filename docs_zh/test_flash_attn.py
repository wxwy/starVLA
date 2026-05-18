import torch

print("=" * 60)
print("PyTorch 信息")
print("=" * 60)
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
    print("compute capability:", torch.cuda.get_device_capability(0))

print()

# ===== 检查 flash-attn 是否可导入 =====
print("=" * 60)
print("FlashAttention 导入检测")
print("=" * 60)

try:
    import flash_attn
    print("flash_attn import: SUCCESS")
    print("flash_attn version:", flash_attn.__version__)
except Exception as e:
    print("flash_attn import: FAILED")
    print(e)
    exit(1)

print()

# ===== 检查 CUDA kernel 是否可调用 =====
print("=" * 60)
print("FlashAttention CUDA Kernel 检测")
print("=" * 60)

try:
    from flash_attn import flash_attn_func

    device = "cuda"

    # batch, seq, heads, head_dim
    B = 2
    S = 128
    H = 8
    D = 64

    q = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
    k = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
    v = torch.randn(B, S, H, D, device=device, dtype=torch.float16)

    out = flash_attn_func(q, k, v)

    print("flash_attn_func: SUCCESS")
    print("output shape:", out.shape)
    print("output dtype:", out.dtype)

except Exception as e:
    print("flash_attn_func: FAILED")
    print(type(e).__name__, e)
    exit(1)

print()

# ===== 性能对比：FlashAttention vs 原生 SDPA =====
print("=" * 60)
print("FlashAttention 性能对比")
print("=" * 60)

try:
    import time
    import torch.nn.functional as F

    # 查看 PyTorch 默认 SDPA 后端
    backends = {
        "flash_sdp": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math": torch.backends.cuda.math_sdp_enabled(),
    }
    print("SDPA backends:", backends)

    B, S, H, D = 4, 1024, 16, 64
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)

    # ----- PyTorch 原生 SDPA -----
    for _ in range(10):
        F.scaled_dot_product_attention(q, k, v)

    torch.cuda.synchronize()
    start = time.time()

    for _ in range(50):
        F.scaled_dot_product_attention(q, k, v)

    torch.cuda.synchronize()
    sdpa_time = time.time() - start
    print(f"PyTorch SDPA  : {sdpa_time:.4f} sec (avg {sdpa_time / 50 * 1000:.3f} ms/iter)")

    # ----- FlashAttention -----
    for _ in range(10):
        flash_attn_func(q, k, v)

    torch.cuda.synchronize()
    start = time.time()

    for _ in range(50):
        flash_attn_func(q, k, v)

    torch.cuda.synchronize()
    fa_time = time.time() - start
    print(f"flash_attn    : {fa_time:.4f} sec (avg {fa_time / 50 * 1000:.3f} ms/iter)")

    # 加速比
    speedup = sdpa_time / fa_time
    print(f"加速比        : {speedup:.2f}x {'(flash_attn 更快)' if speedup > 1 else '(SDPA 更快)'}")

except Exception as e:
    print("benchmark FAILED")
    print(e)

print()
print("=" * 60)
print("FlashAttention 安装正常")
print("=" * 60)