# Flux.2 packed-latent resample phase-corruption probe (consultation round 1)
# Question A: does bilinear on the PACKED [B,128,h,w] tensor differ from
# bilinear in UNPACKED [B,32,2h,2w] space, and how badly?
#
# Pack/unpack layout is taken verbatim from ComfyUI latent_formats.py:233:
#   unpack: t.reshape(B,32,2,2,h,w).permute(0,1,4,2,5,3).reshape(B,32,2h,2w)
# so channel c in 0..127 -> (c32=c//4, dy=(c%4)//2, dx=c%2), pixel (2y+dy, 2x+dx).
import sys
import torch

sys.path.insert(0, r"C:\code\ComfyUI_experiment")
try:
    from comfy.utils import common_upscale
    SRC = "comfy.utils.common_upscale (real)"
except Exception as e:  # noqa: BLE001
    def common_upscale(samples, width, height, upscale_method, crop):
        return torch.nn.functional.interpolate(samples, size=(height, width), mode=upscale_method)
    SRC = f"fallback F.interpolate ({e})"

def unpack(t):
    B, _, h, w = t.shape
    return t.reshape(B, 32, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(B, 32, h * 2, w * 2)

def pack(t):
    B, _, H, W = t.shape
    return t.reshape(B, 32, H // 2, 2, W // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(B, 128, H // 2, W // 2)

def comb_y(u):
    """Deviation of odd rows from the mean of adjacent even rows (Nyquist comb detector)."""
    interp = 0.5 * (u[..., 0:-2:2, :] + u[..., 2::2, :])
    return (u[..., 1:-1:2, :] - interp).abs().mean().item()

def comb_x(u):
    interp = 0.5 * (u[..., :, 0:-2:2] + u[..., :, 2::2])
    return (u[..., :, 1:-1:2] - interp).abs().mean().item()

print(f"upscale source: {SRC}")
x = torch.randn(2, 128, 48, 84)
assert torch.equal(pack(unpack(x)), x), "pack/unpack roundtrip FAILED"
print("pack(unpack(x)) == x roundtrip: OK  (inverse of latent_formats.py:233 layout)")
print()

torch.manual_seed(0)
# Adreitz geometry: ref 1344x768 px -> packed latent [1,128,48,84] (h=48,w=84)
# canvas 4032x2304 px -> packed latent [1,128,144,252]; scale s=3 exactly.
h, w = 48, 84
for label, make in [
    ("smooth latent (bicubic-upsampled coarse noise)",
     lambda: torch.nn.functional.interpolate(torch.randn(1, 32, 12, 21), size=(2 * h, 2 * w),
                                             mode="bicubic", align_corners=False)),
    ("white-noise latent (worst case)",
     lambda: torch.randn(1, 32, 2 * h, 2 * w)),
]:
    vae = make()  # unpacked 32ch VAE-latent [1,32,96,168]
    packed = pack(vae)
    print(f"=== {label} ===")
    print(f"ref packed {tuple(packed.shape)}  unpacked {tuple(vae.shape)}")
    print(f"{'s':>4} | {'relRMS(P vs U)':>14} | {'comb_y U':>9} {'comb_y P':>9} {'ratio':>6} | "
          f"{'comb_x U':>9} {'comb_x P':>9} {'ratio':>6} | pred inter-plane shift (VAE px)")
    for s in (1.5, 2.0, 3.0):
        H, W = int(round(h * s)), int(round(w * s))
        # Path P = SHIPPED: bilinear on packed tensor, then unpack (tiled_diffusion.py:454)
        P = unpack(common_upscale(packed, W, H, "bilinear", "disabled"))
        # Path U = PROPOSED: unpack, bilinear at true 32ch/8x grid
        U = common_upscale(vae, 2 * W, 2 * H, "bilinear", "disabled")
        rel = ((P - U).pow(2).mean().sqrt() / U.pow(2).mean().sqrt()).item()
        cyU, cyP = comb_y(U), comb_y(P)
        cxU, cxP = comb_x(U), comb_x(P)
        print(f"{s:>4} | {rel:>14.4f} | {cyU:>9.4f} {cyP:>9.4f} {cyP / max(cyU, 1e-9):>6.2f} | "
              f"{cxU:>9.4f} {cxP:>9.4f} {cxP / max(cxU, 1e-9):>6.2f} | +/-{0.5 * (1 - 1 / s):.3f} "
              f"(spread {(1 - 1 / s):.3f} = {(1 - 1 / s) * 8:.1f} image px)")
    print()

# Sanity: fix path (unpack -> bilinear -> repack) sliced per tile equals slicing U then packing.
vae = torch.nn.functional.interpolate(torch.randn(1, 32, 12, 21), size=(96, 168), mode="bicubic", align_corners=False)
fix_packed = pack(common_upscale(vae, 2 * 252, 2 * 144, "bilinear", "disabled"))
print(f"fix path output shape: {tuple(fix_packed.shape)} (expect (1,128,144,252)) "
      f"-> canvas-res packed latent, sliceable by existing bbox code unchanged")
