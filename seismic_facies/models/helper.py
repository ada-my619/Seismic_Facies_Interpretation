import torch
import torch.nn.functional as F

def pad_to_match(src, ref):
    """
    Make src match ref spatially by center-cropping if too big, or symmetric padding if too small.
    src, ref: (B,C,H,W)
    """
    _, _, Hs, Ws = src.shape
    _, _, Hr, Wr = ref.shape

    # center-crop if src is bigger
    if Hs > Hr:
        top = (Hs - Hr) // 2
        src = src[:, :, top:top + Hr, :]
    if Ws > Wr:
        left = (Ws - Wr) // 2
        src = src[:, :, :, left:left + Wr]

    # symmetric pad if src is smaller
    _, _, Hs, Ws = src.shape
    dh, dw = Hr - Hs, Wr - Ws
    if dh > 0 or dw > 0:
        pad_top = dh // 2
        pad_bottom = dh - pad_top
        pad_left = dw // 2
        pad_right = dw - pad_left
        src = F.pad(src, (pad_left, pad_right, pad_top, pad_bottom))
    return src