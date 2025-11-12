import numpy as np
from scipy.stats import pearsonr, spearmanr, entropy
from scipy.optimize import linear_sum_assignment
import torch
#from metrics.utils import  _prepare_for_metric
EPS = 1e-12

def _to_float64_np(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().to("cpu")
     
        if x.dtype in (torch.bfloat16, torch.float16):
            x = x.to(torch.float32)
        x = x.numpy()
    return np.asarray(x, dtype=np.float64)

def _avg_heads_if_needed(M):
    if M.ndim == 4:
        # (B,H,Lq,Lk) or (1,H,Lq,Lk)
        M = M.mean(axis=0)  
        M = M.mean(axis=0)  
    elif M.ndim == 3:
        # (H, Lq, Lk)
        M = M.mean(axis=0)
 
    return M

def _crop_to_overlap(A, B):
    """
    Make A and B the same 2D shape by center-cropping (or left-top cropping) to the min dims.
    We’ll do left-top cropping to keep it simple and deterministic.
    """
    Ha, Wa = A.shape[-2], A.shape[-1]
    Hb, Wb = B.shape[-2], B.shape[-1]
    Hm, Wm = min(Ha, Hb), min(Wa, Wb)
    A2 = A[..., :Hm, :Wm]
    B2 = B[..., :Hm, :Wm]
    return A2, B2

def _prepare_for_metric(A, B):
    """
    Convert to 2D float64 numpy with identical shape.
    Returns (A2d, B2d) or (None, None) if cannot be aligned.
    """
    A = _to_float64_np(A)
    B = _to_float64_np(B)
    # squeeze trivial dims
    while A.ndim > 2 and A.shape[0] == 1:
        A = A.squeeze(0)
    while B.ndim > 2 and B.shape[0] == 1:
        B = B.squeeze(0)

  
    if A.ndim in (3,4): A = _avg_heads_if_needed(A)
    if B.ndim in (3,4): B = _avg_heads_if_needed(B)


    if A.ndim != 2 or B.ndim != 2:
        return None, None

  
    A2, B2 = _crop_to_overlap(A, B)
    if A2.size == 0 or B2.size == 0:
        return None, None
    return A2, B2


"""def _flatten(x):
    return np.asarray(x, dtype=np.float64).reshape(-1)"""
def _as_numpy_2d(x) -> np.ndarray:
 
    if isinstance(x, torch.Tensor):
        x = x.detach().to(device="cpu", dtype=torch.float64).contiguous().numpy()
    else:
        x = np.asarray(x, dtype=np.float64)


    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]
  
    if x.ndim == 3:
        x = x.mean(axis=0)
  
    if x.ndim == 1:
        n = x.size
        s = int(np.sqrt(n))
        if s * s == n:
            x = x.reshape(s, s)
        else:
            raise ValueError(f"Attention map is 1D of len {n} and cannot be reshaped to square.")

    if x.ndim != 2:
        raise ValueError(f"Expected 2D attention map, got shape {x.shape}")
    return x

def _flatten(x):
    if isinstance(x, torch.Tensor):
 
        return x.detach().to(dtype=torch.float64, device="cpu").contiguous().view(-1).numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1)

def _l1_normalize(x):
    x = _flatten(x)
    s = x.sum()
    if s <= 0:
       
        return np.ones_like(x) / x.size
    return x / s

def _softmax(x):
    x = _flatten(x)
    m = np.max(x)
    ex = np.exp(x - m)
    return ex / (ex.sum() + EPS)

def _centroid(attn_map):

    if isinstance(attn_map, torch.Tensor):
        M = attn_map.detach().to(dtype=torch.float64, device="cpu").numpy()
    else:
        M = np.asarray(attn_map, dtype=np.float64)
    #M = np.asarray(attn_map, dtype=np.float64)
    H, W = M.shape
    y = np.arange(H)[:, None]
    x = np.arange(W)[None, :]
    Z = M.sum()
    if Z <= 0:
        return (H / 2.0, W / 2.0)
    cy = (M * y).sum() / Z
    cx = (M * x).sum() / Z
    return (cy, cx)

def cosine_sim(A, B):
    a, b = _flatten(A), _flatten(B)
    na = np.linalg.norm(a) + EPS
    nb = np.linalg.norm(b) + EPS
    return float(np.dot(a, b) / (na * nb))

def pearson_sim(A, B):
    a, b = _flatten(A), _flatten(B)
    if np.std(a) < EPS or np.std(b) < EPS:
        return 0.0
    return float(pearsonr(a, b)[0])

def spearman_sim(A, B):
    a, b = _flatten(A), _flatten(B)
    return float(spearmanr(a, b).correlation)

def kl_divergence(A, B, normalize="l1"):
    if normalize == "l1":
        p, q = _l1_normalize(A), _l1_normalize(B)
    else:
        p, q = _softmax(A), _softmax(B)

    p_safe = np.clip(p, EPS, None)
    q_safe = np.clip(q, EPS, None)
    return float(np.sum(p_safe * (np.log(p_safe) - np.log(q_safe))))

def js_divergence(A, B, normalize="l1"):
    if normalize == "l1":
        p, q = _l1_normalize(A), _l1_normalize(B)
    else:
        p, q = _softmax(A), _softmax(B)
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))  

def iou_topk(A, B, topk=0.2):
    """
    IoU after thresholding to top-k proportion (e.g., top 20% mass).
    Works on flattened maps by selecting the highest-k fraction of values.
    """
    a = _flatten(A); b = _flatten(B)
    k = max(1, int(round(topk * len(a))))
    idx_a = np.argpartition(a, -k)[-k:]
    idx_b = np.argpartition(b, -k)[-k:]
    set_a, set_b = set(idx_a.tolist()), set(idx_b.tolist())
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return float(inter / (union + EPS))

def entropy_diff(A, B, normalize="l1"):
    if normalize == "l1":
        p, q = _l1_normalize(A), _l1_normalize(B)
    else:
        p, q = _softmax(A), _softmax(B)
    return float(abs(entropy(p) - entropy(q)))

def center_shift(A2d, B2d, normalize_by="diag"):
    assert A2d.ndim == 2 and B2d.ndim == 2, "Expect 2D maps"
    (cy1, cx1) = _centroid(A2d)
    (cy2, cx2) = _centroid(B2d)
    d = np.hypot(cy1 - cy2, cx1 - cx2)
    if normalize_by == "diag":
        H, W = A2d.shape
        d /= (np.hypot(H, W) + EPS)
    return float(d)





def compare_attention_runs(maps_A, maps_B):

    B_by_name = {x["name"]: x for x in maps_B}
    results = []
    for a in maps_A:
        name = a["name"]
        if name not in B_by_name: 
            continue
        ta = a["map"]  
        tb = B_by_name[name]["map"]

        pa, pb = _prepare_for_metric(ta, tb)
        #pa = ta #_prepare_dist(ta)
        #pb = tb #_prepare_dist(tb)
        #if name== "text.L0.full":
        #    print("*** pa ",pa)
        #    print("*** pb ",pb)
        #    print("kl ",kl_divergence(pa, pb))
        #print("name ",name)
        #print("kl ",kl_divergence(pa, pb))
        #print("cosine ",cosine_sim(pa, pb))
        #print("entropy_diff ",entropy_diff(pa,pb))
        res = {
            "layer": name,
            "shape_A": tuple(ta.shape),
            "shape_B": tuple(tb.shape),
            "kl_div": kl_divergence(pa, pb),
            #"cosine": cosine_sim(pa, pb),
            #"spearman": spearman_sim(pa, pb),
            "jl_div":js_divergence(pa,pb),
            "iou_topk": iou_topk(pa,pb),
            #"entropy_diff":entropy_diff(pa,pb),
            "center_shift":center_shift(pa,pb),
            "type": a.get("type", B_by_name[name].get("type", "unknown")),
            #"type": "vision" if (pa.numel() == (ta.shape[-1]*ta.shape[-2]) and ta.shape[-1]==ta.shape[-2]) else "textish",
        }
        #print("*** res ",res)
        results.append(res)
    return results

def print_results(results, top=10):
    import statistics as st

    """print("\n=== Per-layer metrics (first few) ===")
    for r in results[:top]:
        print(
            f"{r['layer']}: "
            f"KL={r['kl_div']:.4f} | "
            f"JS={r['jl_div']:.4f} | "
            f"Cos={r['cosine']:.4f} | "
            f"Spearman={r['spearman']:.4f} | "
            f"IoU@TopK={r['iou_topk']:.4f} | "
            f"ΔEntropy={r['entropy_diff']:.4f} | "
            f"CenterShift={r['center_shift']:.4f} "
            f"[{r['shape_A']} vs {r['shape_B']}]"
        )"""

   
    def avg(metric, kind=None):
        vals = [x[metric] for x in results if (kind is None or x["type"] == kind)]
        return st.mean(vals) if vals else float("nan")

    print("\n=== Summary (averages by component type) ===")
    for comp in ("vision", "text"):
        print(
            f"{comp:7s} | "
            f"KL={avg('kl_div', comp):.4f} | "
            f"JS={avg('jl_div', comp):.4f} | "
            #f"Cos={avg('cosine', comp):.4f} | "
            #f"Spearman={avg('spearman', comp):.4f} | "
            f"IoU@TopK={avg('iou_topk', comp):.4f} | "
            #f"ΔEntropy={avg('entropy_diff', comp):.4f} | "
            f"CenterShift={avg('center_shift', comp):.4f}"
        )

    print("\n=== Overall means ===")
    print(
        f"KL={avg('kl_div'):.4f} | JS={avg('jl_div'):.4f} | "
        #f"Cos={avg('cosine'):.4f} | Spearman={avg('spearman'):.4f} | "
        f"IoU@TopK={avg('iou_topk'):.4f} | "
        #f"ΔEntropy={avg('entropy_diff'):.4f} | CenterShift={avg('center_shift'):.4f}"
        f"CenterShift={avg('center_shift'):.4f}"
    )


def cross_attention_cosine(A, B):
    """
    A, B: shape [T_text, T_img] or [heads, T_text, T_img]
    Flattens everything and computes cosine similarity.
    """
    a = _flatten(A); b = _flatten(B)
    na = np.linalg.norm(a) + EPS
    nb = np.linalg.norm(b) + EPS
    return float(np.dot(a, b) / (na * nb))

def cross_attention_jsd(A, B):
    a = _l1_normalize(A); b = _l1_normalize(B)
    m = 0.5 * (a + b)
    return 0.5 * (entropy(a, m) + entropy(b, m))

def cross_attention_rank_corr(A, B, axis="img"):
    """
    Spearman rank correlation of attention over visual tokens or text tokens.
    axis='img' compares distributions over image tokens after averaging over text tokens.
    axis='text' does the opposite.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim == 3:  
        A = A.mean(0)
        B = B.mean(0)
    if axis == "img":
        a = A.mean(0) 
        b = B.mean(0)
    else:
        a = A.mean(1)
        b = B.mean(1)
    return float(spearmanr(a, b).correlation)


def token_alignment_consistency(text_emb_orig, text_emb_para, attn_orig, attn_para, sim="cosine"):
    """
    Align tokens between two questions by semantic similarity (Hungarian matching),
    then compute correlation between the aligned attention weights.

    text_emb_*: [T, D] token embeddings
    attn_*:    [T] attention weights for those tokens (e.g., self-attn importance)
    Returns: dict with keys { "mean_pair_sim", "weight_pearson", "weight_spearman" }
    """
    E1 = np.asarray(text_emb_orig, dtype=np.float64)
    E2 = np.asarray(text_emb_para, dtype=np.float64)
    w1 = np.asarray(attn_orig, dtype=np.float64).reshape(-1)
    w2 = np.asarray(attn_para, dtype=np.float64).reshape(-1)
    T1, D1 = E1.shape
    T2, D2 = E2.shape
    assert D1 == D2, "Embedding dims must match"

    # cosine similarity matrix (T1 x T2)
    E1n = E1 / (np.linalg.norm(E1, axis=1, keepdims=True) + EPS)
    E2n = E2 / (np.linalg.norm(E2, axis=1, keepdims=True) + EPS)
    S = E1n @ E2n.T  # in [-1,1], higher is better

    # Hungarian to maximize total similarity -> minimize cost = -S
    row_ind, col_ind = linear_sum_assignment(-S)
    pair_sims = S[row_ind, col_ind]

  
    w1_aligned = w1[row_ind]
    w2_aligned = w2[col_ind]

    w1n = _l1_normalize(w1_aligned)
    w2n = _l1_normalize(w2_aligned)

    wp = 0.0
    ws = 0.0
    if np.std(w1n) > EPS and np.std(w2n) > EPS:
        wp = float(pearsonr(w1n, w2n)[0])
        ws = float(spearmanr(w1n, w2n).correlation)

    return {
        "mean_pair_sim": float(pair_sims.mean()),
        "weight_pearson": wp,
        "weight_spearman": ws,
        "match_idx_q1": row_ind,
        "match_idx_q2": col_ind,
    }


def mean_attention_similarity(similarities):
    sims = np.asarray(list(similarities), dtype=np.float64)
    return float(np.nanmean(sims))

def consistency_index(similarities, threshold):
    sims = np.asarray(list(similarities), dtype=np.float64)
    return float(np.mean(sims >= threshold))