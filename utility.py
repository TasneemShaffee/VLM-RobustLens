from collections import defaultdict
import json
from dataset import *
import os
from src import *
from metrics import *
import json
import os, json, math, statistics as stats
from collections import defaultdict

"""
def set_text_layers_eager(model,model_name=None):
    model_layers= None
    if model_name=="internvl":
      model_layers=model.language_model.model.layers
    else: model_layers= model.model.language_model.layers  
    #for layer in model.model.language_model.layers:
    for layer in model_layers:      
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None) or getattr(layer, "attention", None)
        if hasattr(attn, "config"):
            attn.config._attn_implementation = "eager"
    # safer for returning weights
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
def attach_attention_hooks(runner,model_name=None):
    hooks = []
    model_layers=None
    vision_layers=None
    if model_name=="internvl":
      model_layers=runner.model.language_model.model.layers
      vision_layers=runner.model.vision_model.encoder.layers
    else: 
        model_layers= runner.model.model.language_model.layers
        vision_layers=runner.model.model.visual.blocks
    
    # Vision (recompute weights from Q/K)
    for blk in vision_layers:
        hooks.append(blk.attn.qkv.register_forward_hook(_qkv_hook))
        hooks.append(blk.attn.register_forward_hook(_vision_attn_hook, with_kwargs=True))
    # Text (eager returns weights directly)
    #for layer in runner.model.model.language_model.layers:
    for layer in model_layers:    
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None) or getattr(layer, "attention", None)
        hooks.append(attn.register_forward_hook(_text_attn_hook, with_kwargs=True))
    return hooks
"""
# _METRIC_KEYS = ["kl_div", "jl_div", "cosine", "spearman",
#                "iou_topk", "entropy_diff", "center_shift"]

_METRIC_KEYS = ["kl_div", "jl_div", "iou_topk", "center_shift"]


def _atomic_write_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def set_text_layers_eager(model, model_name=None):
    name = (model_name or "").lower()

    # Locate text layers
    if name == "internvl":
        # InternVL: language_model.model.layers
        # lm = getattr(model, "language_model", None)
        # layers = getattr(getattr(lm, "model", lm), "layers", None) if lm is not None else None
        layers = model.language_model.model.layers
    else:

        # base = getattr(model, "model", None)
        # lm = getattr(base, "language_model", None) if base is not None else None
        # layers = getattr(lm, "layers", None) if lm is not None else None
        layers = model.model.language_model.layers
    if not layers:
        raise RuntimeError("Could not find text layers for attn eager setup.")

    for layer in layers:
        attn = (
            getattr(layer, "self_attn", None)
            or getattr(layer, "attn", None)
            or getattr(layer, "attention", None)
        )
        if hasattr(attn, "config"):
            # print("Setting text attention to eager mode...")
            attn.config._attn_implementation = "eager"  # type: ignore

    # Returning attention weights is safer with cache off
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    # print("layers set to eager.",layers)


def attach_attention_hooks(runner, model_name=None):
    name = (model_name or "").lower()
    hooks = []

    # -------- TEXT LAYERS (both families) --------
    if name == "internvl":
        # lm = getattr(runner.model, "language_model", None)
        # text_layers = getattr(getattr(lm, "model", lm), "layers", None) if lm is not None else None
        text_layers = runner.model.language_model.model.layers

    else:
        # lm = getattr(getattr(runner.model, "model", None), "language_model", None)
        # text_layers = getattr(lm, "layers", None) if lm is not None else None
        text_layers = runner.model.model.language_model.layers
    if not text_layers:
        raise RuntimeError("Could not find text layers to attach hooks.")
    # print("Attaching text attention hooks...",text_layers)
    for layer in text_layers:
        t_attn = (
            getattr(layer, "self_attn", None)
            or getattr(layer, "attn", None)
            or getattr(layer, "attention", None)
        )
        if t_attn is not None:
            if name == "internvl":
                hooks.append(
                    t_attn.register_forward_hook(
                        _internvl_text_attn_hook, with_kwargs=True
                    )
                )
            else:
                hooks.append(
                    t_attn.register_forward_hook(_text_attn_hook, with_kwargs=True)
                )

    # -------- VISION LAYERS --------
    if name == "internvl":

        # print("Attaching InternVL ...")
        def _iter_internvl_vision_layers(model):
            # vision_model path
            vm = getattr(model, "vision_model", None)
            if vm is not None and hasattr(vm, "encoder"):
                for blk in getattr(vm.encoder, "layer", []):
                    yield blk
            # vision_tower path
            vt = getattr(model, "vision_tower", None)
            if vt is not None and hasattr(vt, "encoder"):
                for blk in getattr(vt.encoder, "layer", []):
                    yield blk

        # print("Setting InternVL vision attention to eager mode...")
        for blk in runner.model.vision_model.encoder.layers:
            # print("Setting InternVL vision attention to eager mode...")
            # v_attn = getattr(blk, "attention", None)
            attn = getattr(blk, "attn", None) or getattr(lyr, "attention", None)
            if attn is None:
                continue
            if hasattr(attn, "use_flash_attn"):
                attn.use_flash_attn = False
                # print("Disabled flash attention.")

            if hasattr(attn, "config"):
                # print("eager")
                attn.config._attn_implementation = "eager"

        for blk in runner.model.vision_model.encoder.layers:
            v_attn = getattr(blk, "attention", None) or getattr(blk, "attn", None)
            if v_attn is not None:
                # print("_internvl_vision_attn_hook attached.")
                hooks.append(
                    v_attn.register_forward_hook(
                        _internvl35_vision_attn_hook, with_kwargs=True
                    )
                )

    elif name == "gemma3":

        #   runner.model.model.vision_tower.encoder.layers (ModuleList of SiglipEncoderLayer)
        v_layers = runner.model.model.vision_tower.vision_model.encoder.layers
        if not v_layers:
            raise RuntimeError("Could not find Gemma-3 vision encoder layers.")
        for blk in v_layers:

            v_attn = (
                getattr(blk, "self_attn", None)
                or getattr(blk, "attn", None)
                or getattr(blk, "attention", None)
            )
            if v_attn is None:
                continue

            if hasattr(v_attn, "config"):
                # print("Setting Gemma-3 vision attention to eager mode...")
                v_attn.config._attn_implementation = "eager"
            hooks.append(
                v_attn.register_forward_hook(_gemma3_vision_attn_hook, with_kwargs=True)
            )

    else:

        # visual = getattr(getattr(runner.model, "model", None), "visual", None)
        blocks = (
            runner.model.model.visual.blocks
        )  # getattr(visual, "blocks", None) if visual is not None else None
        if not blocks:
            raise RuntimeError("Could not find Qwen/Gemma vision blocks.")
        for blk in blocks:

            if hasattr(blk, "attn") and hasattr(blk.attn, "qkv"):
                hooks.append(blk.attn.qkv.register_forward_hook(_qkv_hook))
                hooks.append(
                    blk.attn.register_forward_hook(_vision_attn_hook, with_kwargs=True)
                )
            else:

                pass

    return hooks


def detach_hooks(hooks):
    for h in hooks:
        h.remove()


def _mean_or_nan(vals):
    vals = [
        v
        for v in vals
        if v is not None
        and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
    ]
    return float(stats.mean(vals)) if vals else float("nan")


def _agg_rows(rows):
    out = {}
    for k in _METRIC_KEYS:
        out[k] = _mean_or_nan([r.get(k) for r in rows])
    return out


def _group_by_type(rows):
    by_type = defaultdict(list)
    for r in rows:
        # r["type"] expected in {"vision","text","text↔text","text→vision","vision→text","textish", ...}
        by_type[r.get("type", "unknown")].append(r)
    return {t: _agg_rows(rr) for t, rr in by_type.items()}


from PIL import Image


def _image_exists(img_path: str) -> bool:
    if not isinstance(img_path, str) or img_path.startswith(("http://", "https://")):
        return True
    if not os.path.isfile(img_path):
        return False
    try:
        with Image.open(img_path) as im:
            im.verify()
        return True
    except Exception:
        return False
