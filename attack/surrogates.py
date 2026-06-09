"""Heterogeneous CLIP surrogate encoders behind one differentiable interface,
plus a UCB bandit for X-Transfer's "surrogate scaling".

A Surrogate exposes  encode(x[B,3,224,224 in 0..1]) -> [B, d].  delta lives in the
victim's 224x224 [0,1] cover space; each surrogate differentiably resizes to its own
input resolution and applies its own (mean,std) before model.encode_image. The loss
is cosine similarity, which is scale-invariant, so we never L2-normalize here.

Memory: surrogates are loaded on CPU (fp32, good for autograd) and moved to GPU only
while selected (to_gpu / to_cpu), so the large search space never sits on GPU at once.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F

import config as C

_CANONICAL = 224  # matches clip_extract.preprocess_to_224 (the deployable cover space)

# OpenAI CLIP normalization constants (same as clip_extract._CLIP_MEAN/_STD)
_OPENAI_MEAN = (0.48145466, 0.4578275, 0.40821073)
_OPENAI_STD = (0.26862954, 0.26130258, 0.27577711)


def sid_of(entry):
    """Filesystem-safe surrogate id, e.g. open_clip__ViT-B-16__laion2b_s34b_b88k."""
    name = str(entry["name"]).replace("/", "-").replace("@", "_at_")
    return "%s__%s__%s" % (entry["backend"], name, str(entry.get("pretrained")))


def _meanstd_from_preprocess(preprocess):
    """Pull (mean, std) off a torchvision Compose by finding its Normalize transform."""
    for t in getattr(preprocess, "transforms", []):
        m, s = getattr(t, "mean", None), getattr(t, "std", None)
        if m is not None and s is not None:
            return tuple(float(x) for x in m), tuple(float(x) for x in s)
    return _OPENAI_MEAN, _OPENAI_STD


def _res_from_visual(model, fallback=_CANONICAL):
    v = getattr(model, "visual", None)
    r = getattr(v, "image_size", None)
    if r is None:
        r = getattr(v, "input_resolution", None)
    if isinstance(r, (tuple, list)):
        r = r[0]
    try:
        return int(r)
    except (TypeError, ValueError):
        return fallback


class Surrogate:
    """One CLIP image encoder; backend in {"open_clip", "openai_clip"}."""

    def __init__(self, entry, device=None):
        self.entry = entry
        self.backend = entry["backend"]
        self.name = entry["name"]
        self.pretrained = entry.get("pretrained")
        self.sid = sid_of(entry)
        self.device = device or C.DEVICE
        self.model = None
        self.input_res = _CANONICAL
        self.output_dim = None
        self._mean = None          # python tuples until on a device
        self._std = None
        self._mean_t = None        # (1,3,1,1) tensors on the active device
        self._std_t = None
        self._on_gpu = False

    # -- lifecycle (lazy; weights on CPU/fp32 for stable autograd) --
    def load(self):
        if self.model is not None:
            return self
        if self.backend == "open_clip":
            try:
                import open_clip
            except ImportError as e:
                raise RuntimeError(
                    "open_clip not installed for the X-Transfer surrogate pool. Either:\n"
                    "  (a) set XT_USE_OPEN_CLIP=False in attack/config.py to use the "
                    "OpenAI-clip-only pool (no new deps; works with the existing env), or\n"
                    "  (b) pip install --no-deps open_clip_torch==2.20.0  (keeps the "
                    "huggingface_hub==0.8.1 pin that transformers==4.17.0 requires; "
                    "see attack/README.md).") from e
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.name, pretrained=self.pretrained, device="cpu")
            self._mean, self._std = _meanstd_from_preprocess(preprocess)
            self.input_res = _res_from_visual(model)
            self.output_dim = getattr(getattr(model, "visual", None), "output_dim", None)
        elif self.backend == "openai_clip":
            try:
                import clip
            except ImportError as e:
                raise RuntimeError(
                    "OpenAI clip not installed; pip install "
                    "git+https://github.com/openai/CLIP.git (see attack/README.md).") from e
            model, _ = clip.load(self.name, device="cpu", jit=False)  # cpu => fp32
            self._mean, self._std = _OPENAI_MEAN, _OPENAI_STD
            self.input_res = _res_from_visual(model)
        else:
            raise ValueError("unknown surrogate backend: " + str(self.backend))
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        return self

    def to_gpu(self):
        self.load()
        if not self._on_gpu:
            self.model.to(self.device)
            self._mean_t = torch.tensor(self._mean, device=self.device).view(1, 3, 1, 1)
            self._std_t = torch.tensor(self._std, device=self.device).view(1, 3, 1, 1)
            self._on_gpu = True
        return self

    def to_cpu(self):
        if self.model is not None and self._on_gpu:
            self.model.to("cpu")
            self._mean_t = self._std_t = None
            self._on_gpu = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return self

    # -- differentiable forward --
    def encode(self, x_canonical):
        """x_canonical: (B,3,224,224) in [0,1] on self.device. Returns (B,d) float."""
        x = x_canonical
        if self.input_res != x.shape[-1]:
            x = F.interpolate(x, size=(self.input_res, self.input_res),
                              mode="bicubic", align_corners=False)
            x = x.clamp(0, 1)
        x = (x - self._mean_t) / self._std_t
        return self.model.encode_image(x).float()


def build_search_space(device=None):
    """Instantiate (lazy) Surrogate objects from config.xt_search_space().

    open_clip entries whose (model, pretrained) tag is absent in the INSTALLED
    open_clip are skipped with a warning (tag availability varies by version), so a
    stale tag never crashes the whole run. openai_clip entries are kept as-is."""
    entries = C.xt_search_space()
    if any(e.get("backend") == "open_clip" for e in entries):
        try:
            import open_clip
            avail = set(open_clip.list_pretrained())     # [(model_name, tag), ...]
            kept = []
            for e in entries:
                if e.get("backend") == "open_clip" and \
                        (e["name"], e.get("pretrained")) not in avail:
                    print("[surrogates][skip] %s/%s not in installed open_clip; skipping"
                          % (e["name"], e.get("pretrained")))
                    continue
                kept.append(e)
            entries = kept
        except ImportError:
            pass   # Surrogate.load() raises the friendly install message later
    if not entries:
        raise RuntimeError("no usable surrogates after filtering; check XT_SEARCH_SPACE "
                           "and the installed open_clip version")
    return [Surrogate(e, device=device) for e in entries]


class UCB:
    """Upper-Confidence-Bound bandit over N arms (surrogates).

    Reward = the per-encoder loss L_i (our minimized 1 - cos for the targeted objective).
    A HIGH reward means the attack is currently *failing* on that surrogate, so UCB
    up-weights it -> the perturbation is pushed to become universally effective. This is
    X-Transfer's "select the less-fooled encoders more often" (Eq. 6); no sign flip is
    needed in the targeted-minimization framing.
    """

    def __init__(self, n_arms, c=C.XT_UCB_C, momentum=C.XT_MOMENTUM):
        self.n = n_arms
        self.c = c
        self.m = momentum
        self.reward = np.zeros(n_arms, dtype="float64")
        self.counts = np.zeros(n_arms, dtype="float64")
        self.t = 0

    def select(self, k):
        k = min(k, self.n)
        unseen = [i for i in range(self.n) if self.counts[i] == 0]
        if len(unseen) >= k:                       # warmup: pull every arm once first
            return unseen[:k]
        scores = self.reward + self.c * np.sqrt(np.log(self.t + 1.0) / (self.counts + 1e-9))
        chosen = list(unseen)
        for i in np.argsort(-scores):
            if len(chosen) >= k:
                break
            if int(i) not in chosen:
                chosen.append(int(i))
        return chosen[:k]

    def update(self, idx, loss):
        if self.counts[idx] == 0:
            self.reward[idx] = loss
        else:
            self.reward[idx] = self.m * self.reward[idx] + (1.0 - self.m) * loss
        self.counts[idx] += 1
        self.t += 1

    def snapshot(self):
        return {"counts": self.counts.tolist(),
                "reward": [round(float(r), 5) for r in self.reward],
                "t": int(self.t)}
