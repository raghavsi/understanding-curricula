"""
Gradient statistics (norm, variance, alignment) for training loops.

Two ways to get the gradient matrix G (rows = gradients whose spread we measure):
  - microbatch_grads : split the batch into k equal chunks, one gradient per chunk.
                       Works with BatchNorm (chunks run in train mode, BN buffers restored).
                       Needs only k x P memory. Default for the training script.
  - per_sample_grads : one gradient per sample via torch.func.vmap. Exact, but needs a model
                       without train-mode BatchNorm and B x P memory.

Reported by gradient_stats(G, sample_size):
  grad_norm     : ||g_bar||, norm of the mean gradient
  trace_K       : tr(K), gradient variance. With chunks of size m the chunk-mean covariance is
                  K/m, so tr(K) is estimated as m * (unbiased chunk variance)
  snr           : ||g_bar||^2 / tr(K)
  fisher_trace  : Tr(F). Exact mean ||g_i||^2 for per-sample rows; for chunks it is approximated
                  by ||g_bar||^2 + tr(K)
  cos_to_mean   : mean cosine(g_i, g_bar)
  pairwise_cos  : mean cosine(g_i, g_j), i != j
                  (cosines are between rows, so with chunks they depend on the chunk size m)
GradTracker adds:
  cos_prev_step : cosine between this and the previous *measured* mean gradient
                  (consecutive steps only if you measure every step)
  cos_to_ref    : cosine with a fixed reference gradient (set_reference)
"""
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- gradients
def microbatch_grads(model, loss_fn, x, y, n_chunks):
    """Gradients of k equal-size chunks of the batch. Returns (G of shape (k, P), chunk_size m).

    Returns (None, m) if chunks would have fewer than 2 samples.
    Does not touch .grad, and restores BatchNorm running stats afterwards.
    """
    m = x.shape[0] // n_chunks
    if m < 2:
        return None, m
    was_training = model.training
    model.train()
    saved = {k: v.clone() for k, v in model.named_buffers()}
    params = [p for p in model.parameters() if p.requires_grad]
    rows = []
    for i in range(n_chunks):
        xs, ys = x[i * m:(i + 1) * m], y[i * m:(i + 1) * m]
        grads = torch.autograd.grad(loss_fn(model(xs), ys), params, allow_unused=True)
        rows.append(torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                               for g, p in zip(grads, params)]).float())
    with torch.no_grad():
        for k, v in model.named_buffers():
            v.copy_(saved[k])
    model.train(was_training)
    return torch.stack(rows), m


def per_sample_grads(model, loss_fn, x, y):
    """(B, P) per-sample gradients via torch.func. Use model.eval() if it has BatchNorm."""
    from torch.func import functional_call, vmap, grad
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def loss_single(p, b, xi, yi):
        out = functional_call(model, (p, b), (xi.unsqueeze(0),))
        return loss_fn(out, yi.unsqueeze(0))

    g = vmap(grad(loss_single), in_dims=(None, None, 0, 0))(params, buffers, x, y)
    return torch.cat([g[k].reshape(x.shape[0], -1) for k in params], dim=1).float()


# ---------------------------------------------------------------- statistics
@torch.no_grad()
def gradient_stats(G, sample_size=1):
    """G: (n, P). sample_size = number of samples averaged into each row (1 = per-sample)."""
    n = G.shape[0]
    assert n > 1, "need at least 2 rows"
    g_bar = G.mean(0)
    bar_sq = g_bar.pow(2).sum()

    trace_K = (G - g_bar).pow(2).sum() / (n - 1) * sample_size
    if sample_size == 1:
        fisher_trace = G.pow(2).sum(1).mean()
    else:
        fisher_trace = bar_sq + trace_K

    cos_to_mean = F.cosine_similarity(G, g_bar.unsqueeze(0), dim=1).mean()
    Gn = F.normalize(G, dim=1)
    S = Gn @ Gn.T
    pairwise_cos = (S.sum() - S.diagonal().sum()) / (n * (n - 1))

    return {
        "grad_norm": bar_sq.sqrt().item(),
        "trace_K": trace_K.item(),
        "snr": (bar_sq / trace_K).item(),
        "fisher_trace": fisher_trace.item(),
        "cos_to_mean": cos_to_mean.item(),
        "pairwise_cos": pairwise_cos.item(),
    }


class GradTracker:
    """Adds across-time alignment on top of gradient_stats."""

    def __init__(self):
        self.prev = None
        self.ref = None

    def set_reference(self, g_ref):
        self.ref = g_ref.detach().clone()

    @torch.no_grad()
    def update(self, G, sample_size=1):
        stats = gradient_stats(G, sample_size)
        g_bar = G.mean(0)
        if self.prev is not None:
            stats["cos_prev_step"] = F.cosine_similarity(g_bar, self.prev, dim=0).item()
        if self.ref is not None:
            stats["cos_to_ref"] = F.cosine_similarity(g_bar, self.ref, dim=0).item()
        self.prev = g_bar.clone()
        return stats
