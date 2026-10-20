def get_grad_vector(model):
    """Concatenate all parameter gradients into one vector."""
    grads = []
    for p in model.parameters():
        #print('a',p.grad.shape)
        if p.grad is not None:
            grads.append(p.grad.view(-1))
    return torch.cat(grads)
def hutchinson_trace(model, optimizer, loss_fn, X, y, num_samples=1):
    """Estimate trace(Fisher) using Hutchinson's estimator."""
    # Compute log-probabilities
    output = model(X)
    log_probs = F.log_softmax(output, dim=1)

    trace_estimates = []
    for _ in range(num_samples):
        # Draw random ±1 vector z same shape as params
        z = [torch.randint(0, 2, p.shape, device="cuda", dtype=torch.float32) * 2 - 1
             for p in model.parameters() if p.requires_grad]
        #optimizer.zero_grad()

        # Compute vector-Jacobian product (vjp)
        # grad(log p)·z  →  stochastic trace estimate
        total = 0.0
        for i in range(X.size(0)):
            logp = log_probs[i, y[i]]
            grads = torch.autograd.grad(logp, [p for p in model.parameters() if p.requires_grad],
                                        retain_graph=True, create_graph=True)
            total += sum((g*z_i).sum() for g, z_i in zip(grads, z))**2
        trace_estimates.append(total.item() / X.size(0))
    return sum(trace_estimates) / len(trace_estimates)
