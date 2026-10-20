import numpy as np

def ema(x, alpha=0.9):
    y = np.zeros_like(x, dtype=np.float64)
    y[0] = x[0]

    for i in range(1, len(x)):
        y[i] = alpha * y[i-1] + (1-alpha) * x[i]

    return y


def convergence_signal(train_loss, val_loss, alpha=0.9, eps=1e-8):
    """
    Returns:
        r > 0 : relative divergence
        r < 0 : relative convergence
        r ~ 0 : neither
    """

    train_s = ema(np.asarray(train_loss), alpha)
    val_s   = ema(np.asarray(val_loss), alpha)

    # log losses
    log_train = np.log(train_s + eps)
    log_val   = np.log(val_s + eps)

    # finite difference
    d_log_train = np.gradient(log_train)
    d_log_val   = np.gradient(log_val)

    r = d_log_val - d_log_train

    return r, train_s, val_s
def classify_dynamics(r, window=5, threshold=0.001):

    score = np.zeros_like(r)

    for t in range(window, len(r)):
        score[t] = np.mean(r[t-window:t])

    state = np.zeros_like(r, dtype=int)

    # +1 = divergence
    # -1 = convergence
    #  0 = stable
    state[score > threshold] = 1
    state[score < -threshold] = -1

    return state, score
def find_settling_time(
    r,
    window=5,
    mean_threshold=0.005,
    std_threshold=0.005,
    min_warmup=5
):

    for t in range(max(window, min_warmup), len(r)):

        w = r[t-window:t]

        mean_r = np.mean(w)
        std_r = np.std(w)

        if (
            abs(mean_r) < mean_threshold
            and std_r < std_threshold
        ):
            return t

    return None