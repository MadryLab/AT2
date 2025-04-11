import torch as ch


def aggregate_logit_probs(logit_probs):
    """Compute sequence-level logit-probabilities from token-level logit-probabilities."""
    logit_probs = ch.tensor(logit_probs)
    log_probs = ch.nn.functional.logsigmoid(logit_probs).sum(dim=1)
    log_1mprobs = ch.zeros_like(log_probs)
    large_mask = log_probs > -1e-1
    log_1mprobs[large_mask] = ch.log(-ch.expm1(log_probs[large_mask]))
    log_1mprobs[~large_mask] = ch.log1p(-ch.exp(log_probs[~large_mask]))
    return (log_probs - log_1mprobs).numpy()
