"""Token-exact PPO clipping for the explicitly declared temperature policy."""
import torch
from torch.utils.checkpoint import checkpoint


def selected_logps(model, rec, indices):
    from layer_replay import replay_logps
    indices = list(indices)
    count = len(rec["output_ids"])
    if not indices or any(type(i) is not int or i < 0 or i >= count
                          for i in indices):
        raise ValueError("invalid selected token indices")
    # Preserve the complete prefix graph; select only the requested loss tokens.
    values = replay_logps(model, rec, checkpoint_layers=True)
    positions = torch.tensor(indices, device=values.device, dtype=torch.long)
    return values.index_select(0, positions)



def clipped_loss(current, old, advantage, epsilon=0.2):
    if current.shape != old.shape or current.numel() == 0:
        raise ValueError('token probability mismatch')
    if not torch.isfinite(current).all() or not torch.isfinite(old).all():
        raise ValueError('nonfinite token log probability')
    delta = current.float()-old.float()
    # Stop instead of concealing explosive drift behind an arbitrary clamp.
    if delta.detach().abs().max() > 20:
        raise ValueError('extreme behavior/current mismatch')
    ratio = delta.exp()
    loss = -torch.minimum(ratio*advantage, ratio.clamp(1-epsilon,1+epsilon)*advantage).mean()
    return loss, {'kl_k3':float((ratio-1-delta).detach().mean()),
                  'ratio_mean':float(ratio.detach().mean()),
                  'clip_fraction':float(((ratio<1-epsilon)|(ratio>1+epsilon)).float().mean().detach())}
