import torch
import torch.nn.functional as F

from train_utils import ce_loss


class Get_Scalar:
    """
    Small callable wrapper used by the training loop for scalar schedules.
    The original configs pass constants, so this keeps the same interface.
    """

    def __init__(self, value):
        self.value = value

    def __call__(self, _):
        return self.value


def consistency_loss(logits_s, logits_w, name='ce', T=0.5, p_cutoff=0.95,
                     use_hard_labels=True):
    """
    FixMatch-style directional consistency loss.

    Args:
        logits_s: logits of the strongly augmented view to optimize.
        logits_w: logits of the weakly augmented anchor view for pseudo labels.
        name: currently only cross entropy is used by this project.
        T: temperature for soft pseudo labels.
        p_cutoff: confidence threshold.
        use_hard_labels: use class indices when True, probability vectors otherwise.

    Returns:
        loss, selected-ratio, selected-indices, pseudo-labels.
    """
    assert name == 'ce'

    probs = torch.softmax(logits_w.detach(), dim=-1)
    max_probs, pseudo_label = torch.max(probs, dim=-1)
    mask = max_probs.ge(p_cutoff).float()
    select = mask.nonzero(as_tuple=True)[0]

    if use_hard_labels:
        loss = ce_loss(logits_s, pseudo_label, use_hard_labels=True, reduction='none')
    else:
        pseudo_label = torch.softmax(logits_w.detach() / T, dim=-1)
        loss = ce_loss(logits_s, pseudo_label, use_hard_labels=False, reduction='none')

    loss = (loss * mask).mean()
    return loss, mask.mean(), select, pseudo_label


def consistency_loss_con(logits_s, logits_w, confidence, name='ce', T=0.5,
                         p_cutoff=0.95, use_hard_labels=True,
                         use_threshold=False):
    """
    Confidence-guided consistency loss used by ConMatch.

    The pseudo label still comes from the anchor logits. The per-sample loss is
    weighted by the confidence estimator output; optional thresholding preserves
    the FixMatch-style masking interface when requested by configs/ablations.
    """
    assert name == 'ce'

    probs = torch.softmax(logits_w.detach(), dim=-1)
    max_probs, pseudo_label = torch.max(probs, dim=-1)
    mask = max_probs.ge(p_cutoff).float()
    select = mask.nonzero(as_tuple=True)[0]

    if use_hard_labels:
        loss = ce_loss(logits_s, pseudo_label, use_hard_labels=True, reduction='none')
    else:
        pseudo_label = torch.softmax(logits_w.detach() / T, dim=-1)
        loss = ce_loss(logits_s, pseudo_label, use_hard_labels=False, reduction='none')

    weight = confidence.float()
    if use_threshold:
        weight = weight * mask

    loss = (loss * weight).mean()
    return loss, mask.mean(), select, pseudo_label


def confidence_loss(con_true, con_pred, num_true=None, num_total=None,
                    is_weighted_BCE=False):
    """
    Binary confidence-estimator loss.

    Kept for compatibility with older commented training paths. The active code
    uses nn.BCELoss directly, but some ablations still import this function.
    """
    con_true = con_true.float()
    con_pred = con_pred.float().clamp(1e-6, 1.0 - 1e-6)

    if not is_weighted_BCE:
        return F.binary_cross_entropy(con_pred, con_true, reduction='mean')

    if num_true is None or num_total is None or num_true == 0:
        return F.binary_cross_entropy(con_pred, con_true, reduction='mean')

    num_false = max(float(num_total - num_true), 1.0)
    pos_weight = float(num_false) / float(num_true)
    weight = torch.ones_like(con_true)
    weight = torch.where(con_true > 0.5, weight * pos_weight, weight)
    return F.binary_cross_entropy(con_pred, con_true, weight=weight, reduction='mean')
