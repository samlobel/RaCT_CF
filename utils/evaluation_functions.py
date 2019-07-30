import bottleneck as bn
import numpy as np
from scipy import sparse

def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=100, input_batch=None, normalize=True):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance

    If normalize is set to False, then we actually return DCG, not NDCG.


    '''
    if input_batch is not None:
        X_pred[input_batch.nonzero()] = -np.inf
    batch_users = X_pred.shape[0]
    # Get the indexes of the top K predictions.
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    # Get only the top k predictions.
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    # Get sorted index...
    idx_part = np.argsort(-topk_part, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    # Get sorted index...
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))
    # You add up the ones you've seen, scaled by their discount...
    # top_k_results = heldout_batch[np.arange()]
    maybe_sparse_top_results = heldout_batch[np.arange(batch_users)[:, np.newaxis], idx_topk]
    try:
        top_results = maybe_sparse_top_results.toarray()
    except:
        top_results = maybe_sparse_top_results

    try:
        number_non_zero = heldout_batch.getnnz(axis=1)
    except:
        number_non_zero = ((heldout_batch > 0) * 1).sum(axis=1)

    DCG = (top_results * tp).sum(axis=1)
    # DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
    #                      idx_topk].toarray() * tp).sum(axis=1)
    # Gets denominator, could be the whole sum, could be only part of it if there's not many.
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in number_non_zero])

    IDCG = np.maximum(0.1, IDCG) #Necessary, because sometimes you're not given ANY heldout things to work with. Crazy...
    # IDCG = np.array([(tp[:min(n, k)]).sum()
    #                  for n in heldout_batch.getnnz(axis=1)])
    # to_return = DCG / IDCG
    # if np.any(np.isnan(to_return)):
    #     print("bad?!")
    #     import ipdb; ipdb.set_trace()
    #     print("dab!?")
    if normalize:
        result = (DCG / IDCG)
    else:
        result = DCG
    result = result.astype(np.float32)
    return result

def Recall_at_k_batch(X_pred, heldout_batch, k=100, input_batch=None):
    if input_batch is not None:
        X_pred[input_batch.nonzero()] = -np.inf
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0)#.toarray()
    try:
        X_true_binary = X_true_binary.toarray()
    except:
        # print("Wasn't sparse")
        pass

    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)
    recall = tmp / np.maximum(np.minimum(k, X_true_binary.sum(axis=1)), 1)
    recall = recall.astype(np.float32)
    return recall

def average_precision_at_k(scores, ground_truth, k=100):
    """
    Assumes that ground-truth is 0 for false and 1 for true. This rests heavily on that.
    """

    assert scores.shape == ground_truth.shape
    assert len(scores.shape) == 1

    if len(ground_truth) < k:
        k = len(ground_truth)

    total_num_good = np.sum(ground_truth)
    if total_num_good < 1:
        return 0.0 #If there are no true items, say the precision is 0. Just convention.

    # Argpartition on the whole length does nothing and throws an error.
    if k == len(ground_truth):
        idx_topk_part = np.arange(k)
    else:
        idx_topk_part = bn.argpartition(-1 * scores, k)
    topk_part = scores[idx_topk_part[:k]]
    idx_part = np.argsort(-topk_part)

    top_k_sorted_indices = idx_topk_part[idx_part]

    running_score = 0.0
    num_good_seen = 0.0
    for i in range(k):
        ranked_k_index = top_k_sorted_indices[i]
        if ground_truth[ranked_k_index]:
            num_good_seen += 1
            precision = num_good_seen / (i + 1.0)
            running_score += precision

    recall_scaler = min(total_num_good, k)
    return running_score / recall_scaler


def average_precision_at_k_batch(X_pred, heldout_batch, k=100, input_batch=None):
    if input_batch is not None:
        X_pred[input_batch.nonzero()] = -np.inf
    
    assert X_pred.shape == heldout_batch.shape
    assert len(X_pred.shape) == 2

    zipped = zip(X_pred, heldout_batch)
    aps = [average_precision_at_k(scores, ground_truth, k=k) for scores, ground_truth in zipped]
    aps = np.asarray(aps, dtype=np.float32)
    return aps
