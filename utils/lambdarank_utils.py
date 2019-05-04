"""
An implementation of LambdaRank in Tensorflow. Largely based off of
PyTorch implementation:
https://github.com/airalcorn2/RankNet/blob/master/lambdarank.py

Adding custom gradients in purely tensorflow:
https://gist.github.com/harpone/3453185b41d8d985356cbe5e57d67342
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

import random

def idcg(n_rel):
    # IDEAL DCG
    # Assuming binary relevance.
    nums = np.ones(n_rel)
    denoms = np.log2(np.arange(n_rel) + 1 + 1)
    return (nums / denoms).sum()


def get_lambdarank_gradients_for_single_example(_scores, _targets):
    assert _scores.shape == _targets.shape
    assert len(_scores.shape) == 2
    assert _scores.shape[1] == 1

    num_documents= _targets.shape[0]
    new_ordering = np.argsort(-_targets.reshape(num_documents)) #Puts the ones first.
    scores = _scores[new_ordering]
    targets = _targets[new_ordering]

    n_rel = np.sum(_targets).astype(np.int32)
    n_irr = num_documents - n_rel

    sorted_idxs = np.argsort(-scores, axis=0)
    doc_ranks = np.zeros(num_documents)
    doc_ranks[sorted_idxs] = 1 + np.arange(num_documents).reshape((num_documents, 1)).astype(np.float32)
    doc_ranks = doc_ranks.reshape((num_documents, 1))

    score_diffs = scores[:n_rel] - scores[n_rel:].reshape(n_irr)
    exped = np.exp(score_diffs)
    N = 1 / idcg(n_rel)
    dcg_diffs = 1 / np.log2((1 + doc_ranks[:n_rel])).reshape((n_rel, 1)) - (1 / np.log2((1 + doc_ranks[n_rel:]))).reshape(n_irr)

    lamb_updates = 1 / (1 + exped) * N * np.absolute(dcg_diffs)
    lambs = np.zeros((num_documents, 1))

    lambs[:n_rel] += lamb_updates.sum(axis=1, keepdims=True)
    lambs[n_rel:] -= lamb_updates.sum(axis=0, keepdims=True).T

    lambs = lambs[np.argsort(new_ordering)].reshape(num_documents)

    # Correct sign
    lambs = -lambs

    return lambs

def get_lambdarank_gradients_for_batch(scores, targets):
    """
    Should be simple, it batches it. It involves mapping is all
    This is my gradient function essentially.
    """

    assert scores.shape == targets.shape
    assert len(scores.shape) == 2
    num_docs = scores.shape[1]

    grads = []
    for score, target in zip(scores, targets):
        score = score.reshape((num_docs, 1))
        target = target.reshape((num_docs, 1))
        grads.append(get_lambdarank_gradients_for_single_example(score, target))
    
    grads = np.asarray(grads, dtype=np.float32)
    return grads

def _lambdarank_grads(op, grad):
    """Calculates gradients in a py_func because the arguments to this function
    are Tensorflow tensors"""

    scores, targets = op.inputs
    calculated_grads = tf.py_func(get_lambdarank_gradients_for_batch, [scores, targets], tf.float32)
    new_grads = calculated_grads * grad
    return new_grads, None


# Define custom py_func which takes also a grad op as argument:
def my_py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    
    # Need to generate a unique name to avoid duplicates:
    
    # We need to use regular random here, so that we don't use the same seed for train/test.
    # Pretty interesting actually.
    rnd_name = 'PyFuncGrad' + str(random.randint(0, 1E+8)) #We ues random here, so that 
    
    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name, "PyFuncStateless": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def lambdarank_forward(scores, targets):
    """
    We really never do anything with this... it just needs to return some value.
    """
    return scores


def my_lambdarank(scores, targets, name=None):
    """
    A complete tensorflow implementation of lambdarank!
    """
    with tf.name_scope(name, "my_lambdarank", [scores, targets]):
        lambdarank_op = my_py_func(lambdarank_forward,
                            [scores, targets],
                            tf.float32,
                            name=name,
                            grad=_lambdarank_grads)
        return lambdarank_op
