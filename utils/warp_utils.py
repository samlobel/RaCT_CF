"""
Creates a function to be used in tf.py_func, which produces a faithful
representation of the WARP loss (with margin set to 0) that runs in
almost linear time (aside from a sorting operation).
"""

import numpy as np

from time import time
import random

class timing_decorator:
    def __init__(self, name):
        self.name = name

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            t = time()
            v = func(*args, **kwargs)
            print("Time to do {} is {}".format(self.name, time()-t))
            return v

        return wrapper



def create_rank_weighting_vector(dimension):
    """This part is easy at least."""
    transformer = np.zeros((dimension,), dtype=np.float32)
    transformer[0] = 1  # I think this is right.
    for i in range(1, dimension):
        transformer[i] = transformer[i - 1] + (1. / (1 + i))
    return transformer


def create_count_above_vector(target_vector=None, argsort_vector=None):
    """
    For each index, this should tell you how many "unseen" items are ranked above it.
    For a "seen" item, this is equivalent to the number of pairs the item
    will appear in for the loss function.
    """

    reverse_target_vector = -1*target_vector + 1 # So it's 1 when it was 0, and vise versa.
    sorted_reverse_target_vector = reverse_target_vector[argsort_vector]
    cumsum = np.cumsum(sorted_reverse_target_vector)
    count_above_vector = cumsum[np.argsort(argsort_vector)]

    count_above_vector+=target_vector
    count_above_vector-= 1

    count_above_vector = count_above_vector.astype(np.float32)

    return count_above_vector


def create_error_vector_with_count_above_vector(target_vector=None,
                                                argsort_vector=None,
                                                rank_weighting_vector=None,
                                                count_above_vector=None):
    """
    Given the number of "unseen" items above each "seen" item, calculates the
    error vector for a single user.
    """

    assert target_vector is not None
    assert argsort_vector is not None
    assert rank_weighting_vector is not None
    assert count_above_vector is not None

    assert target_vector.shape == argsort_vector.shape == rank_weighting_vector.shape == count_above_vector.shape
    assert len(target_vector.shape) == 1

    reversed_argsort_vector = np.flip(argsort_vector, axis=0)
    # reverse_argsorted_rank_weight = rank_weighting_vector[reversed_argsort_vector]
    flipped_rank_weight = np.flip(rank_weighting_vector, axis=0)
    reverse_argsorted_target_vector = target_vector[reversed_argsort_vector]
    reverse_argsorted_count_above = count_above_vector[reversed_argsort_vector]

    good_part_of_argsorted_error_vector = (
        reverse_argsorted_count_above *
        flipped_rank_weight * 
        reverse_argsorted_target_vector
        * -1) #The target-vector part is so we only have that component.

    amount_per_bad_vector = np.cumsum(reverse_argsorted_target_vector * flipped_rank_weight)
    bad_part_of_argsorted_error_vector = amount_per_bad_vector * (1 - reverse_argsorted_target_vector)

    argsorted_error_vector = bad_part_of_argsorted_error_vector + good_part_of_argsorted_error_vector
    error_vector = argsorted_error_vector[np.argsort(reversed_argsort_vector)]

    return error_vector


def create_error_vector_from_raw_inputs(score_vector=None,
                                        target_vector=None,
                                        rank_weighting_vector=None):
    """
    score_vector: The prediction from the model
    target_vector: 0 if an item was not seen, 1 if it was.
    rank_weighting_vector: a precomputed vector to scale the loss associated
    with a given pair of items.

    Outputs the WARP loss for a SINGLE user (un-batched)
    """

    assert score_vector is not None
    assert target_vector is not None
    assert rank_weighting_vector is not None

    assert score_vector.shape == target_vector.shape
    assert len(score_vector.shape) == 1

    argsort_vector = np.argsort(score_vector)
    # rank_vector = np.argsort(argsort_vector)

    count_above_vector = create_count_above_vector(
        target_vector=target_vector, argsort_vector=argsort_vector)

    error_vector = create_error_vector_with_count_above_vector(
        target_vector=target_vector,
        argsort_vector=argsort_vector,
        rank_weighting_vector=rank_weighting_vector,
        count_above_vector=count_above_vector)

    return error_vector


def batch_create_error_vector(score_vector=None, target_vector=None, rank_weighting_vector=None):
    """
    Confirms the inputs. Then makes one per
    """

    assert score_vector is not None
    assert target_vector is not None
    assert rank_weighting_vector is not None

    assert score_vector.shape == target_vector.shape
    assert len(score_vector.shape) == 2

    to_return = np.zeros_like(score_vector, dtype=np.float32)

    for i in range(len(score_vector)):
        single_score_vector = score_vector[i]
        single_target_vector = target_vector[i]
        # t = time()
        to_return[i][...] = create_error_vector_from_raw_inputs(
            score_vector=single_score_vector,
            target_vector=single_target_vector,
            rank_weighting_vector=rank_weighting_vector)
        # print("Time to do single error vector from raw inputs: {}".format(time() - t))
    return to_return

class ErrorVectorCreator:
    """
    A truthful implementation (sans margin) of the WARP loss, described here
    http://www.thespermwhale.com/jaseweston/papers/wsabie-ijcai.pdf

    Because WARP loss is a piecewise linear pairwise loss, we found a
    smart way to speed it up from O(n*m) to O(n log n), which on our datasets
    is a practical increase of ~100x

    It works by sorting the inputs, grouping the pairwise loss terms by
    "seen" item, simplifying each of these groups in constant time to one term,
    and then summing them up.

    This will be initialized once, and then used as the py_func function.

    It outputs an error_scaling vector for a listwise margin-less WARP loss.

    You take the dot product of the prediction-vector with this output in
    order to get the WARP-loss.
    """

    def __init__(self, input_dim=None, margin=0.0, verbose=False):
        assert input_dim is not None
        self.input_dim = input_dim
        self.transformer = create_rank_weighting_vector(input_dim)
        self.transformer.flags.writeable = False
        self.verbose = verbose

    def __call__(self, score_vector, target_vector):
        assert score_vector.shape[1] == self.input_dim
        assert target_vector.shape[1] == self.input_dim

        score_vector = -1 * score_vector

        t = time()
        to_return = batch_create_error_vector(
            score_vector=score_vector,
            target_vector=target_vector,
            rank_weighting_vector=self.transformer)
        if self.verbose:
            print("Time to do new error_vector_creator is {}".format(time() - t))
        return to_return
