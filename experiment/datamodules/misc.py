import numpy as np


def binary_rating_label(rating):
    return int(rating) - 1


def truncate(x, max_len):
    return x[:max_len]


def pad_and_truncate(sequence, maxlen, dtype="int64", padding="post", truncating="post", value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == "pre":
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == "post":
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x
