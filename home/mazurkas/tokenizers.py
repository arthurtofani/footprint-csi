import footprint.tokenizers as tokenizers
from itertools import zip_longest
from datasketch import MinHash
import hashlib
import numpy as np

def magic_tokenizer(feature_name, min_hash_fns=10, shingle_size=3):
  def tokenize_method(audio):
    feature = audio.features[feature_name]
    #proc_ocmi_feature ...
    return tokenizers.magic_hash(feature, min_hash_fns=min_hash_fns, shingle_size=shingle_size)
  return tokenize_method

def matrix_profile_hash(feature_name, min_hash_fns=10, shingle_size=3, use_minhash=True):
  def tokenize_method(audio):
    feature = audio.features[feature_name]
    pace = 20
    offset = 70
    blocks = zip(*[feature[i:] for i in range(shingle_size)])
    prewords = [[chr(int(i*pace)+offset) for i in normalize(np.array(bl))] for bl in blocks]
    a = []
    if use_minhash:
      for preword in prewords:
        m = MinHash(num_perm=min_hash_fns)
        m.update(' '.join(preword).encode('utf-8'))
        tx = ''.join([str(c) for c in m.hashvalues])
        a.append(hashlib.md5(tx.encode('utf-8')).hexdigest())
      return ' '.join(a)
    else:
      return ' '.join([''.join(p) for p in prewords])
  return tokenize_method

def normalize(bl):
  return (bl - bl.min()) / (bl.max() - bl.min())
