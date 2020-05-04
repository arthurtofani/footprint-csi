import footprint.tokenizers as tokenizers

def magic_tokenizer(feature_name, min_hash_fns=10, shingle_size=3):
  def tokenize_method(audio):
    feature = audio.features[feature_name]
    #proc_ocmi_feature ...
    return tokenizers.magic_hash(feature, min_hash_fns=min_hash_fns, shingle_size=shingle_size)
  return tokenize_method
