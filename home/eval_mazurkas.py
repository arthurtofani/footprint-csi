import footprint.clients as db
import footprint.tokenizers as tokenizers
import footprint.evaluators as evaluators
from footprint.models import Audio
from footprint.models import Project
from footprint.features import ocmi
from footprint.features import crema as cremalib

import os
import csv
import random
import logging
import numpy as np
import librosa


def generate_clique_map(entries_path, filename):
  f = open(entries_path, 'r', encoding='utf-8')
  files = f.read().split("\n")[0:-1]
  with open(filename, 'w') as f2:
    writer = csv.writer(f2, delimiter='\t')
    for file in files:
      writer.writerow([os.path.dirname(file), file])
  f.close()


def abs_path(path):
    dirname = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(dirname, path)

def magic_tokenizer(feature_name, min_hash_fns=10, shingle_size=3):
  def tokenize_method(audio):
    feature = audio.features[feature_name]
    #proc_ocmi_feature ...
    return tokenizers.magic_hash(feature, min_hash_fns=min_hash_fns, shingle_size=shingle_size)
  return tokenize_method

def proc_ocmi_feature(feature_name, method, reshape=(None, None)):
  def feat_method(audio):
    feature = audio.features[feature_name]
    f2 = method(feature)
    return tokenizers.reshape(f2, reshape[0], reshape[1])
  return feat_method

def feat_chroma_cens(audio):
  print('running chroma for ', audio.filename)
  return librosa.feature.chroma_cens(audio.y, audio.sr, hop_length=2**10)

def chroma_ocmi(audio):
  chroma = audio.features['chroma_cens']
  return ocmi.ocmi(chroma)

def crema(audio):
  return cremalib.process(audio)

def crema_ocmi(audio):
  crema = audio.features['crema']
  return ocmi.ocmi(crema)

def beat_chroma_ocmi(audio):
  chroma = audio.features['beat_chroma_cens']
  return ocmi.ocmi(chroma)

def beat_sync_chroma_cens(audio):
  chroma =  librosa.feature.chroma_cens(audio.y, sr=audio.sr, hop_length=2**10)
  audio.load_beats()
  beat_f = librosa.util.fix_frames(audio.beats, x_max=chroma.shape[1])
  return librosa.util.sync(chroma, beat_f, aggregate=np.median)

def connect_to_elasticsearch(p):
  cli = db.elasticsearch.Connection(host='elasticsearch', port=9200)
  cli.clear_index('csi')
  cli.setup_index('csi', initial_settings())
  p.set_connection(cli)


def initial_settings():
  return {
    "settings" : {
      "analysis" : {
        "analyzer" : {
          "tokens_by_spaces": {
            "tokenizer": "divide_tokens_by_spaces"
          }
        },
        "tokenizer": {
          "divide_tokens_by_spaces": {
            "type": "simple_pattern_split",
            "pattern": " "
          }
        }
      }
    }
  }



entries_path = abs_path('mazurkas/configs/mazurka_49x11.txt')
queries_path = abs_path('mazurkas/configs/mazurka_49x11.txt')
expect_path = abs_path('mazurkas/configs/mazurka_cliques.csv')
#entries_path = abs_path('mazurkas/configs/mazurka_test_entries.txt')
#queries_path = abs_path('mazurkas/configs/mazurka_test_entries.txt')

#queries_path = abs_path('fixtures/test/queries_small.txt')

#generate_clique_map(entries_path, expect_path)
#read_clique_map(filename)

p = Project(cache=True, cache_folder='/cache')
p.process_feature('beat_chroma_cens', beat_sync_chroma_cens)
p.process_feature('chroma_censx', feat_chroma_cens)
#p.process_feature('chroma_cens_12', feat_chroma_cens)
p.process_feature('beat_chroma_ocmi', beat_chroma_ocmi)
p.process_feature('chroma_ocmi', chroma_ocmi)

p.process_feature('crema', crema)
p.process_feature('crema_ocmi_4b', proc_ocmi_feature('crema', ocmi.ocmi, reshape=(0, 4)))

#p.process_feature('bchroma_ocmi_norm_4', proc_ocmi_feature('chroma_cens', ocmi.ocmi_norm, reshape=(0, 4)))
p.process_feature('chroma_ocmi_4b', proc_ocmi_feature('chroma_censx', ocmi.ocmi, reshape=(0, 4)))

p.use_tokenizer('magic2', magic_tokenizer('beat_chroma_ocmi', min_hash_fns=20, shingle_size=2))
p.use_tokenizer('magic1', magic_tokenizer('beat_chroma_cens', min_hash_fns=20, shingle_size=2))
p.use_tokenizer('magic3', magic_tokenizer('chroma_ocmi_4b', min_hash_fns=20, shingle_size=1))
p.use_tokenizer('magic4', magic_tokenizer('chroma_censx', min_hash_fns=20, shingle_size=2))

p.use_tokenizer('magic5', magic_tokenizer('crema', min_hash_fns=20, shingle_size=1))
p.use_tokenizer('magic7', magic_tokenizer('crema_ocmi_4b', min_hash_fns=20, shingle_size=1))

connect_to_elasticsearch(p)
#p.client.set_scope('csi', ['magic3', 'magic4', 'magic5', 'magic7'], 'tokens_by_spaces')
p.client.set_scope('csi', ['magic7'], 'tokens_by_spaces')

#p.client.set_scope('csi', ['magic4', 'magic3'], 'tokens_by_spaces') -- best
#p.add('/dataset/YTCdataset/letitbe/test.mp3')
#import code; code.interact(local=dict(globals(), **locals()))

evaluator = evaluators.CSI(p)
print('building')
evaluator.build(entries_path)
print('matching')
evaluator.match(queries_path)


print("\n\n\n==== Results ===")
df1, df2 = evaluator.evaluate(expect_path, 'mazurkas/out.txt')
print(df1.sum())

print("==== Total correct covers at rank positions ===")
print(df2.sum())


#      import code; code.interact(local=dict(globals(), **locals()))

# p.use_tokenizer('magic2', magic_tokenizer('beat_chroma_ocmi', min_hash_fns=20, shingle_size=2))
# p.use_tokenizer('magic1', magic_tokenizer('beat_chroma_cens', min_hash_fns=20, shingle_size=2))
# p.use_tokenizer('magic3', magic_tokenizer('chroma_ocmi_4b', min_hash_fns=20, shingle_size=1))
# p.use_tokenizer('magic4', magic_tokenizer('chroma_censx', min_hash_fns=20, shingle_size=2))
# p.use_tokenizer('magic5', magic_tokenizer('crema', min_hash_fns=20, shingle_size=1))
# p.use_tokenizer('magic7', magic_tokenizer('crema_ocmi_4b', min_hash_fns=20, shingle_size=1))

# p.client.set_scope('csi', ['magic3', 'magic4', 'magic5', 'magic7'], 'tokens_by_spaces')
# ==== Results ===
# Mean Average Precision (MAP)                 0.680111
# Mean number of covers in top 10              7.205937
# Mean rank of first correct cover (MRR)       3.510204
# Total candidates                           539.000000
# Total cliques                               49.000000
# Total covers in top 10                    4113.000000
# Total queries                              539.000000
# dtype: float64
# ==== Total correct covers at rank positions ===
# 1     488
# 2     482
# 3     469
# 4     455
# 5     443
# 6     423
# 7     408
# 8     383
# 9     333
# 10    229
