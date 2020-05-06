import footprint.evaluators as evaluators
from footprint.models import Project
from multiprocessing import Pool
import os
import csv
import random
import mazurkas

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

#generate_clique_map(entries_path, expect_path)


expect_path = abs_path('mazurkas/configs/mazurka_cliques.csv')
entries_path = abs_path('mazurkas/configs/mazurka_test_entries.txt')
queries_path = abs_path('mazurkas/configs/mazurka_test_entries.txt')
#entries_path = abs_path('mazurkas/configs/mazurka_49x11.txt')
#queries_path = abs_path('mazurkas/configs/mazurka_49x11.txt')

max_processors = 3

p = Project(cache_signal=True, cache_features=True, cache_tokens=False, cache_folder='/cache')
p.process_feature('beats', mazurkas.features.beats)
p.process_feature('chroma_cens', mazurkas.features.chroma_cens)

ms = [3, 6]
for m in ms:
  p.process_feature('matrix_profile_%s' % m, mazurkas.features.matrix_profile('chroma_cens', m))

p.process_feature('chroma_ocmi', mazurkas.features.chroma_ocmi)
#p.process_feature('beat_chroma_cens', mazurkas.features.beat_chroma_cens)
#p.process_feature('beat_chroma_ocmi', mazurkas.features.beat_chroma_ocmi)
#p.process_feature('crema', mazurkas.features.crema)
#p.process_feature('crema_ocmi', mazurkas.features.crema_ocmi)

#for m in ms:
#  p.tokenize('tk_mprofile_hsh_%s' % m, mazurkas.tokenizers.matrix_profile_hash('matrix_profile_%s' % m, min_hash_fns=20, shingle_size=5))

for m in ms:
  p.tokenize('tk_mprofile_%s' % m, mazurkas.tokenizers.matrix_profile_hash('matrix_profile_%s' % m, min_hash_fns=20, shingle_size=4, use_minhash=False))
p.tokenize('tk_chroma_cens', mazurkas.tokenizers.magic_tokenizer('chroma_cens', min_hash_fns=20, shingle_size=2))
p.tokenize('tk_chroma_ocmi', mazurkas.tokenizers.magic_tokenizer('chroma_ocmi', min_hash_fns=20, shingle_size=1))
#p.tokenize('tk_crema', mazurkas.tokenizers.magic_tokenizer('crema', min_hash_fns=20, shingle_size=1))
#p.tokenize('tk_crema_ocmi', mazurkas.tokenizers.magic_tokenizer('crema_ocmi', min_hash_fns=20, shingle_size=1))
#p.tokenize('tk_beat_chroma_cens', mazurkas.tokenizers.magic_tokenizer('beat_chroma_cens', min_hash_fns=20, shingle_size=1))
#p.tokenize('tk_beat_chroma_ocmi', mazurkas.tokenizers.magic_tokenizer('beat_chroma_ocmi', min_hash_fns=20, shingle_size=1))

mazurkas.db.connect_to_elasticsearch(p)
#p.client.set_scope('csi', ['tk_chroma_cens', 'tk_chroma_ocmi', 'tk_crema', 'tk_crema_ocmi'], 'tokens_by_spaces')
profiles = ['tk_mprofile_%s' % m for m in ms]
p.client.set_scope('csi', ['tk_mprofile_6', 'tk_mprofile_3', 'tk_chroma_ocmi', 'tk_chroma_cens'], 'tokens_by_spaces')


def preprocess_audio(filename):
  p.load_audio(filename)

evaluator = evaluators.CSI(p)

#for total_files in evaluator.preprocess(entries_path):
#  with Pool(max_processors) as pool:
#    pool.map(preprocess_audio , total_files)

print('building')
evaluator.build(entries_path)

import time
time.sleep(3)
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



## ==== Results ===
## Mean Average Precision (MAP)                 0.712668
## Mean number of covers in top 10              7.484230
## Mean rank of first correct cover (MRR)       2.886827
## Total candidates                           539.000000
## Total cliques                               49.000000
## Total covers in top 10                    4294.000000
## Total queries                              539.000000

## ==== Total correct covers at rank positions ===
## 1     490
## 2     487
## 3     474
## 4     470
## 5     466
## 6     460
## 7     439
## 8     410
## 9     338
## 10    260



##p.process_feature('beat_chroma_cens', beat_sync_chroma_cens)
##p.process_feature('beat_chroma_ocmi', beat_chroma_ocmi)
##p.process_feature('chroma_ocmi_4b', proc_ocmi_feature('chroma_censx', ocmi.ocmi, reshape=(0, 4)))
##p.process_feature('crema', crema)
##p.process_feature('crema_ocmi_4b', proc_ocmi_feature('crema', ocmi.ocmi, reshape=(0, 4)))
#p.process_feature('mfcc', feat_mfcc)
#p.process_feature('mfcc_delta', feat_mfcc_delta)

##p.use_tokenizer('magic2', magic_tokenizer('beat_chroma_ocmi', min_hash_fns=20, shingle_size=2))
##p.use_tokenizer('magic1', magic_tokenizer('beat_chroma_cens', min_hash_fns=20, shingle_size=2))
##p.use_tokenizer('magic5', magic_tokenizer('crema', min_hash_fns=20, shingle_size=1))
##p.use_tokenizer('magic7', magic_tokenizer('crema_ocmi_4b', min_hash_fns=20, shingle_size=1))
#p.use_tokenizer('magic8', magic_tokenizer('mfcc', min_hash_fns=20, shingle_size=1))
#p.use_tokenizer('magic9', magic_tokenizer('mfcc_delta', min_hash_fns=20, shingle_size=1))

#p.client.set_scope('csi', ['magic4', 'magic3'], 'tokens_by_spaces') -- best
#p.add('/dataset/YTCdataset/letitbe/test.mp3')
#import code; code.interact(local=dict(globals(), **locals()))
