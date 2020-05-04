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

#entries_path = abs_path('mazurkas/configs/mazurka_49x11.txt')
#queries_path = abs_path('mazurkas/configs/mazurka_49x11.txt')
#generate_clique_map(entries_path, expect_path)


expect_path = abs_path('mazurkas/configs/mazurka_cliques.csv')
entries_path = abs_path('mazurkas/configs/mazurka_test_entries.txt')
queries_path = abs_path('mazurkas/configs/mazurka_test_entries.txt')
max_processors = 3

p = Project(cache_signal=True, cache_features=True, cache_folder='/cache')
p.process_feature('beats', mazurkas.features.beats)
p.process_feature('chroma_cens', mazurkas.features.chroma_cens)
p.process_feature('chroma_ocmi', mazurkas.features.chroma_ocmi)

p.tokenize('tk_chroma_cens', mazurkas.tokenizers.magic_tokenizer('chroma_cens', min_hash_fns=20, shingle_size=2))
p.tokenize('tk_chroma_ocmi', mazurkas.tokenizers.magic_tokenizer('chroma_ocmi', min_hash_fns=20, shingle_size=1))

mazurkas.db.connect_to_elasticsearch(p)
p.client.set_scope('csi', ['tk_chroma_cens', 'tk_chroma_ocmi'], 'tokens_by_spaces')


def preprocess_audio(filename):
  p.load_audio(filename)

evaluator = evaluators.CSI(p)

for total_files in evaluator.preprocess(entries_path):
  with Pool(max_processors) as pool:
    pool.map(preprocess_audio , total_files)

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
