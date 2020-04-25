import csv
import os
import numpy as np
import random

amnt_cliques = 4
files_per_clique = 3
filename = 'configs/mazurkas_49x11.txt'

def read_clique_map(filename):
  f = open(filename, 'r', encoding='utf-8')
  return list(csv.reader(f, delimiter='\t'))

clique_map = read_clique_map('configs/mazurka_cliques.csv')
clique_list = np.unique([f[0] for f in clique_map])
selected_cliques = random.sample(list(clique_list), amnt_cliques)

files = []
for clique in selected_cliques:
  ff = [f[1] for f in clique_map if f[0]==clique]
  files += random.sample(ff, files_per_clique)

with open(filename, 'w') as f2:
  writer = csv.writer(f2, delimiter='\t')
  for file in files:
    writer.writerow([file])
