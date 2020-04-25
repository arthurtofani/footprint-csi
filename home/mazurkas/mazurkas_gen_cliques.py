import csv
import os

def generate_clique_map(entries_path, filename):
  '''
  Creates a clique map in a csv file
  rows are in the format: <group>\t<filepath>
  whereas group is the folder path (the same for all files in the clique)
  '''
  print('Generating clique map...')
  f = open(entries_path, 'r', encoding='utf-8')
  files = f.read().split("\n")[0:-1]
  with open(filename, 'w') as f2:
    writer = csv.writer(f2, delimiter='\t')
    for file in files:
      writer.writerow([os.path.dirname(file), file])
  f.close()
  print('Done!')

generate_clique_map('configs/mazurka_all.txt', 'configs/mazurka_cliques.csv')


