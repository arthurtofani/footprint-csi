# footprint-csi

### Build

run `docker-compose build`

### Accessing bash

run `docker-compose run --rm csi` or `./bash.sh`


### Running CSI evaluation over Chopin's Mazurkas dataset

1. Add the dataset's folder structure into `dataset` folder (or to the place where `/dataset` is pointed at your `docker-compose.yml` file)
2. Use the following script to generate the clique map:
`python3 mazurkas_gen_cliques.py`
3. Generate some queries:
`python3 mazurkas_gen_queries.py`
