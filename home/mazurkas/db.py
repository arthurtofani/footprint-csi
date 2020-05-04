import footprint.clients as db

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
