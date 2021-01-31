# "Named Entity Linking by Deep Learning" source code

The thesis can be found [here](https://drive.google.com/file/d/1dic_9HvdhlYbFwDJS0Z3OJbLuJGGBod-/view?usp=sharing).
The code is based on https://github.com/dalab/end2end_neural_el/. [[1]](1)
## Set up environment
- clone the repository, create Python environment (optional) and install the required libraries
```
git clone https://github.com/filiprafaj/neldl.git

cd neldl
python3 -m pip install virtualenv
python3 -m virtualenv neldl_env
source neldl_env/bin/activate

pip install -r requirements.txt
```
- Download our `data` folder [here](https://drive.google.com/file/d/1Gpss0Bjeph1JvEgpA7CMqKl_gRDikFfq/view?usp=sharing) and place the it under `neldl/` (under `data/experiments` there are some of our experiments).
---
## Reproduce our results

- Choose one of our experiments (e.g. `base_cased`). If you choose other than `base_cased` or `all_embeddings_base`, you have to copy contents of `base_cased/embeddings` to the `embeddings` folder of the chosen experiment.

**1. BERT embeddings of datasets**

Generate BERT embeddings of the datasets (all *\*.txt* files in `data/base_data/new_datasets`). Choose BERT size and casing correspondinly to the chosen experiment (e.g. base, cased).
```
python -m preprocessing.create_new_datasets_embeddings --bert_size=base --bert_casing=cased 
```

**2. Create TFRecords**

For the chosen experiment (e.g. `base_cased`) create a dataset in TFRecords format for each *\*.txt* file in `data/base_data/new_datasets`(it will also contain the BERT embeddings).
```
python -m preprocessing.create_tfrecords --experiment_name bert_cased --bert_size=base --bert_casing=cased 
```

**3. Train**

Choose training settings. For example you can set `no_attention` model using `--local_score_components=pem_similarity`. Our training scripts can be found [here](https://github.com/filiprafaj/neldl/tree/main/code/scripts)
```
python -m model.train --experiment_name=base_cased --training_name=no_attention_reproduce --local_score_components=pem_similarity
```

**4. Evaluate**

To summarize all results you can use:
```
python -m evaluation.summarize_all_experiments --order_by_test_set
```
To evaluate and to print predictions similar to those [here](https://github.com/filiprafaj/neldl/tree/main/predictions) you can use a command analogous to:
```
python -m evaluation.evaluate --experiment_name base_cased --training_name no_attention --checkpoint_model_num 54
```

---
## Train your model from scratch

To train from scratch you only need the `base_data` folder from the download. Put it under `data` folder.

**I. Create entity universe**

This will collect candidate entities for all spans in *\*.txt* files in `data/base_data/new_datasets`. Also count words and character frequencies.
```
python -m preprocessing.create_entity_universe --experiment_name your_experiment
```
Alternatively you can collect all entities from the probabilistic map *p(e|m)* (`data/base_data/prob_yago_crosswikis_wikipedia_p_e_m.txt`)
```
python -m preprocessing.create_entity_universe_all_entities --experiment_name your_experiment_all_entities
```

**II. Train entity embeddings**

For training entity embeddings, use the [deep-ed-neldl](https://github.com/filiprafaj/deep-ed-neldl) project.
- follow the instructions in the README.md of the [project](https://github.com/filiprafaj/deep-ed-neldl)
  - you will need `data/experiments/your_experiment/entity_universe.txt`
- create folder `data/experiments/your_experiment/embeddings`
- copy *ent_vecs__ep_XX.t7* to `data/experiments/your_experiment/embeddings`
- copy generated files *wikiid2nnid.txt* and *nnid2wikiid.txt* from the root of $DATA_PATH to `data/experiments/your_experiment`
- use this command to extract embeddings from *ent_vecs__ep_XX.t7* to *entity_embeddings.txt*:
```
th preprocessing/bridge_code_lua/ent_vecs_to_txt.lua -ent_vecs_folder ../data/experiments/your_experiment/embeddings -ent_vecs_file ent_vecs__ep_XX.t7
```
- and the following one to create *entity_embeddings.npy* from *entity_embeddings.txt*
```
python -m preprocessing.bridge_code_lua.ent_vecs_from_txt_to_npy --experiment_name your_experiment
```

**III. BERT embeddings of datasets**

Generate BERT embeddings of the datasets (all *\*.txt* files in `data/base_data/new_datasets`)
```
python -m preprocessing.create_new_datasets_embeddings
```

**IV. Create TFRecords**

Create a dataset in TFRecords format for each *\*.txt* file in `data/base_data/new_datasets`(it will also contain the BERT embeddings).
```
python -m preprocessing.create_tfrecords --experiment_name your_experiment
```

**V. Train**

```
python -m model.train --experiment_name=base --training_name=your_training
```
---
## Try models on arbitrary input text
Choose an *experiment*, corresponding *training* and its *checkpoint* to serve. `all_embeddings_base` experiment contains embeddings of all available entities, unlike embeddings in `base_cased`, which are limited only to candidate entities found for the datasets.
```
python -m server.server --experiment_name="all_embeddings_base" --training_name="no_attention" --checkpoint_model_num="15"
```
The server runs on `http://localhost:5555` and expects a json object in a format following format (provide spans for NED or leave spans empty for NEL).
```
{ "text": "Obama will visit Germany and have a meeting with Merkel tomorrow.", "spans": [{"start":0,"length":5}, {"start":49,"length":6}]  }
{ "text": "Obama will visit Germany and have a meeting with Merkel tomorrow.", "spans": []  }
```
To post queries you can use this [Jupyter Notebook](https://github.com/filiprafaj/neldl/blob/main/code/server/client.ipynb), which offers a convenient way of displaying servers response. Of course, you can instead simply run a Python console in another terminal and post queries from there:
```
import requests, json
myjson = { "text": "Obama will visit Germany and have a meeting with Merkel tomorrow.", "spans": []  }
requests.post("http://localhost:5555", json=myjson)
```
The server return linked spans with corresponding Wikipedia titles (just prepend 'https://en.wikipedia.org/wiki/')

---
## References
<a id="1">[1]</a> 
Kolitsas, Nikolaos & Ganea, Octavian & Hofmann, Thomas. (2018). End-to-End Neural Entity Linking. 519-529. 10.18653/v1/K18-1050. 
