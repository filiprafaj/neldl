# "Named Entity Linking by Deep Learning" source code

The thesis can be found [here](https://drive.google.com/file/d/1dic_9HvdhlYbFwDJS0Z3OJbLuJGGBod-/view?usp=sharing).
- setup python environment (optional) and install requirements
```
git clone https://github.com/filiprafaj/neldl.git
cd neldl
python3 -m pip install virtualenv
python3 -m virtualenv neldl_env
source neldl_env/bin/activate
pip install -r requirements.txt
```
- Download our `data` folder [here]() and place the it under ``neldl/'' (under `data/experiments` there are some of our experiments).
---
## Reproduce our results

- Choose one of our experiments (e.g. `base_cased`). If you choose other than `base_cased`, you have to copy contents of `base_cased/embeddings` to the `embeddings` folder of the chosen experiment.

**1. BERT embeddings of datasets**

Generate BERT embeddings of the datasets (all *\*.txt* files in `data/base_data/new_datasets`). Choose settings BERT size and casing correspondinly to the chosen experiment (e.g. base, cased)
```
python -m preprocessing.create_new_datasets_embeddings --bert_size=base --bert_casing=cased 
```

**2. Create TFRecords**

For the chosen experiment (e.g. `base_cased`) create a dataset in TFRecords format for each *\*.txt* file in `data/base_data/new_datasets`(it will also contain the BERT embeddings).
```
python -m preprocessing.create_tfrecords --experiment_name bert_cased --bert_size=base --bert_casing=cased 
```

**3. Train**

Choose training settings. For example you can set `no_attention` model using `--local_score_components=pem_similarity`. Our training scripts can be found [here]()
```
python -m model.train --experiment_name=base_cased --training_name=no_attention_reproduce --local_score_components=pem_similarity
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
