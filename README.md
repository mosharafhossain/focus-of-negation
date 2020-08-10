Focus of Negation 
===================================================================

## Requirements
Python 3.6+ (recommended: Python 3.7) \
Python packages: list of packages are provided in ./env-setup/requirements.txt file. \
Embedding: Download the ELMo embedding weights and options files from https://allennlp.org/elmo and put it into ./embeddings/elmo directory
(Additional guideline is provided inside ./embeddings/elmo directory )

```bash
# Create virtual env (Assuming you have Python 3.6 or 3.7 installed in your machine) -> optional step
python3 -m venv your_location/focus-of-negation
source your_location/focus-of-negation/bin/activate

# Install required packages -> required step
pip install -r ./env-setup/requirements.txt
python -m spacy download en_core_web_sm
```

## How to Run

- Example command to train the focus-of-negation model: 
```bash
  python train.py -c ./config/config.json 
```
  + Arguments:
	  - -c, --config_path: path to the configuration file, (required)
  
  *Note that: a trained model is already provided in "./model" folder if the user does not want to train the model.  
	
- Example command to prepare prediction on PB-FOC test corpus. 
```bash
  python predict.py -c ./config/config.json 
```
  + Arguments:
	  - -c, --config-path: path to the configuration file; (required). Contains details parameter settings.

## Evaluation
- Example command to get evaluation scores for PB-FOC test corpus.
```bash
 python ./data/pb-foc/src/pb-foc_evaluation.py ./data/pb-foc-prediction/prediction_on_test.txt ./data/pb-foc/corpus/SEM-2012-SharedTask-PB-FOC-te.merged 
```

## Citation

Please cite our paper if the paper (and code) is useful to you. paper: "Predicting the Focus of Negation: Model and Error Analysis". 
```bibtex
@inproceedings{hossain-etal-2020-predicting,
    title = "Predicting the Focus of Negation: Model and Error Analysis",
    author = "Hossain, Md Mosharaf  and
      Hamilton, Kathleen  and
      Palmer, Alexis  and
      Blanco, Eduardo",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.743",
    pages = "8389--8401",
    abstract = "The focus of a negation is the set of tokens intended to be negated, and a key component for revealing affirmative alternatives to negated utterances. In this paper, we experiment with neural networks to predict the focus of negation. Our main novelty is leveraging a scope detector to introduce the scope of negation as an additional input to the network. Experimental results show that doing so obtains the best results to date. Additionally, we perform a detailed error analysis providing insights into the main error categories, and analyze errors depending on whether the model takes into account scope and context information.",
}
```
