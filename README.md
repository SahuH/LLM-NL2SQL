# NL2SQL

## Evaluation
We are implmenting Component Matching metric of SPIDER dataset using the spider validation test suite given in the paper:

```
Ruiqi Zhong, Tao Yu, and Dan Klein. 2020. Semantic evaluation for text-to-sql with distilled test suite.
In The 2020 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics.
```

Clone Test suite Repo:
```
git clone https://github.com/taoyds/test-suite-sql-eval.git
```

Run the evaluation script:
```
python test-suite-sql-eval/evaluation.py --gold gold.txt --pred pred.txt --db spider_dataset/database --table spider_dataset/tables.json --etype all
```

-------

## Use Pre-trained T5

Run `notebooks/inference_testing.ipynb` (`gold.txt` and `pred.txt` will be generated)

## Fine Tuning T5 with schema

Run `notebooks/Finetuning_T5.ipynb` (`gold.txt` and `pred.txt` will be generated)

-------

## Post-processing
* Create `output.csv` from `gold.txt` and `pred.txt`:
```
python create_output_csv.py
```
* Apply table/column name correction on predictions in `output.csv`:
```
python ./post_processing/entity_correction.py <path to tables.json> <path to output.csv> 
```
* Apply table aliasing correction on predictions in `output.csv`:
```
python ./post_processing/tables_alias_correction.py <path to tables.json> <path to output.csv> 
```

-------

## Dependencies
* Python 3.6
* torch 2.1.0
* transformers 4.34.1
* sentencepiece 0.1.99
* datasets 2.13.2
* nltk 3.8.1
* torchinfo 1.5.4
