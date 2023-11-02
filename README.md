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

Run `T5_inferencing.py` (`gold.txt` and `pred.txt` will be generated)

## Fine Tuning T5 with schema

Run `Finetuning_T5.py` (`gold.txt` and `pred.txt` will be generated)



-------

## Dependencies
* Python 3.6
* torch 2.1.0
* transformers 4.34.1
* sentencepiece 0.1.99
* datasets 2.13.2
* nltk 3.8.1
* torchinfo 1.5.4
