# NL2SQL: An LLM approach to Natural to SQL


## Description
Enhance natural language to SQL conversion accuracy using T5, a transformer-based model by Google. Fine-tuned on the SPIDER dataset, our approach achieved 95% accuracy, with an additional 6% improvement through constrained decoding steps and post-processing. Bridging the gap between natural language and SQL, this project contributes to NLP and AI advancement.


## Key Features
- Fine-tuned T5 model for precise SQL query generation
- Evaluated on SPIDER dataset, a Text-to-SQL benchmark
- Achieved 95% accuracy with iterative improvements
- Explored constrained decoding steps and post-processing
- Significant implications for user-friendly database interactions


## Deployment

### Fine Tuning T5 with schema

Run `models/t5finetuning.py` 

### Adapter Training

Run `models/adaptertraining.py` 


### Download Finetuned T5
```
git lfs install
git clone https://huggingface.co/anmolsharma142/t5finetuned
```
### Download Trained adapter
```
git clone https://huggingface.co/anmolsharma142/adapternlp2sql
```
### Inference from finetuned T5


Run `models/finetuned_inference.py` (`gold.txt` and `pred.txt` will be generated)

### Inference from Adapter


Run `models/adapter_inference.py` (`gold.txt` and `pred.txt` will be generated)


#### TBD: New running commands go here: model to results pipeline


Run `notebooks/inference_testing.ipynb` (`gold.txt` and `pred.txt` will be generated)




-------


### Constrained Decoding Code


Certain extra software support is required for the deployment of the haskell based attoparsec parser which forms the main crux of the constrained decoding. Also, we do not support apple silicon (M series) deployment currently. The main route for deployment has been designed through docker. There are two ways one can go about using the code:


#### Docker: Use Pre-Built
Since, building image can take up time. The script has been also set to a desired dockerhub image which contains all the necessary library support for running the code and generating output. Before deployment, the model name and other params can be set in `configs/eval.json`. Below is the simple command that should run the code in docker, evaluate it through test suite and return the results in the desired format:

```
# Download dataset (run only once)
sh ./download_datasets.sh
```

```
make eval


## Post processing Code goes here
git clone https://github.com/taoyds/test-suite-sql-eval.git


python test-suite-sql-eval/evaluation.py --gold gold.txt --pred pred.txt --db spider_dataset/database --table spider_dataset/tables.json --etype all
```


#### Docker: Build from scratch
Users can alternatively build the image using the command `docker build .`. The same image can be used to run the same Makefile with the new image name in place.


-------


## Post-processing
* Create `./post_processing/output.csv` from `gold.txt` and `pred.txt`:
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


For more requirements checkout requirements.txt


-------


## Evaluation Credits
We are implmenting Component Matching metric of SPIDER dataset using the spider validation test suite given in the paper:


```
Ruiqi Zhong, Tao Yu, and Dan Klein. 2020. Semantic evaluation for text-to-sql with distilled test suite.
In The 2020 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics.
```


-------


## Future Work
Explore scalability, diverse datasets, and advanced techniques for continued accuracy enhancement. Also explore ML based parsing.


*Contributions welcome!*

