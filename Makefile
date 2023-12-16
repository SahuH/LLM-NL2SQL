GIT_HEAD_REF := $(shell git rev-parse HEAD)
BASE_IMAGE := pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel
IMAGE_NAME := totem37/docu-t5
IMAGE_TAG := 03112022
BUILDKIT_BUILDER ?= buildx-local
BASE_DIR := $(shell pwd)

.PHONY: eval
eval:
		mkdir -p -m 777 eval
		mkdir -p -m 777 transformers_cache
		mkdir -p -m 777 wandb
		docker run \
			--gpus all \
			-it --user root -p 8000:8000 \
			--mount type=bind,source=$(BASE_DIR)/wandb,target=/app/wandb \
			--mount type=bind,source=$(BASE_DIR)/transformers_cache,target=/transformers_cache  \
			--mount type=bind,source=$(BASE_DIR)/configs,target=/app/configs \
			--mount type=bind,source=$(BASE_DIR)/models/picard_runner,target=/app/picard_runner \
			--mount type=bind,source=$(BASE_DIR)/eval,target=/eval \
			--mount type=bind,source=$(BASE_DIR)/dataset_files,target=/app/dataset_files \
			$(IMAGE_NAME):$(IMAGE_TAG) \
			/bin/bash -c "pip install stanza && python picard_runner/run_picard.py configs/eval.json"

		@echo "Decompressing predictions_eval_None.json.gz..."
		gzip -d $(BASE_DIR)/eval/predictions_eval_None.json.gz

		
		@echo "Running outputeval.py..."
		python /models/picard_runner/utils/output.py $(BASE_DIR)/eval/predictions_eval_None.json

		
		@echo "Running the evaluation command..."
		python test-suite-sql-eval/evaluation.py --gold $(BASE_DIR)/eval/gold.txt --pred $(BASE_DIR)/eval/pred.txt --db dataset_files/ori_dataset/database --table dataset_files/ori_dataset/tables.json --etype all

		@echo "Evaluation completed"

