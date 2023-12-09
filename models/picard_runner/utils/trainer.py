import collections
import torch
from torch import nn
from torch.cuda.amp import autocast
from typing import Any, Dict, List, Optional, NamedTuple, Tuple, Union
import transformers.trainer_seq2seq
from transformers.file_utils import WEIGHTS_NAME
from transformers.modeling_utils import unwrap_model, PreTrainedModel
from transformers.trainer_utils import PredictionOutput, speed_metrics
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.optimization import get_scheduler, Adafactor, AdamW
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import ShardedDDPOption
from transformers.file_utils import is_sagemaker_mp_enabled
from datasets.arrow_dataset import Dataset
from datasets.metric import Metric
from packaging import version
import numpy as np
import time
import os

if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast
    
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

TRAINING_ARGS_NAME = "training_args.bin"


class EvalPrediction(NamedTuple):
    predictions: List[str]
    label_ids: np.ndarray
    metas: List[dict]


class Seq2SeqTrainer(transformers.trainer_seq2seq.Seq2SeqTrainer):
    def __init__(
        self,
        metric: Metric,
        *args,
        eval_examples: Optional[Dataset] = None,
        ignore_pad_token_for_loss: bool = True,
        target_with_db_id: bool = False,
        convert_numbers_to_text: bool = False,
        should_save: bool = False,
        push_to_hub: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.metric = metric
        self.eval_examples = eval_examples
        self.compute_metrics = self._compute_metrics
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.target_with_db_id = target_with_db_id
        self.convert_numbers_to_text = convert_numbers_to_text
        self.control.should_save = should_save
        self.push_to_hub = push_to_hub

    def _compute_metrics(self, eval_prediction: EvalPrediction) -> dict:
        raise NotImplementedError()

    def _post_process_function(
        self, examples: Dataset, features: Dataset, predictions: np.ndarray, stage: str
    ) -> EvalPrediction:
        raise NotImplementedError()

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        eval_examples: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = None,
        max_time: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> Dict[str, float]:
        self._max_length = max_length
        self._max_time = max_time
        self._num_beams = num_beams

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples
        start_time = time.time()

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output: PredictionOutput = self.evaluation_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics

        # We might have removed columns from the dataset so we put them back.
        if isinstance(eval_dataset, Dataset):
            eval_dataset.set_format(
                type=eval_dataset.format["type"],
                columns=list(eval_dataset.features.keys()),
            )

        if eval_examples is not None and eval_dataset is not None and self.compute_metrics is not None:
            eval_preds = self._post_process_function(
                eval_examples,
                eval_dataset,
                output.predictions,
                "eval_{}".format(self.state.epoch),
            )
            output.metrics.update(self.compute_metrics(eval_preds))

        n_samples = len(eval_dataset if eval_dataset is not None else self.eval_dataset)
        output.metrics.update(speed_metrics(metric_key_prefix, start_time, n_samples))

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(output.metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                output.metrics[f"{metric_key_prefix}_{key}"] = output.metrics.pop(key)

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def predict(
        self,
        test_dataset: Dataset,
        test_examples: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = None,
        max_time: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> PredictionOutput:
        self._max_length = max_length
        self._max_time = max_time
        self._num_beams = num_beams

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        if test_dataset is not None and not isinstance(test_dataset, collections.abc.Sized):
            raise ValueError("test_dataset must implement __len__")

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output: PredictionOutput = self.evaluation_loop(
                test_dataloader,
                description="Prediction",
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.compute_metrics is not None:
            # We might have removed columns from the dataset so we put them back.
            if isinstance(test_dataset, Dataset):
                test_dataset.set_format(
                    type=test_dataset.format["type"],
                    columns=list(test_dataset.features.keys()),
                )

            eval_preds = self._post_process_function(
                test_examples, test_dataset, output.predictions, metric_key_prefix)
            output.metrics.update(self.compute_metrics(eval_preds))

        output.metrics.update(speed_metrics(metric_key_prefix, start_time, len(test_dataset)))

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(output.metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                output.metrics[f"{metric_key_prefix}_{key}"] = output.metrics.pop(key)

        self.log(output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Modified https://huggingface.co/transformers/v4.11.3/_modules/transformers/trainer_seq2seq.html

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        
        # Extract the model from any wrapper classes, e.g. DistributedDataParallel
        unwrapped_model = unwrap_model(self.model)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else unwrapped_model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else unwrapped_model.config.num_beams,
            "synced_gpus":  True if is_deepspeed_zero3_enabled() else False,
        }

        if "relations" in inputs.keys():
            generated_tokens = unwrapped_model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                relations=inputs["relations"],
                **gen_kwargs,
            )
        else:
            generated_tokens = unwrapped_model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs,
            )
            
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return (loss, generated_tokens, labels)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        print(f'Model type: {type(self.model)}')
        print(f'Unwrapped model type: {type(unwrap_model(self.model))}')
        if not isinstance(self.model, PreTrainedModel):
            print("Model is not an instance of PreTrainedModel")
            if isinstance(unwrap_model(self.model), PreTrainedModel):
                print("Unwrapped model is an instance of PreTrainedModel, saving unwrapped model")
                if state_dict is None:
                    state_dict = self.model.state_dict()
                unwrap_model(self.model).save_pretrained(output_dir, state_dict=state_dict)
            else:
                print("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if state_dict is None:
                    state_dict = self.model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            print("Model is an instance of PreTrainedModel, saving.")
            self.model.save_pretrained(output_dir, state_dict=state_dict)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            # decay_parameters = [name for name in decay_parameters if "shared" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]
            optimizer_cls = Adafactor if self.args.adafactor else AdamW
            if self.args.adafactor:
                optimizer_cls = Adafactor
                optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
            else:
                optimizer_cls = AdamW
                optimizer_kwargs = {
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                }
            optimizer_kwargs["lr"] = self.args.learning_rate
            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer
        
