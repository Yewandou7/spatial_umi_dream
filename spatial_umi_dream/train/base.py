# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# This file is modified from https://github.com/haotian-liu/LLaVA/

import gc
import json
import os
from pathlib import Path
import shutil
import time
from typing import Optional
import warnings

from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
from torch.utils.data import DataLoader
import transformers
from transformers import TrainerCallback, set_seed
from transformers.trainer import (
    TRAINER_STATE_NAME,
    TrainerState,
    get_last_checkpoint,
    get_parameter_names,
    is_sagemaker_mp_enabled,
)

from spatial_umi_dream.data.dataset.sharded import ShardedLeRobotMixtureDataset
from spatial_umi_dream.data.schema import EmbodimentTag
from spatial_umi_dream.data.transform import ComposedModalityTransform
from spatial_umi_dream.train.timer import ContextTimer
from spatial_umi_dream.train.utils import (
    compute_grad_accum_to_match_global_bs,
    dtype_from_string,
    get_checkpoint_path,
    mprint,
    safe_save_model_for_hf_trainer,
)

# Fix resume: https://github.com/huggingface/transformers/pull/34632/files
np_core = np.core
allowlist = [np_core.multiarray._reconstruct, np.ndarray, np.dtype]
# numpy >1.25 defines numpy.dtypes.UInt32DType, but below works for
# all versions of numpy
allowlist += [type(np.dtype(np.uint32))]
torch.serialization.add_safe_globals(allowlist)

# Define LayerNorm classes locally to replace deprecated ALL_LAYERNORM_LAYERS
LAYERNORM_LAYERS = [
    torch.nn.LayerNorm,
    torch.nn.GroupNorm,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
    torch.nn.LocalResponseNorm,
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
]


class LossLoggerCallback(TrainerCallback):
    """Callback that writes per-step loss metrics to a JSONL file for offline analysis."""

    def __init__(self, output_path: str):
        self.output_path = output_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or logs is None:
            return
        entry = {"step": state.global_step}
        for key in ("loss", "rgb_loss", "spatial_loss", "action_loss", "learning_rate"):
            if key in logs:
                entry[key] = logs[key]
        if len(entry) > 1:  # more than just "step"
            with open(self.output_path, "a") as f:
                f.write(json.dumps(entry) + "\n")


class CheckpointFormatCallback(TrainerCallback):
    """Makes every checkpoint standalone: copies experiment_cfg/ (conf.yaml, metadata.json; the
    policy loads both at inference) and wandb_config.json into /checkpoint-{step}/."""

    def __init__(self, exp_cfg_dir: Path):
        self.exp_cfg_dir = exp_cfg_dir

    def on_save(self, args, state, control, **kwargs):
        """Called after the trainer saves a checkpoint."""
        if state.is_world_process_zero:
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"

            exp_cfg_dst = checkpoint_dir / self.exp_cfg_dir.name
            if self.exp_cfg_dir.exists():
                print(f"Copying experiment config directory {self.exp_cfg_dir} to {exp_cfg_dst}")
                shutil.copytree(self.exp_cfg_dir, exp_cfg_dst, dirs_exist_ok=True)

            wandb_config_src = Path(args.output_dir) / "wandb_config.json"
            wandb_config_dst = checkpoint_dir / "wandb_config.json"
            if wandb_config_src.exists():
                print(f"Copying wandb_config.json from {wandb_config_src} to {wandb_config_dst}")
                shutil.copy2(wandb_config_src, wandb_config_dst)


class _StepBoundaryTimer(TrainerCallback):
    """Times the wall-clock between training_step exit and optimizer.step,
    plus optimizer.step itself. Reads/writes a few private attrs on the
    parent BaseTrainer so the values land in wandb via trainer.log()."""

    def __init__(self, trainer):
        self.trainer = trainer
        self._opt_start = None

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        self._opt_start = time.time()

    def on_optimizer_step(self, args, state, control, **kwargs):
        if self._opt_start is not None:
            self.trainer.log({"optimizer_step_time": time.time() - self._opt_start})
            self._opt_start = None

    def on_step_end(self, args, state, control, **kwargs):
        self.trainer._last_step_end_time = time.time()


class BaseTrainer(transformers.Trainer):

    def __init__(self, **kwargs):
        # Increase the cache size limit for torch._dynamo to
        # accommodate videos with different numbers of frames.
        torch._dynamo.config.cache_size_limit = 1000

        self.compute_dtype = kwargs.pop("compute_dtype")
        self.output_dir = kwargs.pop("output_dir")
        self.timer = ContextTimer(self)
        self._last_step_end_time: float | None = None

        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.global_rank = int(os.environ.get("RANK", "0"))
        self.node_rank = int(os.environ.get("NODE_RANK", "0"))

        self.current_step = 0

        super().__init__(**kwargs)

        # Per-step sums of the component losses since the last log, averaged over ranks at log time.
        self.component_loss_sums: dict[str, torch.Tensor] = {}
        self.component_loss_steps = 0
        self.add_callback(_StepBoundaryTimer(self))

    def training_step(self, model, inputs, num_items_in_batch=None):
        # Wall-clock from previous step's on_step_end to entering this step's
        # training_step is dominated by dataloader fetch (worker -> main proc).
        # _last_step_end_time is set by _StepBoundaryTimer; first step is None.
        step_entered_at = time.time()
        if self._last_step_end_time is not None:
            self.log({"dataloader_wait_time": step_entered_at - self._last_step_end_time})

        start_time = time.time()

        with self.timer.with_label("training_step"):
            output = super().training_step(model, inputs)

        time_taken = time.time() - start_time
        print(
            f"Rank {self.global_rank} time taken for training_step {self.current_step}: {time_taken:.2f} seconds"
        )

        self.current_step += 1
        return output

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        with self.timer.with_label("model_forward"):
            outputs = model(inputs)
        # Accumulate the component losses (rgb/spatial/action); log() reports them with "loss".
        for key, value in outputs.items():
            if key.endswith("_loss"):
                value = value.detach().float()
                if key in self.component_loss_sums:
                    self.component_loss_sums[key] += value
                else:
                    self.component_loss_sums[key] = value.clone()
        self.component_loss_steps += 1

        loss = outputs["loss"]

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        # The periodic training log (the one carrying "loss") runs on every rank, so the component
        # losses are averaged over the same steps and ranks as "loss" and land in the same row.
        if "loss" in logs and self.component_loss_steps > 0:
            for key, total in sorted(self.component_loss_sums.items()):
                mean = total / self.component_loss_steps
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(mean, op=torch.distributed.ReduceOp.AVG)
                logs[key] = mean.item()
            self.component_loss_sums = {}
            self.component_loss_steps = 0
        super().log(logs, start_time)

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = transformers.Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            # DeepSpeed's optimizer wrapper expects 'bias_correction' in each param group;
            # HuggingFace Trainer's AdamW does not set it.
            if getattr(self.args, "deepspeed", None):
                for group in self.optimizer.param_groups:
                    group.setdefault("bias_correction", True)

        return self.optimizer

    def save_model(self, output_dir: Optional[str], _internal_call: bool):
        # Under ZeRO-2 every rank holds the full weights, and get_state_dict copies them to host
        # memory, so only the saving rank builds the state dict (8 copies of ~24 GB at once
        # exhausted host memory at the final save).
        if not self.args.should_save:
            return
        if self.is_deepspeed_enabled:
            state_dict = self.accelerator.get_state_dict(self.deepspeed)
        else:
            state_dict = self.model.state_dict()
        return self.model.save_pretrained(output_dir, state_dict=state_dict)

    def train(
        self,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
        **kwargs,
    ):
        """Correctly set self.state from checkpoint so get_train_dataloader can read from it."""
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(
                    f"No valid checkpoint found in output directory ({self.args.output_dir})"
                )

        if resume_from_checkpoint is not None:
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )
        return super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)

    def get_train_dataloader(self) -> DataLoader:
        """DataLoader over the sharded mixture (an IterableDataset: no sampler, the dataset
        shards and shuffles itself per rank/worker)."""
        train_dataset = self.train_dataset
        assert isinstance(train_dataset, ShardedLeRobotMixtureDataset), type(train_dataset)

        # During resume, don't skip the data
        self.args.ignore_data_skip = True
        curr_global_step = self.state.global_step
        print(f"Current global step: {curr_global_step}")
        if curr_global_step > 0:
            new_seed = train_dataset.seed + curr_global_step
            train_dataset.reset_seed(new_seed)
            print(
                f"Resetting seed to {new_seed}. Please note that this will make the experiment non-reproducible."
            )

        print("Creating custom train dataloader")
        data_collator = self.data_collator
        data_collator = self._get_collator_with_removed_columns(
            data_collator, description="training"
        )

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        # persistent_workers is only valid when num_workers > 0 (PyTorch raises otherwise)
        if self.args.dataloader_num_workers > 0:
            dataloader_params["persistent_workers"] = self.args.dataloader_persistent_workers

        return DataLoader(train_dataset, **dataloader_params)


class BaseExperiment:
    """Builds model, dataset and trainer from the composed Hydra config and runs training.

    Writes {output_dir}/experiment_cfg/{conf.yaml,metadata.json}; both are copied into every
    checkpoint and are what the inference policy loads (transforms, modality configs,
    normalization stats)."""

    def __init__(self, cfg: DictConfig):
        assert cfg.training_args.max_steps > 0, "max_steps must be > 0 for standarized evaluation"
        assert cfg.training_args.save_total_limit >= 5, "save_total_limit must be >= 5 for standarized evaluation"

        # Check if evaluation transforms are valid.
        assert cfg.transforms is not None, "Evaluation transforms are not provided."
        for tag, transform_cfg in cfg.transforms.items():
            try:
                # Check if the tag is a valid EmbodimentTag
                _ = EmbodimentTag(tag)
                # Check if the transform is a valid ComposedModalityTransform
                transform = instantiate(transform_cfg)
                assert isinstance(transform, ComposedModalityTransform), f"{transform=}"
            except Exception as e:
                raise ValueError(f"Evaluation transform {tag} is invalid: {e}")

        # Instantiate the training arguments.
        cfg.training_args.output_dir = cfg.training_args.output_dir.rstrip("/")
        cfg.training_args.run_name = cfg.training_args.output_dir.split("/")[-1]
        print(f"Run name: {cfg.training_args.run_name}")
        training_args = instantiate(cfg.training_args)
        set_seed(training_args.seed)

        # Set the environment variables for wandb.
        if "WANDB_PROJECT" not in os.environ:
            os.environ["WANDB_PROJECT"] = cfg.wandb_project
        if "WANDB_RUN_ID" not in os.environ:
            runtime_id = os.environ.get("RUNTIME_ID", None)
            """If a RUNTIME_ID is available in the environment, we use it as the wandb id,
            which will allow to display the evaluation results and the training results
            in the same wandb run. Otherwise, we create a new run."""
            if runtime_id:
                os.environ["WANDB_RUN_ID"] = runtime_id
        os.environ["WANDB_DIR"] = training_args.output_dir

        # Create the experiment config directory.
        output_dir = Path(training_args.output_dir)
        exp_cfg_dir = output_dir / "experiment_cfg"
        exp_cfg_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, exp_cfg_dir / "conf.yaml", resolve=True)

        wandb_config_file = output_dir / "wandb_config.json"
        with open(wandb_config_file, "w") as f:
            json.dump(
                {
                    "project": os.environ.get("WANDB_PROJECT", ""),
                    "run_id": os.environ.get("WANDB_RUN_ID", ""),
                },
                f,
            )

        # Check if we are resuming training.
        resume_path, continue_training = get_checkpoint_path(training_args.output_dir)
        if not continue_training:
            print(f"Models is ready under {training_args.output_dir}. Skip training.")
            exit(0)
        if resume_path:
            print(f"Resuming training from {resume_path}")
            resume_from_checkpoint = True
        else:
            # First time training.
            resume_from_checkpoint = False

        # Instantiate the model.
        model = self.create_model(cfg, training_args)

        # Make sure model_dtype and training_args dtype are compatible.
        compute_dtype = dtype_from_string(model.config.model_dtype)

        # Create the train dataset.
        # Dump the metadata; necessary for policy to normalize the input and unnormalize the output
        train_dataset = self.create_train_dataset(cfg, model)
        print("Using dataset:")
        print(train_dataset)
        assert (
            train_dataset.merged_metadata is not None
        ), "You must set metadata_config.merge=true in order to save the metadata."

        metadata_save_path = exp_cfg_dir / "metadata.json"
        with open(metadata_save_path, "w") as f:
            json.dump(
                {k: v.model_dump(mode="json") for k, v in train_dataset.merged_metadata.items()},
                f,
                indent=4,
            )
        print("Successfully dumped metadata")

        data_collator = instantiate(cfg.data_collator)
        trainer = self.create_trainer(
            cfg=cfg,
            exp_cfg_dir=exp_cfg_dir,
            model=model,
            training_args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            compute_dtype=compute_dtype,
        )
        self.cfg = cfg
        self.exp_cfg_dir = exp_cfg_dir
        self.training_args = training_args
        self.resume_from_checkpoint = resume_from_checkpoint
        self.train_dataset = train_dataset
        self.trainer = trainer

    def create_model(self, cfg, training_args):
        model = instantiate(cfg.model)

        if cfg.pretrained_model_path is not None:
            # Warm start from a checkpoint (full weights, strict=False so a module whose shape
            # changed, e.g. state registers, keeps its fresh init). Optimizer, scheduler and
            # step counter start fresh; resuming a run goes through get_checkpoint_path.
            mprint(f"Loading pretrained weights from: {cfg.pretrained_model_path}")
            from safetensors.torch import load_file

            ckpt_dir = cfg.pretrained_model_path
            safetensors_index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
            safetensors_path = os.path.join(ckpt_dir, "model.safetensors")

            if os.path.exists(safetensors_index_path):
                with open(safetensors_index_path, 'r') as f:
                    index = json.load(f)
                for shard_file in sorted(set(index["weight_map"].values())):
                    shard_path = os.path.join(ckpt_dir, shard_file)
                    mprint(f"Loading shard: {shard_path}")
                    shard_state_dict = load_file(shard_path)
                    model.load_state_dict(shard_state_dict, strict=False)
                    del shard_state_dict
                    gc.collect()
            elif os.path.exists(safetensors_path):
                state_dict = load_file(safetensors_path)
                model.load_state_dict(state_dict, strict=False)
            else:
                raise FileNotFoundError(
                    f"No weights found at '{ckpt_dir}'. "
                    "Expected 'model.safetensors' or 'model.safetensors.index.json'."
                )

            mprint("Successfully loaded pretrained weights")

        model.config.resume_path = model.config._name_or_path = training_args.output_dir
        mprint(f"{model}\n")
        return model

    def create_train_dataset(self, cfg, model):
        assert torch.distributed.is_initialized()
        train_dataset = instantiate(cfg.train_dataset)
        return train_dataset

    def create_trainer(
        self,
        cfg,
        exp_cfg_dir,
        model,
        training_args,
        train_dataset,
        data_collator,
        compute_dtype,
    ):
        # Set the gradient accumulation steps.
        if cfg.global_batch_size is not None:
            global_bs = cfg.global_batch_size
            bs = training_args.per_device_train_batch_size
            grad_acc = compute_grad_accum_to_match_global_bs(global_bs, bs)
            training_args.gradient_accumulation_steps = grad_acc
            print(
                f"Set global batch size to {global_bs}, set gradient accumulation steps to {grad_acc}"
            )
        else:
            warnings.warn(
                "global_batch_size is not set. This is fine for debugging, but please set this for real experiments."
            )

        # Instantiate the partial trainer.
        trainer_partial = instantiate(
            cfg.trainer,
            model=model,
            output_dir=training_args.output_dir,
            train_dataset=train_dataset,
            eval_dataset=None,
            compute_dtype=compute_dtype,
        )

        # Fully instantiate the trainer with dataclasses instances.
        trainer = trainer_partial(data_collator=data_collator, args=training_args)
        train_dl_len = len(trainer.get_train_dataloader())

        # Save the total training steps in the config.
        with open_dict(cfg):
            cfg.total_training_steps = train_dl_len * cfg.training_args.num_train_epochs

        # Save config.
        OmegaConf.save(cfg, exp_cfg_dir / "conf.yaml", resolve=True)

        trainer.add_callback(CheckpointFormatCallback(exp_cfg_dir=exp_cfg_dir))

        loss_log_path = str(Path(training_args.output_dir) / "loss_log.jsonl")
        trainer.add_callback(LossLoggerCallback(output_path=loss_log_path))

        mprint(
            f"train dataloader length: {train_dl_len}\n"
            f"eval dataloader length: no eval dataloader\n"
            f"train dataset length: {len(trainer.train_dataset)}\n"
            f"GPU memory before training: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024} GB",
            flush=True,
        )
        return trainer

    def train(self):
        import wandb

        class _DatasetInfoCallback(TrainerCallback):
            def __init__(self, info: dict):
                self._info = info
                self._done = False

            def on_step_begin(self, args, state, control, **kwargs):
                if self._done or state.global_step != 0:
                    return
                if wandb.run is not None:
                    wandb.run.summary.update(self._info)
                self._done = True

        # Datasets truncated by max_trajectories record what was kept; surface it in the run summary.
        dataset_info = {}
        for ds in self.train_dataset.datasets:
            if hasattr(ds, "_truncation_info"):
                dataset_info.update(ds._truncation_info)
        if dataset_info:
            self.trainer.add_callback(_DatasetInfoCallback(dataset_info))

        # Start training.
        self.trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)
        self.trainer.save_state()
        safe_save_model_for_hf_trainer(
            trainer=self.trainer, output_dir=self.training_args.output_dir
        )
