# Copyright (c) 2025 ASLP-LAB
#               2025 Ziqian Ning   (ningziqian@mail.nwpu.edu.cn)
#               2025 Huakang Chen  (huakang@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" This implementation is adapted from github repo:
    https://github.com/SWivid/F5-TTS.
"""

from __future__ import annotations

import os
import gc
from tqdm import tqdm
import wandb

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from diffrhythm.dataset.dataset import DiffusionDataset

from torch.utils.data import DataLoader

from ema_pytorch import EMA

from diffrhythm.model import CFM
from diffrhythm.model.utils import exists, default

class Trainer:
    def __init__(
        self,
        model: CFM,
        args,
        epochs,
        learning_rate,
        num_warmup_updates=20000,
        save_per_updates=1000,
        checkpoint_path=None,
        batch_size=32,
        batch_size_type: str = "sample",
        max_samples=32,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        noise_scheduler: str | None = None,
        duration_predictor: torch.nn.Module | None = None,
        wandb_project="test_e2-tts",
        wandb_run_name="test_run",
        wandb_resume_id: str = None,
        last_per_steps=None,
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        bnb_optimizer: bool = False,
        reset_lr: bool = False,
        use_style_prompt: bool = False,
        grad_ckpt: bool = False
    ):
        self.args = args

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False, )

        logger = "wandb" if wandb.api.api_key else None
        
        self.accelerator = Accelerator(
            log_with=logger,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=grad_accumulation_steps,
            **accelerate_kwargs,
        )

        if logger == "wandb":
            if exists(wandb_resume_id):
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name, "id": wandb_resume_id}}
            else:
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name}}
            self.accelerator.init_trackers(
                project_name=wandb_project,
                init_kwargs=init_kwargs,
                config={
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "num_warmup_updates": num_warmup_updates,
                    "batch_size": batch_size,
                    "batch_size_type": batch_size_type,
                    "max_samples": max_samples,
                    "grad_accumulation_steps": grad_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                    "gpus": self.accelerator.num_processes,
                    "noise_scheduler": noise_scheduler,
                },
            )

        self.precision = self.accelerator.state.mixed_precision
        self.precision = self.precision.replace("no", "fp32")

        self.model = model

        if self.is_main:
            self.ema_model = EMA(model, include_online_model=False, **ema_kwargs)

            self.ema_model.to(self.accelerator.device)
            if self.accelerator.state.distributed_type in ["DEEPSPEED", "FSDP"]:
                self.ema_model.half()

        self.epochs = epochs
        self.num_warmup_updates = num_warmup_updates
        self.save_per_updates = save_per_updates
        self.last_per_steps = default(last_per_steps, save_per_updates * grad_accumulation_steps)
        self.checkpoint_path = default(checkpoint_path, "ckpts/test_e2-tts")

        self.max_samples = max_samples
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm

        self.noise_scheduler = noise_scheduler

        self.duration_predictor = duration_predictor

        self.reset_lr = reset_lr

        self.use_style_prompt = use_style_prompt
        
        self.grad_ckpt = grad_ckpt

        if bnb_optimizer:
            import bitsandbytes as bnb

            self.optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = AdamW(model.parameters(), lr=learning_rate)

        if self.accelerator.state.distributed_type == "DEEPSPEED":
            self.accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = batch_size
        
        self.get_dataloader()
        self.get_scheduler()

        self.model, self.optimizer, self.scheduler, self.train_dataloader = self.accelerator.prepare(self.model, self.optimizer, self.scheduler, self.train_dataloader)

    def get_scheduler(self):
        warmup_steps = (
            self.num_warmup_updates * self.accelerator.num_processes
        )  # consider a fixed warmup steps while using accelerate multi-gpu ddp
        total_steps = len(self.train_dataloader) * self.epochs / self.grad_accumulation_steps
        decay_steps = total_steps - warmup_steps
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_steps)
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_steps]
        )

    def get_constant_scheduler(self):
        total_steps = len(self.train_dataloader) * self.epochs / self.grad_accumulation_steps
        self.scheduler = ConstantLR(self.optimizer, factor=1, total_iters=total_steps)

    def get_dataloader(self):
        print(self.args)
        dd = DiffusionDataset(self.args.file_path, self.args.max_frames, self.args.min_frames, self.args.sampling_rate, self.args.downsample_rate, self.precision)
        self.train_dataloader = DataLoader(
            dataset=dd,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=dd.custom_collate_fn,
            persistent_workers=True
        )


    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save_checkpoint(self, step, last=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict=self.accelerator.unwrap_model(self.optimizer).state_dict(),
                ema_model_state_dict=self.ema_model.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                step=step,
            )
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            if last:
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_last.pt")
                print(f"Saved last checkpoint at step {step}")
            else:
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_{step}.pt")

    def load_checkpoint(self):
        if (
            not exists(self.checkpoint_path)
            or not os.path.exists(self.checkpoint_path)
            or not os.listdir(self.checkpoint_path)
        ):
            return 0

        self.accelerator.wait_for_everyone()
        if "model_last.pt" in os.listdir(self.checkpoint_path):
            latest_checkpoint = "model_last.pt"
        else:
            latest_checkpoint = sorted(
                [f for f in os.listdir(self.checkpoint_path) if f.endswith(".pt")],
                key=lambda x: int("".join(filter(str.isdigit, x))),
            )[-1]
        
        checkpoint = torch.load(f"{self.checkpoint_path}/{latest_checkpoint}", map_location="cpu")

        if self.is_main:
            ema_dict = self.ema_model.state_dict()
            ema_checkpoint_dict = checkpoint["ema_model_state_dict"]

            filtered_ema_dict = {
                k: v for k, v in ema_checkpoint_dict.items()
                if k in ema_dict and ema_dict[k].shape == v.shape 
            }

            self.ema_model.load_state_dict(filtered_ema_dict, strict=False)

        model_dict = self.accelerator.unwrap_model(self.model).state_dict()
        checkpoint_model_dict = checkpoint["model_state_dict"]

        filtered_model_dict = {
            k: v for k, v in checkpoint_model_dict.items()
            if k in model_dict and model_dict[k].shape == v.shape  
        }

        self.accelerator.unwrap_model(self.model).load_state_dict(filtered_model_dict, strict=False)
        
        if "step" in checkpoint:
            if self.scheduler and not self.reset_lr:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            step = checkpoint["step"]
        else:
            step = 0

        del checkpoint
        gc.collect()
        print("Checkpoint loaded at step", step)
        return step

    def train(self, resumable_with_seed: int = None):
        train_dataloader = self.train_dataloader

        start_step = self.load_checkpoint()
        global_step = start_step

        if resumable_with_seed > 0:
            orig_epoch_step = len(train_dataloader)
            skipped_epoch = int(start_step // orig_epoch_step)
            skipped_batch = start_step % orig_epoch_step
            skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
        else:
            skipped_epoch = 0

        for epoch in range(skipped_epoch, self.epochs):
            self.model.train()
            if resumable_with_seed > 0 and epoch == skipped_epoch:
                progress_bar = tqdm(
                    skipped_dataloader,
                    desc=f"Epoch {epoch+1}/{self.epochs}",
                    unit="step",
                    disable=not self.accelerator.is_local_main_process,
                    initial=skipped_batch,
                    total=orig_epoch_step,
                    smoothing=0.15
                )
            else:
                progress_bar = tqdm(
                    train_dataloader,
                    desc=f"Epoch {epoch+1}/{self.epochs}",
                    unit="step",
                    disable=not self.accelerator.is_local_main_process,
                    smoothing=0.15
                )

            for batch in progress_bar:
                with self.accelerator.accumulate(self.model):
                    text_inputs = batch["lrc"]
                    mel_spec = batch["latent"].permute(0, 2, 1)
                    mel_lengths = batch["latent_lengths"]
                    style_prompt = batch["prompt"]
                    style_prompt_lens = batch["prompt_lengths"]
                    start_time = batch["start_time"]

                    loss, cond, pred = self.model(
                        mel_spec, text=text_inputs, lens=mel_lengths, noise_scheduler=self.noise_scheduler,
                        style_prompt=style_prompt,
                        style_prompt_lens=style_prompt_lens,
                        grad_ckpt=self.grad_ckpt, start_time=start_time
                    )
                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.is_main:
                    self.ema_model.update()

                global_step += 1

                if self.accelerator.is_local_main_process:
                    self.accelerator.log({"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]}, step=global_step)

                progress_bar.set_postfix(step=str(global_step), loss=loss.item())

                if global_step % (self.save_per_updates * self.grad_accumulation_steps) == 0:
                    self.save_checkpoint(global_step)

                if global_step % self.last_per_steps == 0:
                    self.save_checkpoint(global_step, last=True)

        self.save_checkpoint(global_step, last=True)

        self.accelerator.end_training()
