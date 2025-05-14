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

from importlib.resources import files

from diffrhythm.model import CFM, DiT, Trainer

from prefigure.prefigure import get_all_args
import json
import os

os.environ['OMP_NUM_THREADS']="1"
os.environ['MKL_NUM_THREADS']="1"

def main():
    args = get_all_args("config/default.ini")

    with open(args.model_config) as f:
        model_config = json.load(f)

    if model_config["model_type"] == "diffrhythm":
        wandb_resume_id = None
        model_cls = DiT

    model = CFM(
        transformer=model_cls(**model_config["model"], max_frames=args.max_frames),
        num_channels=model_config["model"]['mel_dim'],
        audio_drop_prob=args.audio_drop_prob,
        cond_drop_prob=args.cond_drop_prob,
        style_drop_prob=args.style_drop_prob,
        lrc_drop_prob=args.lrc_drop_prob,
        max_frames=args.max_frames
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    trainer = Trainer(
        model,
        args,
        args.epochs,
        args.learning_rate,
        num_warmup_updates=args.num_warmup_updates,
        save_per_updates=args.save_per_updates,
        checkpoint_path=f"ckpts/{args.exp_name}",
        grad_accumulation_steps=args.grad_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        wandb_project="diffrhythm-test",
        wandb_run_name=args.exp_name,
        wandb_resume_id=wandb_resume_id,
        last_per_steps=args.last_per_steps,
        bnb_optimizer=False,
        reset_lr=args.reset_lr,
        batch_size=args.batch_size,
        grad_ckpt=args.grad_ckpt
    )

    trainer.train(
        resumable_with_seed=args.resumable_with_seed,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()
