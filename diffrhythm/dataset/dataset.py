# Copyright (c) 2025 ASLP-LAB
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

import torch
import random

class DiffusionDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, max_frames=2048, min_frames=512, sampling_rate=44100, downsample_rate=2048, precision='fp16'):

        self.max_frames = max_frames
        self.min_frames = min_frames
        self.sampling_rate = sampling_rate
        self.downsample_rate = 2048
        self.max_secs = max_frames / (sampling_rate / downsample_rate)
        
        self.file_path = file_path
        with open(file_path, 'r') as f:
            self.file_lst = [line.strip() for line in f.readlines()]

        self.pad_token_id = 0
        self.comma_token_id = 1
        self.period_token_id = 2
        self.start_token_id = 355

        if precision == 'fp16':
            self.feature_dtype = torch.float16
        elif precision == 'bf16':
            self.feature_dtype = torch.bfloat16
        elif precision == 'fp32':
            self.feature_dtype = torch.float32

        random.seed(42)
        random.shuffle(self.file_lst)

    def load_item(self, item, field):
        try:
            item, reader_idx = item[field]
            item = self.lance_connections[reader_idx].get_datas_by_rowids([item._rowid])[0]
        except Exception as e:
            return None
        return item
    
    def get_triple(self, item):
        utt, lrc_path, latent_path, style_path = item.split("|")

        time_lrc = torch.load(lrc_path, map_location='cpu')
        input_times = time_lrc['time']
        input_lrcs = time_lrc['lrc']
        lrc_with_time = list(zip(input_times, input_lrcs))
        
        latent = torch.load(latent_path, map_location='cpu') # [b, d, t]
        latent = latent.squeeze(0)
        
        prompt = torch.load(style_path, map_location='cpu') # [b, d]
        prompt = prompt.squeeze(0)
        
        max_start_frame = max(0, latent.shape[-1] - self.max_frames)
        start_frame = random.randint(0, max_start_frame)
        start_time = start_frame * self.downsample_rate / self.sampling_rate
        normalized_start_time = start_frame / latent.shape[-1]
        latent = latent[:, start_frame:]
    
        lrc_with_time = [(time_start - start_time, line) for (time_start, line) in lrc_with_time if (time_start - start_time) >= 0] # empty for pure music
        lrc_with_time = [(time_start, line) for (time_start, line) in lrc_with_time if time_start < self.max_secs] # drop time longer than max_secs

        if len(lrc_with_time) >= 1:
            latent_end_time = lrc_with_time[-1][0]
        else:
            raise
        
        if self.max_frames == 2048:
            lrc_with_time = lrc_with_time[:-1] if len(lrc_with_time) >= 1 else lrc_with_time # drop last, can be empty

        lrc = torch.zeros((self.max_frames,), dtype=torch.long)
        
        tokens_count = 0
        last_end_pos = 0
        for time_start, line in lrc_with_time:
            tokens = [token if token != self.period_token_id else self.comma_token_id for token in line] + [self.period_token_id]
            tokens = torch.tensor(tokens, dtype=torch.long)
            num_tokens = tokens.shape[0]

            gt_frame_start = int(time_start * self.sampling_rate / self.downsample_rate)
            
            frame_shift = 0

            frame_start = max(gt_frame_start - frame_shift, last_end_pos)
            frame_len = min(num_tokens, self.max_frames - frame_start)

            lrc[frame_start:frame_start + frame_len] = tokens[:frame_len]

            tokens_count += num_tokens
            last_end_pos = frame_start + frame_len        

        latent = latent[:, :int(latent_end_time * self.sampling_rate / self.downsample_rate)]

        latent = latent.to(self.feature_dtype)
        prompt = prompt.to(self.feature_dtype)

        return prompt, lrc, latent, normalized_start_time

    def __getitem__(self, index):
        idx = index
        while True:
            try:
                prompt, lrc, latent, start_time = self.get_triple(self.file_lst[idx])
                if latent.shape[-1] < self.min_frames: # Too short
                    raise
                item = {'prompt': prompt, "lrc": lrc, "latent": latent, "start_time": start_time}
                return item
            except Exception as e:
                idx = random.randint(0, self.__len__() - 1)
                continue

    def __len__(self):
        return len(self.file_lst)

    def custom_collate_fn(self, batch):
        latent_list = [item['latent'] for item in batch]
        prompt_list = [item['prompt'] for item in batch]
        lrc_list = [item['lrc'] for item in batch]
        start_time_list = [item['start_time'] for item in batch]

        latent_lengths = torch.LongTensor([latent.shape[-1] for latent in latent_list])
        prompt_lengths = torch.LongTensor([prompt.shape[-1] for prompt in prompt_list])
        lrc_lengths = torch.LongTensor([lrc.shape[-1] for lrc in lrc_list])

        max_prompt_length = prompt_lengths.amax()

        padded_prompt_list = []
        for prompt in prompt_list:
            padded_prompt = torch.nn.functional.pad(prompt, (0, max_prompt_length - prompt.shape[-1]))
            padded_prompt_list.append(padded_prompt)

        padded_latent_list = []
        for latent in latent_list:
            padded_latent = torch.nn.functional.pad(latent, (0, self.max_frames - latent.shape[-1]))
            padded_latent_list.append(padded_latent)

        padded_start_time_list = []
        for start_time in start_time_list:
            padded_start_time = start_time
            padded_start_time_list.append(padded_start_time)

        prompt_tensor = torch.stack(padded_prompt_list)
        lrc_tensor = torch.stack(lrc_list)
        latent_tensor = torch.stack(padded_latent_list)
        start_time_tensor = torch.tensor(padded_start_time_list)

        return {'prompt': prompt_tensor, 'lrc': lrc_tensor, 'latent': latent_tensor, \
                "prompt_lengths": prompt_lengths, "lrc_lengths": lrc_lengths, "latent_lengths": latent_lengths, \
                "start_time": start_time_tensor}


if __name__ == "__main__":
    dd = DiffusionDataset("train.scp", 2048, 512)
    x = dd[0]
    import pdb; pdb.set_trace()
    print(x)