from tqdm import tqdm
from collections import defaultdict
import time
from utils import generate_rating_matrix_valid, generate_rating_matrix_test, neg_sample
from torch.utils.data import Dataset, DataLoader
import random
import copy 
import torch
from torch.utils.data import RandomSampler, SequentialSampler
import numpy as np
from collections import Counter
import math

class DataGenerator(object):

    def __init__(self, args):
        self.args = args
        self.dataset = args.dataset
        self.bs = args.train_batch_size
        self.data_file = args.data_path + args.dataset + ".txt"
        print("========dataset :===========", self.data_file)
        self.create_dataset()
       
    def get_user_seqs(self, data_file):
        lines = open(data_file).readlines()
        user_seq = []
        item_set = set()
        for line in lines:
            user, items = line.strip().split(" ", 1)
            items = items.split(" ")
            items = [int(item) for item in items]
            if len(items) >= 5:
                user_seq.append(items)
            item_set = item_set | set(items)
        max_item = max(item_set)
        num_users = len(lines)
        num_items = max_item + 2

        valid_rating_matrix = generate_rating_matrix_valid(user_seq, num_users, num_items)
        test_rating_matrix = generate_rating_matrix_test(user_seq, num_users, num_items)
    
        return user_seq, max_item, valid_rating_matrix, test_rating_matrix
  
    
    def create_dataset(self):
        '''Load train, validation, test dataset'''
        self.user_seq, self.max_item, self.valid_rating_matrix, self.test_rating_matrix = self.get_user_seqs(self.data_file)
        self.item_size = self.max_item + 2
        self.mask_id = self.max_item + 1
        self.args.item_size = self.item_size
        
        self.train_dataset = SASRecDataset(self.args, self.user_seq, data_type="train")
        self.valid_dataset = SASRecDataset(self.args, self.user_seq, data_type="valid")
        self.test_dataset = SASRecDataset(self.args, self.user_seq, data_type="test")
##################################
        # random.seed(42)
        # p = self.args.mlm_probability_train
        # num_samples = min(2000, len(self.train_dataset))  # 抽 2000 条看看就够
        # total_nonpad = 0           # 所有非 PAD token 数
        # total_masked = 0           # 被 mask 的非 PAD token 数
        # total_long = 0             # 所有非 PAD token 中属于长尾的数量
        # total_long_masked = 0      # 被 mask 的 token 中属于长尾的数量
        # for idx in range(num_samples):
        #     d = self.train_dataset[idx]
        #     input_ids = d["input_ids"].tolist()
        #     mask = d["masked_indices0"].tolist()  # 就是你的 mask_indices 输出
        #     for tok, m in zip(input_ids, mask):
        #         if tok == 0:
        #             continue  # 跳过 PAD
        #         total_nonpad += 1
        #         if tok in self.train_dataset.long_tail_items:
        #             total_long += 1
        #         if m:  # 被 mask
        #             total_masked += 1
        #             if tok in self.train_dataset.long_tail_items:
        #                 total_long_masked += 1
        # # 1) 整体 mask 比例
        # overall_mask_ratio = total_masked / total_nonpad if total_nonpad > 0 else 0.0
        # # 2) 长尾在“所有 token”中的比例
        # long_ratio_all = total_long / total_nonpad if total_nonpad > 0 else 0.0
        # # 3) 长尾在“被 mask 的 token”中的比例
        # long_ratio_masked = total_long_masked / total_masked if total_masked > 0 else 0.0
        # print(f"整体 mask 比例 (非 PAD): {overall_mask_ratio:.4f}  (期望 p = {p})")
        # print(f"全体 token 中长尾占比   : {long_ratio_all:.4f}")
        # print(f"被 mask 的 token 中长尾占比: {long_ratio_masked:.4f}")
###################################
        self.train_dataloader = self.make_train_dataloader()
        self.valid_dataloader = self.make_valid_dataloader()
        self.test_dataloader = self.make_test_dataloader()
        
        
        
    def make_train_dataloader(self):
        train_dataloader = DataLoader(self.train_dataset,
                                        sampler=RandomSampler(self.train_dataset),
                                        batch_size=self.args.train_batch_size,
                                        drop_last=False,
                                        num_workers=4
                                        ) 
     
        return train_dataloader
    
    
    def make_valid_dataloader(self):
        valid_dataloader = DataLoader(self.valid_dataset,
                                        sampler=SequentialSampler(self.valid_dataset),
                                        batch_size=self.args.test_batch_size,
                                        drop_last=True) 
        return valid_dataloader
        
       
    def make_test_dataloader(self):
        test_dataloader = DataLoader(self.test_dataset,
                                        sampler=SequentialSampler(self.test_dataset),
                                        batch_size=self.args.test_batch_size,
                                        drop_last=True) 
        return test_dataloader
    
    
class SASRecDataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, data_type="train"):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

        #new,计算频率
        item_counter = Counter()
        total_nonpad_tokens = 0
        for seq in user_seq:
            for token in seq:
                if token > 0: #如果不是padding
                    item_counter[token] += 1
                    total_nonpad_tokens += 1
        
        self.item_counter = item_counter
        self.total_nonpad_tokens = total_nonpad_tokens
        #计算一个长尾阈值
        freqs = list(item_counter.values())
        freqs.sort()
        if len(freqs) > 0:
            tail_k = max(1, int(len(freqs)*0.8)) #后80%视为长尾物品
            tail_threshold = freqs[tail_k - 1]
        else:
            tail_threshold = 0
        #频次小于阈值的item视为长尾items
        self.long_tail_items = {i for i, c in item_counter.items() if c <= tail_threshold}
        #
        tail_tokens = 0
        for token, c in item_counter.items():
            if token in self.long_tail_items:
                tail_tokens += c
        self.tail_ratio = tail_tokens / total_nonpad_tokens if total_nonpad_tokens>0 else 0.0
        
        # #valid 
        # print("总非PAD token 数：", self.total_nonpad_tokens)
        # print("不同 item 数：", len(self.item_counter))
        # print("长尾 item 数：", len(self.long_tail_items))
        # # 看看频次最小 & 最大的 item
        # freqs = sorted(self.item_counter.values())
        # print("最小频次：", freqs[0], "  中位数：", freqs[len(freqs)//2], "  最大频次：", freqs[-1])
        # tail_freqs = [self.item_counter[i] for i in self.long_tail_items]
        # print("长尾 item 频次范围：", min(tail_freqs), " ~ ", max(tail_freqs)) 


        
    def mask_input_ids(self, input_ids, p): #frequency-aware 在这里改
        """
        p:表示整体的mask比例，应该是
        """
        mask_indices = []
        # for token in input_ids:
        #     if token == 0:
        #         mask_indices.append(0)  # 元素为0的位置不参与mask
        #     else:
        #         mask_indices.append(1 if random.random() < p else 0)
        # return mask_indices
        # 1. 为非 PAD 位置计算权重
        weights = []
        indices = []   # 记录非 PAD 的索引
        tail_factor = getattr(self.args, "tail_factor", 3.0)  # 长尾放大倍数（长尾比前20%优先多少倍），==1则退化为随机

        for idx, token in enumerate(input_ids):
            if token == 0:
                continue  # PAD 不参与
            if token in self.long_tail_items:
                w = tail_factor 
            else:
                w = 1.0
            weights.append(w)
            indices.append(idx)

        # 没有非 PAD token 的极端情况（一般不会发生）
        if len(weights) == 0:
            return [0] * len(input_ids)

        # 2. 计算当前序列的平均权重
        avg_w = sum(weights) / len(weights)

        # 3. 为每个位置计算 mask 概率，并采样
        mask_indices = [0] * len(input_ids)
        for idx, token in enumerate(input_ids):
            if token == 0:
                mask_indices[idx] = 0  # PAD 永远不 mask
            else:
                if token in self.long_tail_items:
                    w = tail_factor
                else:
                    w = 1.0

                # prob_i = p * w / avg_w，保证平均概率仍然为 p
                prob = p * w / avg_w
                # 稍微 clamp 一下，防止极端情况 >1
                if prob > 0.99:
                    prob = 0.99
                elif prob < 0.0:
                    prob = 0.0

                mask_indices[idx] = 1 if random.random() < prob else 0

        return mask_indices

    def _data_sample_rec_task(self, user_id, items, input_ids, target_pos, answer):
        # make a deep copy to avoid original sequence be modified
        copied_input_ids = copy.deepcopy(input_ids)
        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len :]
        target_pos = target_pos[-self.max_len :]
        target_neg = target_neg[-self.max_len :]
        attention_mask = [1.0 if token > 0 else 0.0 for token in input_ids]
        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len
        
        masked_indices0 = self.mask_input_ids(input_ids, self.args.mlm_probability_train) # random mask some items and return the indices. for diffusion train 选择
        # masked_indices1 = self.mask_input_ids(input_ids, self.args.mlm_probability) # random mask some items and return the indices.  for sas aug
        # masked_indices2 = self.mask_input_ids(input_ids, self.args.mlm_probability) # random mask some items and return the indices.  for sas aug
        

        dict1 = {"user_id": torch.tensor(user_id, dtype=torch.long),
                "input_ids":torch.tensor(input_ids, dtype=torch.long),
                "target_pos":torch.tensor(target_pos, dtype=torch.long), 
                "target_neg":torch.tensor(target_neg, dtype=torch.long), 
                "answer":torch.tensor(answer, dtype=torch.long), 
                "masked_indices0": torch.tensor(masked_indices0, dtype=torch.bool), 
                # "masked_indices1": torch.tensor(masked_indices1, dtype=torch.bool),  
                # "masked_indices2": torch.tensor(masked_indices2, dtype=torch.bool), 
                "attention_mask":torch.tensor(attention_mask, dtype=torch.float), 
             }
        return dict1
        

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test"}

        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = 0  # no use

        elif self.data_type == "valid":
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]

        return self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer)

    def __len__(self):
        return len(self.user_seq)
    
    
    

       
    