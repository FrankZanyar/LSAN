from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Dataset
import torch
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import csv


# 构建数据集
def split_dataset(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, shuffle=True,
                                                        random_state=2020)
    Xp_train = torch.from_numpy(np.array(X_train)).to(torch.float32)
    yp_train = torch.from_numpy(np.array(y_train)).to(torch.long)
    Xp_test = torch.from_numpy(np.array(X_test)).to(torch.float32)
    yp_test = torch.from_numpy(np.array(y_test)).to(torch.long)
    return Xp_train, yp_train, Xp_test, yp_test


class ImageDataSet(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.length = len(label)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.length


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = {}  # dict with list value
        # {783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (_, pid) in enumerate(self.data_source):
            if pid not in self.index_dic:
                self.index_dic[pid] = [index]
            else:
                self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]  # 每个pid对应的索引
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])  # 复制一份
            if len(idxs) < self.num_instances:  # 如果少于num_instances,随机选择
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)  # 打乱
            batch_idxs_dict[pid] = [idxs[i * self.num_instances: (i + 1) * self.num_instances] for i in
                                    range(len(idxs) // self.num_instances)]
        #             batch_idxs = []
        #             for idx in idxs:
        #                 batch_idxs.append(idx)
        #                 if len(batch_idxs) == self.num_instances:
        #                     batch_idxs_dict[pid].append(batch_idxs)
        #                     batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length

def loadF(path,image_num):
    Features=np.zeros([image_num,192,64])
    with open(path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        i=0
        for row in reader:
            sentense=np.zeros([192,64])
            word_count=0
            for n in range(0,len(row),64):
                k=row[n:n+64]
                sentense[word_count]=row[n:n+64]
                word_count+=1
            Features[i]=sentense
            i=i+1
    
    return Features