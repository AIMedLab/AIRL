import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os


class MultiDomainDatasetTrain(Dataset):
    def __init__(self, data_dir, dataset, domain_split_index):
        assert dataset in ['train', 'val', 'test_id', 'test_od']
        start_domain = 1930
        mtdt = self._read_metadata(os.path.join(data_dir, 'metadata.csv'))
        if dataset in ['train', 'val']:
            self.mtdt = [mtdt[(mtdt['year'] == i) & (mtdt['dataset'] == dataset)] for i in range(start_domain, domain_split_index)]
        elif dataset == 'test_id':
            self.mtdt = [mtdt[(mtdt['year'] == i) & (mtdt['dataset'] == 'test')] for i in range(start_domain, domain_split_index)]
        self.mtdt = [self.mtdt[i].to_numpy() for i in range(len(self.mtdt))]
        self.data_dir = data_dir
        self.dataset = dataset
        self.label_dict = {'F': 0, 'M': 1}
        self.domain_dict = dict(zip(list(range(start_domain, domain_split_index)),
                                    list(range(domain_split_index - start_domain))))
        self.num_domain = len(self.mtdt)
        self.num_sample_per_domain = [len(self.mtdt[i]) for i in range(self.num_domain)]

    def __len__(self):
        return max(self.num_sample_per_domain)

    def __getitem__(self, idx):
        d_lb = [self.mtdt[i][idx % self.num_sample_per_domain[i], 1] for i in range(self.num_domain)]
        lb = [self.mtdt[i][idx % self.num_sample_per_domain[i], 2] for i in range(self.num_domain)]
        img_path = [os.path.join(self.data_dir, str(d_lb[i]), str(lb[i]), self.mtdt[i][idx % self.num_sample_per_domain[i], 0] + '.npy') for i in range(self.num_domain)]
        img = [np.load(i).transpose((2, 0, 1)) for i in img_path]
        img = np.asarray(img, dtype=np.float32)
        lb = np.array([self.label_dict[i] for i in lb])
        d_lb = np.array([self.domain_dict[i] for i in d_lb])
        return {'img': img, 'lb': lb, 'd_lb': d_lb}

    def _read_metadata(self, metadata_file):
        metadata = pd.read_csv(metadata_file)
        return metadata

    def shuffle_data(self):
        idx_list = [np.arange(len(mtdt)) for mtdt in self.mtdt]
        [np.random.shuffle(idx) for idx in idx_list]
        self.mtdt = [mtdt[idx] for idx, mtdt in zip(idx_list, self.mtdt)]



class SingleDomainDatasetTest(Dataset):
    def __init__(self, data_dir, current_year, num_test_domain):
        start_domain = 1930
        end_domain = 2013
        mtdt = self._read_metadata(os.path.join(data_dir, 'metadata.csv'))
        domain_split_index = current_year + 1
        self.mtdt = mtdt[(mtdt['year'] == (domain_split_index + num_test_domain))]
        self.mtdt = self.mtdt.to_numpy()
        self.data_dir = data_dir
        self.label_dict = {'F': 0, 'M': 1}
        self.domain_dict_cont = dict(zip(list(range(start_domain, end_domain + 1)),
                                         list(np.linspace(0, 1, end_domain + 1 - start_domain, dtype=np.float32))))

    def __len__(self):
        return len(self.mtdt)

    def __getitem__(self, idx):
        d_lb = self.mtdt[idx, 1]
        lb = self.mtdt[idx, 2]
        img_path = os.path.join(self.data_dir, str(d_lb), str(lb), self.mtdt[idx, 0] + '.npy')
        img = np.load(img_path).transpose((2, 0, 1))
        img = np.asarray(img, dtype=np.float32)
        lb = self.label_dict[lb]
        d_lb_cont = self.domain_dict_cont[d_lb]
        return {'img': img, 'lb': lb, 'd_lb': d_lb, 'd_lb_cont': d_lb_cont}

    def _read_metadata(self, metadata_file):
        metadata = pd.read_csv(metadata_file)
        return metadata


def calculate_score_per_domain(label, predict, d_label):
    acc_list = []
    score = (label == predict)
    for d_lb in sorted(list(set(d_label))):
        acc = np.mean(score[d_label == d_lb])
        acc_list.append(acc)
    return acc_list
