import numpy as np
from torch.utils.data import Dataset
import os
import pickle


class MultiDomainDatasetTrain(Dataset):
    def __init__(self, data_dir, dataset, domain_split_index):
        assert dataset in ['train', 'val', 'test_id', 'test_od']
        data = [self._read_pickle(os.path.join(data_dir, str(i), 'data.pkl')) for i in range(30)]
        self.mtdt = self._read_pickle(os.path.join(data_dir, 'metadata.pkl'))
        data = data[:domain_split_index]
        img = [np.expand_dims(dt['img'][self.mtdt[i]['idx_%s' % dataset.replace('_id', '')]], axis=1)
                              for i, dt in enumerate(data)]
        lb = [dt['lb'][self.mtdt[i]['idx_%s' % dataset.replace('_id', '')]].astype(int)
                             for i, dt in enumerate(data)]
        d_lb = [dt['d_lb'][self.mtdt[i]['idx_%s' % dataset.replace('_id', '')]]
                               for i, dt in enumerate(data)]
        self.data = {'img': img, 'lb': lb, 'd_lb': d_lb}
        self.num_domain = len(self.data['lb'])
        self.num_sample_per_domain = [len(self.data['lb'][i]) for i in range(self.num_domain)]

    def __len__(self):
        return max(self.num_sample_per_domain)

    def __getitem__(self, idx):
        img = np.vstack([np.expand_dims(self.data['img'][i][idx % self.num_sample_per_domain[i]], axis=0) for i in range(self.num_domain)])
        lb = np.vstack([self.data['lb'][i][idx % self.num_sample_per_domain[i]] for i in range(self.num_domain)])
        d_lb = np.vstack([self.data['d_lb'][i][idx % self.num_sample_per_domain[i]] for i in range(self.num_domain)])
        return {'img': img, 'lb': lb, 'd_lb': d_lb}

    def _read_pickle(self, data_file):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        return data


class SingleDomainDatasetTest(Dataset):
    def __init__(self, data_dir, current_year, num_test_domain):
        domain_split_index = current_year + 1
        data = self._read_pickle(os.path.join(data_dir, str(domain_split_index + num_test_domain), 'data.pkl'))
        self.mtdt = self._read_pickle(os.path.join(data_dir, 'metadata.pkl'))[domain_split_index + num_test_domain]
        img = np.expand_dims(data['img'][self.mtdt['idx_train']], axis=1)
        lb = data['lb'][self.mtdt['idx_train']]
        d_lb = data['d_lb'][self.mtdt['idx_train']]
        self.data = {'img': img, 'lb': lb, 'd_lb': d_lb}
        self.domain_dict_cont = dict(zip(list(range(30)), list(np.linspace(0, 1, 30, dtype=np.float32))))

    def __len__(self):
        return len(self.data['lb'])

    def __getitem__(self, idx):
        img = self.data['img'][idx]
        lb = self.data['lb'][idx]
        d_lb = self.data['d_lb'][idx]
        d_lb_cont = self.domain_dict_cont[d_lb]
        return {'img': img, 'lb': lb, 'd_lb': d_lb, 'd_lb_cont': d_lb_cont}

    def _read_pickle(self, data_file):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        return data


def calculate_score_per_domain(label, predict, d_label):
    acc_list = []
    score = (label == predict)
    for d_lb in sorted(list(set(d_label))):
        acc = np.mean(score[d_label == d_lb])
        acc_list.append(acc)
    return acc_list
