import shlex
import subprocess
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--gpu')
parser.add_argument('--task')
parser.add_argument('--seed')
args = parser.parse_args()


model = 'AIRL'
mode = 'train'

if args.task in ['circle_large', 'rmnist']:
    domain_split_index_list = np.arange(15, 26, 1)
elif args.task == 'circle_hard':
    domain_split_index_list = np.arange(10, 16, 1)
elif args.task == 'yearbook':
    domain_split_index_list = np.arange(1971, 2010, 1)
else:
    raise ValueError

num_test_domain = 5

for domain_split_index in domain_split_index_list:
    with open('output/%s/stream/log_%s_%s_%d_%d.txt' %
              (args.task, model, args.seed, domain_split_index, num_test_domain), 'w') as f:
        subprocess.check_call(shlex.split(
            'python main_%s.py --model_name "%s" --gpu "%s" --seed %s --num_test_domain %d '
            '--domain_split_index %d --mode "%s"'
            % (args.task, model, args.gpu, args.seed, num_test_domain, domain_split_index, mode)), stdout=f)
