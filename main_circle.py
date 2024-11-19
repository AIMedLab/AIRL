import torch
import numpy as np
from torch.utils.data import DataLoader
from utils.circle_data import calculate_score_per_domain, MultiDomainDatasetTrain, SingleDomainDatasetTest
from tqdm import tqdm
from models.AIRL import FeatureExtractorCircle, Transformer, EvolveClassifier, AIRL
from datetime import datetime
import random
import argparse
import pickle
import os


start_time = datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=1024)
parser.add_argument('--max_epoch', type=int, default=150)
parser.add_argument('--domain_split_index', type=int)
parser.add_argument('--num_test_domain', type=int)
parser.add_argument('--model_name', type=str)
parser.add_argument('--gpu')
parser.add_argument('--seed', type=int)
parser.add_argument('--mode')

args = parser.parse_args()

SEED = args.seed

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

max_epoch = args.max_epoch
train_batch_size = args.train_batch_size
test_batch_size = args.test_batch_size
domain_split_index = args.domain_split_index
model_name = args.model_name
num_test_domain = args.num_test_domain

data_dir = 'datasets/processed_data/circle'

gpu = args.gpu
if gpu != 'osc':
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

out_dir = 'saved_model/circle/stream'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

num_label = 2

num_workers = 0

mode = args.mode

train_dataset = MultiDomainDatasetTrain(data_dir=data_dir, dataset='train', domain_split_index=domain_split_index)
val_dataset = MultiDomainDatasetTrain(data_dir=data_dir, dataset='val', domain_split_index=domain_split_index)
test_id_dataset = MultiDomainDatasetTrain(data_dir=data_dir, dataset='test_id', domain_split_index=domain_split_index)
test_od_dataset_list = [SingleDomainDatasetTest(data_dir=data_dir, current_year=domain_split_index-1, num_test_domain=k)
                        for k in range(num_test_domain)]
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=True)
test_id_dataloader = DataLoader(test_id_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers,
                                pin_memory=True)
test_od_dataloader_list = [DataLoader(test_od_dataset, batch_size=test_batch_size, shuffle=True,
                                      num_workers=num_workers, pin_memory=True)
                           for test_od_dataset in test_od_dataset_list]
feature_extractor = FeatureExtractorCircle(hidden_dim=32, output_dim=32, num_layer=4)
transformer = Transformer(input_dim=32, output_dim=32)
evolve_classifier = EvolveClassifier(hidden_dim=128, output_dim=(32 * 32 + 32 + 32 + 1))

model = AIRL(feature_extractor, transformer, evolve_classifier, n_output=1, ts_coef=0, lr=0.001, device=device)
model = model.to(device)

if mode == 'train':
    # training
    best_val_acc = float('-inf')
    val_acc_list = []
    for epoch in range(max_epoch):
        print("Iteration %d:" % (epoch + 1))
        model.train()
        epoch_loss = 0
        predict_list = np.empty(0)
        lb_list = np.empty(0)
        for i, batch in enumerate(tqdm(train_dataloader)):
            img = batch['img'].to(device)
            lb = batch['lb'].to(device).float()
            loss, predict = model(img, lb, num_label, 'train')
            predict_list = np.concatenate((predict_list, predict.cpu().detach().numpy()), axis=0)
            lb_list = np.concatenate((lb_list, lb.flatten().cpu().numpy()), axis=0)
            epoch_loss += loss.item()
        predict_list = 1 / (1 + np.exp(-predict_list))
        predict_list = np.where(predict_list < 0.5, 0, 1)
        acc = np.mean(lb_list == predict_list)
        print('Train loss: %.4f - Train ACC: %.4f' % (epoch_loss / (i + 1), acc))

        model.eval()
        with torch.no_grad():
            predict_list = np.empty(0)
            lb_list = np.empty(0)
            for i, batch in enumerate(tqdm(val_dataloader)):
                img = batch['img'].to(device)
                lb = batch['lb'].to(device).float()
                loss, predict = model(img, lb, num_label)
                predict_list = np.concatenate((predict_list, predict.cpu().detach().numpy()), axis=0)
                lb_list = np.concatenate((lb_list, lb.flatten().cpu().numpy()), axis=0)
            predict_list = 1 / (1 + np.exp(-predict_list))
            predict_list = np.where(predict_list < 0.5, 0, 1)
            acc = np.mean(lb_list == predict_list)
        val_acc_list.append(acc)
        print('Val ACC: %.4f' % acc)

        if best_val_acc < acc:
            best_val_acc = acc
            torch.save({'model_state_dict': model.state_dict()},
                       os.path.join(out_dir, '%s_%s_%d_%d.ckpt'
                                    % (model_name, args.seed, domain_split_index, num_test_domain)))

        model.eval()
        with torch.no_grad():
            predict_list = np.empty(0)
            lb_list = np.empty(0)
            for i, batch in enumerate(tqdm(test_id_dataloader)):
                img = batch['img'].to(device)
                lb = batch['lb'].to(device).float()
                loss, predict = model(img, lb, num_label)
                predict_list = np.concatenate((predict_list, predict.cpu().detach().numpy()), axis=0)
                lb_list = np.concatenate((lb_list, lb.flatten().cpu().numpy()), axis=0)
            predict_list = 1 / (1 + np.exp(-predict_list))
            predict_list = np.where(predict_list < 0.5, 0, 1)
            acc = np.mean(lb_list == predict_list)
        print('Test ID ACC: %.4f' % acc)

    best_val_epoch = np.argmax(val_acc_list)

score = {}

checkpoint = torch.load(os.path.join(out_dir, '%s_%s_%d_%d.ckpt'
                                     % (model_name, args.seed, domain_split_index,
                                        num_test_domain if mode == 'train' else 5)), map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

model.eval()

with torch.no_grad():
    predict_list = np.empty(0)
    lb_list = np.empty(0)
    for i, batch in enumerate(tqdm(val_dataloader)):
        img = batch['img'].to(device)
        lb = batch['lb'].to(device).float()
        loss, predict = model(img, lb, num_label)
        predict_list = np.concatenate((predict_list, predict.cpu().detach().numpy()), axis=0)
        lb_list = np.concatenate((lb_list, lb.flatten().cpu().numpy()), axis=0)
    predict_list = 1 / (1 + np.exp(-predict_list))
    predict_list = np.where(predict_list < 0.5, 0, 1)
    acc = np.mean(lb_list == predict_list)

if mode == 'train':
    print('(Epoch %d) Val ACC: %.4f' % (best_val_epoch + 1, acc))
else:
    print('Val ACC: %.4f' % acc)
score['val'] = acc

with torch.no_grad():
    predict_list = np.empty(0)
    lb_list = np.empty(0)
    for i, batch in enumerate(tqdm(test_id_dataloader)):
        img = batch['img'].to(device)
        lb = batch['lb'].to(device).float()
        loss, predict = model(img, lb, num_label)
        predict_list = np.concatenate((predict_list, predict.cpu().detach().numpy()), axis=0)
        lb_list = np.concatenate((lb_list, lb.flatten().cpu().numpy()), axis=0)
    predict_list = 1 / (1 + np.exp(-predict_list))
    predict_list = np.where(predict_list < 0.5, 0, 1)
    acc = np.mean(lb_list == predict_list)

if mode == 'train':
    print('(Epoch %d) Test ID ACC: %.4f' % (best_val_epoch + 1, acc))
else:
    print('Test ID ACC: %.4f' % acc)
score['test_id'] = acc

with torch.no_grad():
    predict_list = np.empty(0)
    lb_list = np.empty(0)
    d_lb_list = np.empty(0)
    for j, dataloader, in enumerate(test_od_dataloader_list):
        for i, batch in enumerate(tqdm(dataloader)):
            img = batch['img'].to(device)
            d_lb = batch['d_lb']
            lb = batch['lb'].to(device).float()
            predict = model.predict(img, domain_index=domain_split_index + j)
            predict_list = np.concatenate((predict_list, predict.cpu().detach().numpy()), axis=0)
            lb_list = np.concatenate((lb_list, lb.flatten().cpu().numpy()), axis=0)
            d_lb_list = np.concatenate((d_lb_list, d_lb.numpy()), axis=0)

predict_list = 1 / (1 + np.exp(-predict_list))
predict_list = np.where(predict_list < 0.5, 0, 1)
acc = np.mean(lb_list == predict_list)
acc_list = calculate_score_per_domain(lb_list, predict_list, d_lb_list)
if mode == 'train':
    print('(Epoch %d) Test OD ACC: %.4f' % (best_val_epoch + 1, acc))
else:
    print('Test OD ACC: %.4f' % acc)
score['test_od'] = acc
score['test_od_list'] = acc_list

out_dir = 'output/circle/stream'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
with open(os.path.join(out_dir, 'score_%s_%s_%d_%d.pkl'
                                % (model_name, args.seed, domain_split_index, num_test_domain)), 'wb') as f:
    pickle.dump(score, f)

end_time = datetime.now()

print('Running time: %s' % (end_time - start_time))
