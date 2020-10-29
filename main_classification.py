from model import *
from utils import *
from evaluation import *
from tqdm import tqdm
import os
import argparse
import torch.optim as optim
import time

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', help='learning rate in training procedure', type=float, default=1e-4)
parser.add_argument('--user_batch_size', help='user batch size', type=int, default=512)
parser.add_argument('--item_batch_size', help='item batch size', type=int, default=25)
parser.add_argument('--embedding_size', help='embedding dimensions', type=int, default=64)
parser.add_argument('--attention_size', help='attention dimensions', type=int, default=16)
parser.add_argument('--nheads', help='number of attention heads', type=int, default=4)
parser.add_argument('--alpha', help='embedding trade off parameter', type=float, default=0.8)
parser.add_argument('--memory_size', help='number of rows in memory networks', type=int, default=64)
parser.add_argument('--neg_sample', help='negative sampling rate', type=int, default=5)
parser.add_argument('--weight_decay', help='regularizer parameter', type=float, default=0)
parser.add_argument('--epoch_num', help='total iterations in training procedure', type=int, default=20)

args = parser.parse_args()
print(args)

user_train_file = 'data/Delta_Time_DBLP_User.dat'
item_train_file = 'data/Delta_Time_DBLP_Item.dat'
test_file = 'data/DBLP_Label.dat'
user_data = ClassificationDataLoader(user_train_file, test_file, state='user')
item_data = ClassificationDataLoader(item_train_file, test_file, state='item')
user_sampler = Sampler(user_data, args.user_batch_size, args.neg_sample)
item_sampler = Sampler(item_data, args.item_batch_size, args.neg_sample)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':
    model = TigeCMN(args, user_data).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    epoch = 0
    start_time = time.time()
    while epoch < args.epoch_num:
        epoch += 1
        total_loss = 0
        model.user_memory_network.clear_memory()
        model.item_memory_network.clear_memory()

        for users, items, neg_users, delta_time, attrs in tqdm(item_sampler.generate_item_batch(), desc='Item memory module update'):
            optimizer.zero_grad()
            users = torch.from_numpy(np.array(users, dtype=np.int)).cuda()
            items = torch.from_numpy(np.array(items, dtype=np.int)).cuda()
            neg_users = torch.from_numpy(np.array(neg_users, dtype=np.int)).cuda()
            item_delta_time = torch.from_numpy(np.array(delta_time, dtype=np.float32)).cuda()
            attrs = torch.from_numpy(np.array(attrs, dtype=np.float32)).cuda()
            loss_i = model.forward(users, items, neg_users, None, None, item_delta_time, attrs, state='item_update') * (1.0 - args.alpha)
            loss_i.backward()
            optimizer.step()
            total_loss += loss_i.item()

        for users, items, neg_items, delta_time, attrs in tqdm(user_sampler.generate_user_batch(), desc='User memory module update'):
            optimizer.zero_grad()
            users = torch.from_numpy(np.array(users, dtype=np.int)).cuda()
            items = torch.from_numpy(np.array(items, dtype=np.int)).cuda()
            neg_items = torch.from_numpy(np.array(neg_items, dtype=np.int)).cuda()
            user_delta_time = torch.from_numpy(np.array(delta_time, dtype=np.float32)).cuda()
            attrs = torch.from_numpy(np.array(attrs, dtype=np.float32)).cuda()
            loss_u = model.forward(users, items, None, neg_items, user_delta_time, None, attrs, state='user_update') * args.alpha
            loss_u.backward()
            optimizer.step()
            total_loss += loss_u.item()

        first = 1
        X = None
        for i in range(user_data.user_num):
            user = torch.from_numpy(np.array([i], dtype=np.int)).cuda()
            user_embed = model.user_embeddings(user)
            merged_embedding = model.user_memory_network.gen_memory_embedding(user, user_embed).cpu().data.numpy()
            if first:
                X = np.array(merged_embedding)
                first = 0
            else:
                X = np.concatenate((X, np.array(merged_embedding)))
        macro_f1, micro_f1 = multiclass_node_classification_eval(X, user_data.label)
        end_time = time.time()
        print('epoch: %03d, loss: %f, macro_f1 = %f, micro_f1 = %f, time elpsed: %.2f' % (
        epoch, total_loss, macro_f1, micro_f1, end_time - start_time))

    first = 1
    X = None
    for i in range(user_data.user_num):
        user = torch.from_numpy(np.array([i], dtype=np.int)).cuda()
        user_embed = model.user_embeddings(user)
        merged_embedding = model.user_memory_network.gen_memory_embedding(user, user_embed).cpu().data.numpy()
        if first:
            X = np.array(merged_embedding)
            first = 0
        else:
            X = np.concatenate((X, np.array(merged_embedding)))

    print('Repeat 10 times for node classification with random split...')
    node_multiclass_classification(X, user_data.label)
    end_time = time.time()
    print('Final time elpsed: %.2f' % (end_time - start_time))
