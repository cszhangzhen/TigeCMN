from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class ClassificationDataLoader(object):
    def __init__(self, train_file, label_file, state='user'):
        self.train_file = train_file
        self.label_file = label_file
        self.state = state

        self.train_users = []
        self.train_items = []
        self.delta_time = []
        self.user_interact_item_dict = dict()
        self.item_interact_user_dict = dict()

        self.user_num = 0
        self.item_num = 0
        self.attr_num = 0

        self.train_contents = None
        self.label = None

        self.gen_train_data()
        self.gen_label_data()

    def gen_train_data(self):
        f = open(self.train_file, 'r')
        lines = f.readlines()
        f.close()

        is_first_line = 1

        corpus = []

        for line in lines:
            if is_first_line:
                line = line.strip('\n\r').split('@::')
                self.user_num = int(line[0])
                self.item_num = int(line[1])
                is_first_line = 0
            else:
                line = line.strip('\n\r').split('@::')
                if self.state == 'user':
                    user_idx = int(line[0])
                    item_idx = int(line[1])
                else:
                    user_idx = int(line[1])
                    item_idx = int(line[0])
                time = eval(line[2])
                content = line[3]
                corpus.append(content)
                self.train_users.append(user_idx)
                self.train_items.append(item_idx)
                self.delta_time.append(time)

                if item_idx not in self.item_interact_user_dict:
                    self.item_interact_user_dict[item_idx] = dict()
                    self.item_interact_user_dict[item_idx][user_idx] = 1
                else:
                    if user_idx not in self.item_interact_user_dict[item_idx]:
                        self.item_interact_user_dict[item_idx][user_idx] = 1

                if user_idx not in self.user_interact_item_dict:
                    self.user_interact_item_dict[user_idx] = dict()
                    self.user_interact_item_dict[user_idx][item_idx] = 1
                else:
                    if item_idx not in self.user_interact_item_dict[user_idx]:
                        self.user_interact_item_dict[user_idx][item_idx] = 1

        vectorizer = CountVectorizer(stop_words='english', min_df=10)
        transformer = TfidfTransformer()
        self.train_contents = transformer.fit_transform(vectorizer.fit_transform(corpus))
        vocab = vectorizer.get_feature_names()
        self.attr_num = len(vocab)
        # print('The vocab length is: ' + str(len(vocab)))
        # print('The shape of attributes matrix: ' + str(self.train_contents.shape))

    def gen_label_data(self):
        f = open(self.label_file, 'r')
        lines = f.readlines()
        f.close()

        self.label = np.zeros(self.user_num, dtype=int)
        for line in lines:
            line = line.strip('\n\r').split(' ')
            user_idx = int(line[0])
            item_idx = int(line[1])
            self.label[user_idx] = item_idx


class Sampler(object):
    def __init__(self, data, batch_size, neg_sample):
        self.train_users = data.train_users
        self.train_items = data.train_items
        self.delta_time = data.delta_time
        self.user_num = data.user_num
        self.item_num = data.item_num
        self.train_contents = data.train_contents
        self.user_interact_item_dict = data.user_interact_item_dict
        self.item_interact_user_dict = data.item_interact_user_dict
        self.neg_sample = neg_sample
        self.batch_size = batch_size

    def gen_negative_item_sample(self, user):
        neg_samples = list()

        while len(neg_samples) < self.neg_sample:
            neg_item = np.random.randint(self.item_num)
            while neg_item in self.user_interact_item_dict[user] or neg_item in neg_samples:
                neg_item = np.random.randint(self.item_num)
            neg_samples.append(neg_item)

        return neg_samples

    def gen_negative_user_sample(self, item):
        neg_samples = list()

        while len(neg_samples) < self.neg_sample:
            neg_user = np.random.randint(self.user_num)
            while neg_user in self.item_interact_user_dict[item] or neg_user in neg_samples:
                neg_user = np.random.randint(self.user_num)
            neg_samples.append(neg_user)

        return neg_samples

    def generate_user_batch(self):
        offset_users = np.zeros(self.user_num + 1, dtype=np.int)
        offset_users[1:] = np.cumsum(np.unique(self.train_users, return_counts=True)[1])
        user_idx_arr = np.arange(len(offset_users) - 1)
        iters = np.arange(self.batch_size)
        maxiter = iters.max()
        start = offset_users[user_idx_arr[iters]]
        end = offset_users[user_idx_arr[iters] + 1]
        user_neg_items = dict()
        for user in range(self.user_num):
            user_neg_items[user] = self.gen_negative_item_sample(user)

        while True:
            minlen = (end - start).min()
            for i in range(minlen):
                users = np.array(self.train_users)[start + i]
                items = np.array(self.train_items)[start + i]
                delta_time = np.array(self.delta_time)[start + i]
                attrs = np.squeeze(np.array(self.train_contents[start + i].todense()))
                if len(attrs.shape) == 1:
                    attrs = np.expand_dims(attrs, axis=0)
                neg_items = list(map(user_neg_items.get, users))
                yield users, items, neg_items, delta_time, attrs

            start = start + minlen
            finished_mask = (end - start) < 1
            n_finished = finished_mask.sum()
            iters[finished_mask] = maxiter + np.arange(1, n_finished + 1)
            maxiter += n_finished
            valid_mask = (iters < len(offset_users) - 1)
            n_valid = valid_mask.sum()
            if n_valid == 0:
                break
            mask = finished_mask & valid_mask
            idx = user_idx_arr[iters[mask]]
            start[mask] = offset_users[idx]
            end[mask] = offset_users[idx + 1]
            iters = iters[valid_mask]
            start = start[valid_mask]
            end = end[valid_mask]

    def generate_item_batch(self):
        cnt = len(np.unique(self.train_items))
        offset_items = np.zeros(cnt + 1, dtype=np.int)
        offset_items[1:] = np.cumsum(np.unique(self.train_items, return_counts=True)[1])
        item_idx_arr = np.arange(len(offset_items) - 1)
        iters = np.arange(self.batch_size)
        maxiter = iters.max()
        start = offset_items[item_idx_arr[iters]]
        end = offset_items[item_idx_arr[iters] + 1]
        item_neg_users = dict()
        for item in range(self.item_num):
            item_neg_users[item] = self.gen_negative_user_sample(item)

        while True:
            minlen = (end - start).min()
            for i in range(minlen):
                users = np.array(self.train_users)[start + i]
                items = np.array(self.train_items)[start + i]
                delta_time = np.array(self.delta_time)[start + i]
                attrs = np.squeeze(np.array(self.train_contents[start + i].todense()))
                if len(attrs.shape) == 1:
                    attrs = np.expand_dims(attrs, axis=0)
                neg_users = list(map(item_neg_users.get, items))
                yield users, items, neg_users, delta_time, attrs

            start = start + minlen
            finished_mask = (end - start) < 1
            n_finished = finished_mask.sum()
            iters[finished_mask] = maxiter + np.arange(1, n_finished + 1)
            maxiter += n_finished
            valid_mask = (iters < len(offset_items) - 1)
            n_valid = valid_mask.sum()
            if n_valid == 0:
                break
            mask = finished_mask & valid_mask
            idx = item_idx_arr[iters[mask]]
            start[mask] = offset_items[idx]
            end[mask] = offset_items[idx + 1]
            iters = iters[valid_mask]
            start = start[valid_mask]
            end = end[valid_mask]
