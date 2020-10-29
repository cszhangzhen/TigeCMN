import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class AttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AttentionLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.key = Parameter(torch.zeros(in_dim, out_dim))
        nn.init.xavier_uniform_(self.key.data)
        self.query = Parameter(torch.zeros(in_dim, out_dim))
        nn.init.xavier_uniform_(self.query.data)
        self.value = Parameter(torch.zeros(in_dim, out_dim))
        nn.init.xavier_uniform_(self.value.data)

    def forward(self, x):
        key_vector = torch.matmul(x, self.key)
        query_vector = torch.matmul(x, self.query)
        value_vector = torch.matmul(x, self.value)
        mat = torch.matmul(query_vector, key_vector.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.in_dim, dtype=torch.float32)).cuda()
        weight = F.softmax(mat, dim=2)
        attention = torch.matmul(weight, value_vector)
        output = torch.mean(attention, dim=1)

        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, in_dim, out_dim, nheads):
        super(MultiHeadAttention, self).__init__()
        self.nheads = nheads
        self.attentions = [AttentionLayer(in_dim, out_dim) for _ in range(self.nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x):
        x = torch.cat([att(x) for att in self.attentions], dim=1)

        return x


class MemoryNetwork(nn.Module):
    def __init__(self, args, num_memories, key_matrix):
        super(MemoryNetwork, self).__init__()
        self.num_memories = num_memories

        self.embedding_size = args.embedding_size
        self.memory_size = args.memory_size
        self.nheads = args.nheads
        self.attention_size = args.attention_size

        self.key_matrix = key_matrix

        self.register_buffer('value_matrix', torch.zeros(self.num_memories, self.memory_size, self.embedding_size))

        self.erase_linear = nn.Linear(self.embedding_size, self.embedding_size)
        self.add_linear = nn.Linear(self.embedding_size, self.embedding_size)
        self.embed_merge_linear = nn.Linear(self.embedding_size * 2, self.embedding_size)

        self.mulithead_attention = MultiHeadAttention(self.embedding_size, self.attention_size, self.nheads)

    def forward(self, idx, users_embed, items_embed, negs, delta_time, attr_vecs, state='user_update'):
        # [batch_size, 1, embedding_size]
        extract_attr_vecs = torch.unsqueeze(attr_vecs, dim=1)
        # [batch_size, memory_size]
        inner_product = torch.sum(extract_attr_vecs * self.key_matrix, dim=2)
        # [batch_size, 1]
        attrs_length = torch.sqrt(torch.sum(attr_vecs ** 2, dim=1, keepdim=True))
        # [memory_size, 1]
        key_length = torch.sqrt(torch.sum(self.key_matrix ** 2, dim=1, keepdim=True))
        # [batch_size, memory_size]
        cosine_similarity = inner_product / (torch.matmul(attrs_length, key_length.t()))
        correlation_weight = F.softmax(cosine_similarity, dim=1)

        # [batch_size, memory_size, embedding_size]
        memory_matrix = self.value_matrix[idx]
        # [batch_size, memory_size, 1]
        correlation_weight = torch.unsqueeze(correlation_weight, dim=2)
        # erase and add memory values
        # [batch_size, embedding_size]
        erase_vector = self.erase_linear(attr_vecs)
        erase_vector = torch.sigmoid(erase_vector)
        add_vector = self.add_linear(attr_vecs)
        add_vector = torch.tanh(add_vector)
        # [batch_size, 1, embedding_size]
        extract_erase_vector = torch.unsqueeze(erase_vector, dim=1)
        # [batch_size, memory_size, embedding_size]
        erase_mul = extract_erase_vector * correlation_weight
        erase = memory_matrix * (torch.tensor(1.0) - erase_mul)
        # [batch_size, 1, embedding_size]
        extract_add_vector = torch.unsqueeze(add_vector, dim=1)
        # [batch_size, memory_size, embedding_size]
        add = extract_add_vector * correlation_weight
        updated_value = erase + add
        self.value_matrix.data[idx] = updated_value

        # multihead memory reading procedure
        # input size: [batch_size, memory_size, embedding_size]
        # output size: [batch_size, embedding_size]
        memory_embedding = self.mulithead_attention(updated_value)
        # [batch_size, embedding_size]
        memory_embedding = F.normalize(memory_embedding, p=2, dim=1)
        if state == 'user_update':
            merged_embedding = torch.tanh(self.embed_merge_linear(torch.cat([users_embed, memory_embedding], dim=1)))
            pos = torch.mean(torch.log(torch.sigmoid(torch.sum(merged_embedding * items_embed, dim=1)) + 1e-24))
            neg = torch.mean(torch.log(torch.tensor(1.0) - torch.sigmoid(
                torch.sum(torch.unsqueeze(merged_embedding, dim=1) * negs, dim=2)) + 1e-24))
            loss = - torch.mean(pos + neg)
        else:
            merged_embedding = torch.tanh(self.embed_merge_linear(torch.cat([items_embed, memory_embedding], dim=1)))
            pos = torch.mean(torch.log(torch.sigmoid(torch.sum(merged_embedding * users_embed, dim=1)) + 1e-24))
            neg = torch.mean(torch.log(torch.tensor(1.0) - torch.sigmoid(
                torch.sum(torch.unsqueeze(merged_embedding, dim=1) * negs, dim=2)) + 1e-24))
            loss = - torch.mean(pos + neg)

        return loss

    def gen_memory_embedding(self, idx, idx_embed):
        memory_matrix = self.value_matrix[idx]

        memory_embedding = self.mulithead_attention(memory_matrix)
        memory_embedding = F.normalize(memory_embedding, p=2, dim=1)
        merged_embedding = torch.tanh(self.embed_merge_linear(torch.cat([idx_embed, memory_embedding], dim=1)))

        return merged_embedding

    def clear_memory(self):
        self.value_matrix.data = torch.zeros(self.num_memories, self.memory_size, self.embedding_size).cuda()


class TigeCMN(nn.Module):
    def __init__(self, args, data):
        super(TigeCMN, self).__init__()
        self.user_num = data.user_num
        self.item_num = data.item_num
        self.attr_num = data.attr_num

        self.embedding_size = args.embedding_size
        self.memory_size = args.memory_size

        self.key_matrix = Parameter(torch.zeros(self.memory_size, self.embedding_size))
        nn.init.xavier_uniform_(self.key_matrix.data)
        self.time_vector = Parameter(torch.zeros(1, self.embedding_size))
        nn.init.xavier_uniform_(self.time_vector.data)

        self.user_memory_network = MemoryNetwork(args, self.user_num, self.key_matrix)
        self.item_memory_network = MemoryNetwork(args, self.item_num, self.key_matrix)

        self.user_embeddings = nn.Embedding(self.user_num, self.embedding_size, max_norm=1)
        self.item_embeddings = nn.Embedding(self.item_num, self.embedding_size, max_norm=1)

        self.encode_attributes = nn.Sequential(
            nn.Linear(self.attr_num + self.embedding_size * 2, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, self.embedding_size)
        )

    def forward(self, users, items, neg_users, neg_items, user_delta_time, item_delta_time, attrs, state='user_update'):
        # [batch_size, embedding_size]
        users_embed = self.user_embeddings(users)
        items_embed = self.item_embeddings(items)

        # attrs_embed = self.encode_attributes(attrs)
        if state == 'user_update':
            # [batch_size, neg_sample, embedding_size]
            delta_time = torch.unsqueeze(user_delta_time, dim=1)
            time_embed = torch.matmul(delta_time, self.time_vector)
            attrs_input = torch.cat([attrs, items_embed, time_embed], dim=1)
            attrs_embed = self.encode_attributes(attrs_input)
            neg_items_embed = self.item_embeddings(neg_items)
            loss = self.user_memory_network(users, users_embed, items_embed, neg_items_embed, user_delta_time, attrs_embed, state='user_update')
        else:
            # [batch_size, neg_sample, embedding_size]
            delta_time = torch.unsqueeze(item_delta_time, dim=1)
            time_embed = torch.matmul(delta_time, self.time_vector)
            attrs_input = torch.cat([attrs, users_embed, time_embed], dim=1)
            attrs_embed = self.encode_attributes(attrs_input)
            neg_users_embed = self.user_embeddings(neg_users)
            loss = self.item_memory_network(items, users_embed, items_embed, neg_users_embed, item_delta_time, attrs_embed, state='item_update')

        return loss
