import torch.nn as nn
import torch
import json
import random
import numpy as np
from config import Config
from dataset import Dataset,evaluate_model,evaluate_single
from elasticsearch import Elasticsearch

torch.manual_seed(101)
np.random.seed(101)
random.seed(101)

config = Config()
es = Elasticsearch()

def perform_elastic(query_word):
    response = es.search(index="dbentityindex11", body={"query": {"multi_match": {"query": query_word, "fields": ["wikidataLabel", "dbpediaLabel^1.5"]}}, "size": 200})

class ER_model(nn.Module):
    def __init__(self, config, vocab_size, word_embeddings):
        super(ER_model, self).__init__()
        self.config = config
        self.embeddings = nn.Embedding(vocab_size, self.config.embed_size)
        self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)
        self.fc1 = nn.Linear(self.config.embed_size, self.config.hidden_size)
        self.fc2 = nn.Linear(self.config.hidden_size, self.config.output_size)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x):
        embedded_sent = self.embeddings(x).permute(1,0,2)
        h = self.sigmoid(self.fc1(embedded_sent.mean(1)))
        z = self.sigmoid(self.fc2(h))
        return self.softmax(z)

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def add_loss_op(self, loss_op):
        self.loss_op = loss_op

    def reduce_lr(self):
        print("Reducing LR")
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2

vocab =  torch.load("saved_model/vocab.voc")
emb = torch.load("saved_model/emb.voc")
model = ER_model(config, len(vocab), emb)

optimizer = torch.optim.RMSprop(model.parameters(), lr=config.lr)
NLLLoss = nn.NLLLoss()
model.add_optimizer(optimizer)
model.add_loss_op(NLLLoss)
model.load_state_dict(torch.load("saved_model/model.pt"))

print( type(perform_elastic("city")))
result = evaluate_single(config, model, ["license", "president", "city of"], vocab)
print(["Entity" if i==1 else "Relation" for i in result.tolist()])