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

es = Elasticsearch()

entities = set()
predicates = set()

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



    def run_epoch(self, train_iterator, val_iterator, epoch):
        train_losses = []
        val_accuracies = []
        losses = []

        # Reduce learning rate as number of epochs increase
        if (epoch == int(self.config.max_epochs / 3)) or (epoch == int(2 * self.config.max_epochs / 3)):
            self.reduce_lr()

        for i, batch in enumerate(train_iterator):
            self.optimizer.zero_grad()

            x = batch.text
            y = (batch.label - 1).type(torch.LongTensor)
            y_pred = self.__call__(x)
            loss = self.loss_op(y_pred, y)

            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()

            if i % 50 == 0:
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                print("\tTraining loss: {:.5f}".format(loss))
                losses = []

                # Evalute Accuracy on validation set
                val_accuracy = evaluate_model(self, val_iterator)
                print("\tVal Accuracy: {:.4f}".format(val_accuracy))
                self.train()

        return train_losses, val_accuracies


def generate_ent_pred_list(data):
    for d in data:
        if "entity mapping" in d:
            for e in d["entity mapping"]:
                entities.add(e["label"])
        if "predicate mapping" in d:
            for p in d["predicate mapping"]:
                predicates.add(p["label"])

def generate_train_test():
    all_ER = dict()
    train_ER = dict()
    test_ER = dict()
    entities_ = list(entities)
    for e in entities_:
        all_ER.update({e:1})
    predicates_ = list(predicates)
    for p in predicates_:
        all_ER.update({p:2})

    keys = list(all_ER.keys())
    random.shuffle(keys)
    print("Total {} keys.".format(len(keys)))

    eighty = len(keys)*90.0/100.0
    idx = 0
    for key in keys:
        if idx<=eighty:
            train_ER.update({key:all_ER[key]})
        else:
            test_ER.update({key:all_ER[key]})
        idx+=1
    CSV = "\n".join([str(v) + ',' + k.lower() for k, v in train_ER.items()])
    with open("train_ER.csv", "w") as file:
        file.write(CSV)
    file.close()

    CSV = "\n".join([str(v) + ',' + k.lower() for k, v in test_ER.items()])
    with open("test_ER.csv", "w") as file:
        file.write(CSV)
    file.close()


def load_lcquad():
    lcquad_data = json.load(open("data/lcquad.json",'r'))
    return lcquad_data


def perform_elastic(query_word):
    response = es.search(index="dbentityindex11", body={"query": {"multi_match": {"query": query_word, "fields": ["wikidataLabel", "dbpediaLabel^1.5"]}}, "size": 200})

if __name__=="__main__":

    data = load_lcquad()
    generate_ent_pred_list(data)
    #generate_train_test()
    print("Total # of distinct entities: ",len(entities))
    print("Total # of distinct predicates: ",len(predicates))
    train_file = 'train_ER.csv'
    test_file = 'test_ER.csv'
    w2v_file = 'data/wiki.en.vec'
    config = Config()
    dataset = Dataset(config)
    dataset.load_data(w2v_file, train_file, test_file)

    print("printing")
    model = ER_model(config, len(dataset.vocab), dataset.word_embeddings)

    model.train()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.lr)
    NLLLoss = nn.NLLLoss()
    model.add_optimizer(optimizer)
    model.add_loss_op(NLLLoss)

    train_losses = []
    val_accuracies = []
    for i in range(config.max_epochs):
        print("Epoch: {}".format(i))
        train_loss, val_accuracy = model.run_epoch(dataset.train_iterator, dataset.val_iterator, i)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)

    train_acc = evaluate_model(model, dataset.train_iterator)
    val_acc = evaluate_model(model, dataset.val_iterator)
    test_acc = evaluate_model(model, dataset.test_iterator)
    torch.save(model.state_dict(),"saved_model/model.pt")
    torch.save(dataset.vocab,"saved_model/vocab.voc")
    torch.save(dataset.word_embeddings,"saved_model/emb.voc")
    evaluate_single(config,model,["license","president","city of","Barack obama","Seikh Hasina"],dataset.vocab)

    print('Final Training Accuracy: {:.4f}'.format(train_acc))
    print('Final Validation Accuracy: {:.4f}'.format(val_acc))
    print('Final Test Accuracy: {:.4f}'.format(test_acc))