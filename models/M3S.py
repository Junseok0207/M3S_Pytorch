import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from src.utils import reset, set_random_seeds, masking
from sklearn.cluster import KMeans
from embedder import embedder
from torch_geometric.utils import to_dense_adj

class M3S_Trainer(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args

    def _init_model(self):
        self.model = M3S(self.encoder, self.classifier).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.decay)
        
    def _init_dataset(self):
        
        self.labels = deepcopy(self.data.y)
        self.running_train_mask = deepcopy(self.train_mask)
        
        eta = self.data.num_nodes / (to_dense_adj(self.data.edge_index).sum() / self.data.num_nodes)**len(self.hidden_layers)
        self.t = (self.labels[self.train_mask].unique(return_counts=True)[1]*3*eta/len(self.labels[self.train_mask])).type(torch.int64)
        self.t = self.t / self.args.stage

    def pretrain(self, mask, stage):
        for epoch in range(200):
            self.model.train()
            self.optimizer.zero_grad()

            logits, _ = self.model.cls(self.data)
            loss = F.cross_entropy(logits[self.running_train_mask], self.labels[self.running_train_mask])
            loss.backward()
            self.optimizer.step()

            st = '[Fold : {}][Stage : {}/{}][Epoch {}/{}] Loss: {:.4f}'.format(mask+1, stage+1, self.args.stage, epoch+1, 200, loss.item())
            print(st)

        if self.args.clustering:
            # Clustering
            self.model.eval()
            rep = self.model.encoder(self.data).detach()
            rep = F.normalize(rep, dim=1)
            rep = rep.to('cpu').numpy()
            clustering = KMeans(n_clusters=self.args.num_K).fit(rep)
            clustering_result = clustering.predict(rep)
            
            # Aligning Mechanism
            labeled_centroid_list = []
            for m in range(self.num_classes):
                m_mask = torch.logical_and(self.labels==m, self.running_train_mask).to('cpu')
                m_rep = rep[m_mask]
                m_centroid = m_rep.mean(0)
                labeled_centroid_list.append(m_centroid)
            labeled_centroids = np.stack(labeled_centroid_list)

            pseudo_labels = np.zeros_like(clustering_result)
            pseudo_labels -= 1
            clusters = np.unique(clustering_result)
            num_cluster = clusters.shape[0]
            if num_cluster != self.args.num_K:
                print("Empty cluster is occured")
            for l in clusters:
                l_mask = torch.logical_and(torch.tensor(clustering_result==l).to(self.args.device), ~self.running_train_mask).to('cpu')
                l_rep = rep[l_mask]
                l_centroid = l_rep.mean(0)
                distance = (labeled_centroids - l_centroid)**2
                distance = distance.sum(1)
                pseudo_label = np.argmin(distance)
                pseudo_labels[l_mask] = pseudo_label
            assert (pseudo_labels[~self.running_train_mask.to('cpu')]==-1).sum() == 0
       
        # Pseudo-labeling
        self.model.eval()
        logits, _ = self.model.cls(self.data)
        predictions = F.softmax(logits, dim=1)
        if self.args.clustering:
            y_train, self.running_train_mask = self.selftraining_with_checking(predictions, pseudo_labels)
        else:
            y_train, self.running_train_mask = self.selftraining(predictions)
        self.labels[self.running_train_mask] = torch.argmax(y_train[self.running_train_mask], dim=1)

    def train(self):
        
        for fold in range(self.args.folds):
            set_random_seeds(fold)
            self.train_mask, self.val_mask, self.test_mask = masking(fold, self.data, self.args.label_rate)
            self._init_dataset()

            for stage in range(self.args.stage):
                self._init_model()
                self.pretrain(fold, stage)
                
            for epoch in range(1,self.args.epochs+1):
                self.model.train()
                self.optimizer.zero_grad()

                logits, _ = self.model.cls(self.data)
                loss = F.cross_entropy(logits[self.running_train_mask], self.labels[self.running_train_mask])
                loss.backward()
                self.optimizer.step()

                st = '[Fold : {}][Epoch {}/{}] Loss: {:.4f}'.format(fold+1, epoch, self.args.epochs, loss.item())

                # evaluation
                self.evaluate(self.data, st)
                if self.cnt == self.args.patience:
                    print("early stopping!")
                    break
            self.save_results(fold)
        
        self.summary()
                

    def selftraining_with_checking(self, predictions, pseudo_labels):
        new_gcn_index = torch.argmax(predictions, dim=1)
        confidence = torch.max(predictions, dim=1)[0]
        sorted_index = torch.argsort(-confidence)
        y_train = F.one_hot(self.labels).float()
        y_train[~self.running_train_mask] = 0
        no_class = y_train.shape[1]
        assert len(self.t) >= no_class
        index = []
        count = [0 for i in range(no_class)]
        for i in sorted_index:
            for j in range(no_class):
                if new_gcn_index[i] == j and count[j] < self.t[j] and not self.running_train_mask[i]:
                    index.append(i.item())
                    count[j] += 1
        filtered_index = []
        deleted_index = []
        for i in index:
            if pseudo_labels[i] == new_gcn_index[i].item():
                filtered_index.append(i)
            else:
                deleted_index.append(i)
        
        indicator = torch.zeros(self.train_mask.shape, dtype=torch.bool)
        indicator[filtered_index] = True
        indicator = torch.logical_and(torch.logical_not(self.running_train_mask), indicator.to(self.device))
        prediction = torch.zeros(predictions.shape).to(self.device)
        prediction[torch.arange(len(new_gcn_index)), new_gcn_index] = 1.0
        prediction[self.running_train_mask] = y_train[self.running_train_mask]
        y_train = deepcopy(y_train)
        train_mask = deepcopy(self.running_train_mask)
        train_mask[indicator] = 1
        y_train[indicator] = prediction[indicator]
        return y_train, train_mask


    def selftraining(self, predictions):
        new_gcn_index = torch.argmax(predictions, dim=1)
        confidence = torch.max(predictions, dim=1)[0]
        sorted_index = torch.argsort(-confidence)
        
        y_train = F.one_hot(self.labels).float()
        y_train[~self.running_train_mask] = 0
        no_class = y_train.shape[1]  # number of class
        assert len(self.t) >= no_class
        index = []
        count = [0 for i in range(no_class)]
        for i in sorted_index:
            for j in range(no_class):
                if new_gcn_index[i] == j and count[j] < self.t[j] and not self.running_train_mask[i]:
                    index.append(i)
                    count[j] += 1
        indicator = torch.zeros(self.train_mask.shape, dtype=torch.bool)
        indicator[index] = True
        indicator = torch.logical_and(torch.logical_not(self.running_train_mask), indicator.to(self.device))
        prediction = torch.zeros(predictions.shape).to(self.device)
        prediction[torch.arange(len(new_gcn_index)), new_gcn_index] = 1.0
        prediction[self.running_train_mask] = y_train[self.running_train_mask]
        y_train = deepcopy(y_train)
        train_mask = deepcopy(self.running_train_mask)
        train_mask[indicator] = 1
        y_train[indicator] = prediction[indicator]
        return y_train, train_mask



class M3S(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.reset_parameters()

    def forward(self, x):
        out = self.encoder(x)
        logits, predictions = self.classifier(out)
        return logits, predictions

    def cls(self, x):
        return self.forward(x)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.classifier)

def sample_mask(idx, l):
    """Create mask."""
    mask = torch.zeros(l)
    mask[idx] = 1
    return torch.as_tensor(mask, dtype=torch.bool)
