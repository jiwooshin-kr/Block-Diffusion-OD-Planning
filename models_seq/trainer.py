from sklearn.mixture import GaussianMixture
from math import exp
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader, random_split, Sampler, Subset
from models_seq.eps_models import EPSM
from loader.dataset import TrajFastDataset
from models_seq.seq_models import Destroyer, Restorer
import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import cycle
from os.path import join
from utils.coors import wgs84_to_gcj02

import logging


class Trainer:
    def __init__(self, model: nn.Module, dataset, model_path, model_name, args):
        self.model = model 
        self.device = model.device
        self.dataset = dataset
        self.model_path = model_path
        self.model_name = model_name
        self.args = args

    def train(self, n_epoch, batch_size, lr, remove_region=None):
        torch.autograd.set_detect_anomaly(True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr)

        # split train test
        train_num = int(0.8 * len(self.dataset))
        train_dataset, test_dataset = random_split(self.dataset, [train_num , len(self.dataset) - train_num])

        # randomly removed edge for new A' and defined sampler that only sample paths that satisfy A'

        if remove_region is not None:
            print(f'remove {remove_region}')
            A_new = self.dataset.edit(removal={"regions": remove_region}, direct_change=False)
            torch.save(A_new, join(self.model_path, f"{self.model_name}_{remove_region}_A_new.pt"))
            train_sampler = CustomPathBatchSampler(train_dataset, batch_size=batch_size, adjacency_matrix=A_new, shuffle=True)
            test_sampler = CustomPathBatchSampler(test_dataset, batch_size=batch_size, adjacency_matrix=A_new, shuffle=False)
            trainloader = DataLoader(train_dataset, batch_sampler=train_sampler,
                                        collate_fn=lambda data: [torch.Tensor(each).to(self.device) for each in data])
            testloader = DataLoader(test_dataset, batch_sampler=test_sampler,
                                    collate_fn=lambda data: [torch.Tensor(each).to(self.device) for each in data])
        else:
            train_sampler = None
            test_sampler = None
            trainloader = DataLoader(train_dataset, batch_size,
                                        collate_fn=lambda data: [torch.Tensor(each).to(self.device) for each in data])
            testloader = DataLoader(test_dataset, batch_size,
                                    collate_fn=lambda data: [torch.Tensor(each).to(self.device) for each in data])
        self.model.train()
        iter, train_loss_avg = 0, 0
        kl_loss_avg, ce_loss_avg, con_loss_avg = 0, 0, 0

        logging.basicConfig(
            filename=f'./sets_model/{self.model_name}train_log.txt',
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
        )

        try:
            for epoch in range(n_epoch):
                for xs in trainloader:
                    kl_loss, ce_loss, con_loss = self.model(xs)
                    if ce_loss.item() < 60 / 16:
                        loss = kl_loss
                    elif (self.args.lam_ce == 0.) and (self.args.lam_con == 0.):
                        loss = kl_loss
                    elif self.args.lam_ce == 0.:
                        loss = kl_loss + self.args.lam_con * con_loss
                    elif self.args.lam_con == 0.:
                        loss = kl_loss + self.args.lam_ce * ce_loss
                    else:
                        loss = kl_loss + self.args.lam_ce * ce_loss + self.args.lam_con * con_loss
                        # loss = kl_loss + ce_loss
                    train_loss_avg += loss.item()
                    kl_loss_avg += kl_loss.item()
                    ce_loss_avg += ce_loss.item()
                    con_loss_avg += con_loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    # TODO: clip norm
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    iter += 1
                    if iter % 100 == 0 or iter == 1:
                        # eval test
                        denom = 1 if iter == 1 else 100
                        test_kl, test_ce, test_con = next(self.eval_test(testloader))
                        test_loss = test_kl + test_ce + test_con
                        print(f"e: {epoch}, i: {iter}, train loss: {train_loss_avg / denom: .4f}, (kl: {kl_loss_avg / denom: .4f}, ce: {ce_loss_avg / denom: .4f}, co: {con_loss_avg / denom: .4f}), test loss: {test_loss: .4f}, (kl: {test_kl: .4f}, ce: {test_ce: .4f}, co: {test_con: .4f})")
                        logging.info(
                            f"e: {epoch}, i: {iter}, train loss: {train_loss_avg / denom: .4f}, "
                            f"(kl: {kl_loss_avg / denom: .4f}, ce: {ce_loss_avg / denom: .4f}, co: {con_loss_avg / denom: .4f}), "
                            f"test loss: {test_loss: .4f}, "
                            f"(kl: {test_kl: .4f}, ce: {test_ce: .4f}, co: {test_con: .4f})"
                        )
                        train_loss_avg, kl_loss_avg, ce_loss_avg, con_loss_avg = 0., 0., 0., 0.
                model_name = f"{self.model_name}_epoch_{epoch}.pth"
                torch.save(self.model, join(self.model_path, model_name))
        except KeyboardInterrupt as E:
            print("Training interruptted, begin saving...")
            self.model.eval()
            model_name = f"tmp_iter_{iter}.pth"
        # save
        self.model.eval()
        # model_name = f"finished_{iter}.pth"
        # torch.save(self.model, join(self.model_path, model_name))
        # print("save finished!")

    def train_gmm(self, gmm_samples, n_comp):
        gmm = GaussianMixture(n_components=n_comp, covariance_type="tied")
        gmm_samples = min(len(self.dataset), gmm_samples)
        lenghts = np.array([len(self.dataset[k]) for k in range(gmm_samples)]).reshape(-1, 1)
        gmm.fit(lenghts)
        self.model.gmm = gmm  


    def eval_test(self, test_loader):
        with torch.no_grad():
            for txs in cycle(test_loader):
                kl_loss, ce_loss, test_con = self.model(txs)
                yield (kl_loss.item(), ce_loss.item(), test_con.item())

    def drop_edges_symmetric(self, A, drop_ratio=0.1):
        A = A.clone().cpu()
        N = A.size(0)

        row_idx, col_idx = torch.triu_indices(N, N, offset=1)
        edge_mask = A[row_idx, col_idx] == 1
        edge_indices = torch.stack([row_idx[edge_mask], col_idx[edge_mask]], dim=0)

        num_edges = edge_indices.size(1)
        num_to_drop = int(num_edges * drop_ratio)

        perm = torch.randperm(num_edges)
        edges_to_drop = edge_indices[:, perm[:num_to_drop]]

        A[edges_to_drop[0], edges_to_drop[1]] = 0
        A[edges_to_drop[1], edges_to_drop[0]] = 0

        return A


class CustomPathBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, adjacency_matrix, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.A = adjacency_matrix
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        current_batch = []
        for idx in indices:
            path = self.dataset[idx]
            if self._valid_path(path):
                current_batch.append(idx)
            if len(current_batch) == self.batch_size:
                yield current_batch
                current_batch = []

    def _valid_path(self, path):
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if self.A[u, v] == 0:
                return False
        return True

    def __len__(self):
        return len(self.dataset) // self.batch_size


class Trainer_disc:
    def __init__(self, model: nn.Module, dataset, model_path, model_name, dataset_new=None, args=None):
        self.model = model
        self.device = model.device
        self.dataset = dataset
        self.dataset_new = dataset_new
        self.model_path = model_path
        self.model_name = model_name
        self.args = args

    def train(self, n_epoch, batch_size, lr, remove_region=None, remove_random=False):
        torch.autograd.set_detect_anomaly(True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr)

        # ============================================================
        # Prepare train/test datasets
        # ============================================================
        train_num = int(0.8 * min(len(self.dataset), len(self.dataset_new)))
        train_dataset, test_dataset = random_split(self.dataset, [train_num , len(self.dataset) - train_num])
        train_dataset_new, test_dataset_new = random_split(self.dataset_new, [train_num , len(self.dataset_new) - train_num])

        trainloader_A = DataLoader(train_dataset, batch_size,
                                    collate_fn=lambda data: [torch.Tensor(each).to(self.device) for each in data])
        trainloader_new = DataLoader(train_dataset_new, batch_size,
                                    collate_fn=lambda data: [torch.Tensor(each).to(self.device) for each in data])
        testloader_A = DataLoader(test_dataset, batch_size,
                                collate_fn=lambda data: [torch.Tensor(each).to(self.device) for each in data])
        testloader_new = DataLoader(test_dataset_new, batch_size,
                                collate_fn=lambda data: [torch.Tensor(each).to(self.device) for each in data])

        # ============================================================
        # Logging setup
        # ============================================================
        logging.basicConfig(
            filename=f'./sets_model/{self.model_name}_disc_log.txt',
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
        )

        self.model.train()
        iter = 0
        train_loss_avg = 0
        org_logits_avg = 0
        new_logits_avg = 0
        acc_avg = 0

        # ========================================================
        # Training loop
        # ========================================================
        test_generator = self.eval_test_disc(testloader_A, testloader_new)
        try:
            for epoch in range(n_epoch):
                for xs, newxs in zip(trainloader_A, trainloader_new):
                    loss, org_logits, new_logits, acc = self.model(xs, newxs, self.dataset.A, self.dataset_new.A)

                    train_loss_avg += loss.item()
                    org_logits_avg += org_logits.mean().item()
                    new_logits_avg += new_logits.mean().item()
                    acc_avg += acc.item()

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                    iter += 1
                    # =================================================
                    # Periodic evaluation
                    # =================================================
                    if iter % 100 == 0 or iter == 1:
                        denom = 1 if iter == 1 else 100

                        self.model.eval()
                        with torch.no_grad():
                            test_loss, test_org_logits_mean, test_new_logits_mean, test_acc = next(test_generator)
                        self.model.train() 
                        
                        print(f"[Train] e: {epoch}, i: {iter}, train loss: {train_loss_avg / denom: .4f}, train acc: {acc_avg / denom: .4f}, org_logits: {org_logits_avg / denom: .4f}, new_logits: {new_logits_avg / denom: .4f}")
                        print(f"[Test] e: {epoch}, i: {iter}, test loss: {test_loss: .4f}, test acc: {test_acc: .4f}, org_logits: {test_org_logits_mean: .4f}, new_logits: {test_new_logits_mean: .4f}")
                        logging.info( f"e: {epoch}, i: {iter}, train_loss: {train_loss_avg / denom: .4f}, train_acc: {acc_avg / denom: .4f}, train_org_logits: {org_logits_avg / denom: .4f}, train_new_logits: {new_logits_avg / denom: .4f}, test_loss: {test_loss: .4f}, test_acc: {test_acc: .4f}, test_org_logits: {test_org_logits_mean: .4f}, test_new_logits: {test_new_logits_mean: .4f}")
                        
                        train_loss_avg = 0.
                        org_logits_avg = 0.
                        new_logits_avg = 0.
                        acc_avg = 0.

                    # =================================================
                    # Iteration checkpoint (Up to 4000 iterations)
                    # =================================================
                    end_iter = 4000
                    if self.args.save_iter == 500:
                        end_iter = 2500
                        
                    if self.args.save_iter > 0:
                        if iter % self.args.save_iter == 0 and iter <= end_iter:
                            model_name = f"{self.model_name}_iter_{iter}.pth"
                            torch.save(self.model, join(self.model_path, model_name))

                # =====================================================
                # Epoch checkpoint
                # =====================================================
                if epoch % self.args.save_step == 0:
                    model_name = f"{self.model_name}_epoch_{epoch}.pth"
                    torch.save(self.model, join(self.model_path, model_name))
        
        except KeyboardInterrupt as E:
            print("Training interruptted, begin saving...")
            self.model.eval()
            model_name = f"tmp_iter_{iter}.pth"

        self.model.eval()


    def train_gmm(self, gmm_samples, n_comp):
        gmm = GaussianMixture(n_components=n_comp, covariance_type="tied")
        gmm_samples = min(len(self.dataset), gmm_samples)
        lenghts = np.array([len(self.dataset[k]) for k in range(gmm_samples)]).reshape(-1, 1)
        gmm.fit(lenghts)
        self.model.gmm = gmm


    def eval_test_disc(self, testloader_A, testloader_new):
        for org_batch, new_batch in cycle(zip(testloader_A, testloader_new)):
            loss, org_logits, new_logits, acc = self.model(org_batch, new_batch, self.dataset.A, self.dataset_new.A)
            yield (loss.item(), org_logits.mean().item(), new_logits.mean().item(), acc.item())

    def drop_edges_symmetric(self, A, drop_ratio=0.1):
        A = A.clone().cpu()
        N = A.size(0)

        row_idx, col_idx = torch.triu_indices(N, N, offset=1)
        edge_mask = A[row_idx, col_idx] == 1
        edge_indices = torch.stack([row_idx[edge_mask], col_idx[edge_mask]], dim=0)

        num_edges = edge_indices.size(1)
        num_to_drop = int(num_edges * drop_ratio)

        perm = torch.randperm(num_edges)
        edges_to_drop = edge_indices[:, perm[:num_to_drop]]

        A[edges_to_drop[0], edges_to_drop[1]] = 0
        A[edges_to_drop[1], edges_to_drop[0]] = 0

        return A


class Trainer_disc_Multiple:
    def __init__(self, model: nn.Module, dataset, model_path, model_name, dataset_new=None, args=None):
        self.model = model
        self.device = model.device
        self.dataset = dataset
        self.dataset_new = dataset_new
        self.model_path = model_path
        self.model_name = model_name
        self.args = args

    def train(self, n_epoch, batch_size, lr, remove_region=None, remove_random=False):
        torch.autograd.set_detect_anomaly(True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr)

        # ============================================================
        # Logging setup
        # ============================================================
        logging.basicConfig(
            filename=f'./sets_log/{self.model_name}_disc_log.txt',
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
        )

        # ============================================================
        # Prepare train/test datasets
        # ============================================================
        k = len(self.dataset_new)

        min_new_len = min(len(d) for d in self.dataset_new)        
        train_num = int(0.8 * min(len(self.dataset), min_new_len * k))

        print(f"Train num: {train_num}, Original dataset length: {len(self.dataset)}, Min new data length: {min_new_len}")

        # Loader for Normal case
        train_dataset, test_dataset = random_split(self.dataset, [train_num , len(self.dataset) - train_num])

        trainloader_A = DataLoader(train_dataset, batch_size,
                                    collate_fn=lambda data: [torch.Tensor(each).to(self.device) for each in data])

        testloader_A = DataLoader(test_dataset, batch_size,
                                collate_fn=lambda data: [torch.Tensor(each).to(self.device) for each in data])

        # loader for the exceptional cases
        trainloaders_new = []
        testloaders_new = []
        train_adjacency_matrices = []
        test_adjacency_matrices = []  

        per_dataset_train_num = int(0.8 * min_new_len)     

        for dataset_new in self.dataset_new:  # Iterate over the list of new datasets
            train_dataset_new, test_dataset_new = random_split(dataset_new, [per_dataset_train_num, len(dataset_new) - per_dataset_train_num])

            trainloader_new = DataLoader(train_dataset_new, batch_size,
                                        collate_fn=lambda data: [torch.Tensor(each).to(self.device) for each in data])
            testloader_new = DataLoader(test_dataset_new, batch_size,
                                        collate_fn=lambda data: [torch.Tensor(each).to(self.device) for each in data])

            trainloaders_new.append(trainloader_new)
            testloaders_new.append(testloader_new)
            train_adjacency_matrices.append(dataset_new.A)  
            test_adjacency_matrices.append(dataset_new.A) 

        self.model.train()
        iter = 0
        train_loss_avg = 0
        org_logits_avg = 0
        new_logits_avg = 0
        acc_avg = 0

        # ========================================================
        # Training loop
        # ========================================================
        test_generators = [
            self.eval_test_disc_with_adj(testloader_A, testloader_new, adj_matrix_new)
            for testloader_new, adj_matrix_new in zip(testloaders_new, test_adjacency_matrices)
        ]

        try:         
            for epoch in range(n_epoch):
                new_iters = [cycle(loader) for loader in trainloaders_new]
                for batch_idx, xs in enumerate(trainloader_A):
                    # Change the excpetional case for every batch
                    i = batch_idx % len(trainloaders_new)
                    newxs = next(new_iters[i])
                    adj_matrix_new = train_adjacency_matrices[i]

                    # Run model and compute loss
                    loss, org_logits, new_logits, acc = self.model(xs, newxs, self.dataset.A, adj_matrix_new)

                    train_loss_avg += loss.item()
                    org_logits_avg += org_logits.mean().item()
                    new_logits_avg += new_logits.mean().item()
                    acc_avg += acc.item()

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                    iter += 1

                    # =================================================
                    # Periodic evaluation
                    # =================================================
                    if iter % 100 == 0 or iter == 1:
                        denom = 1 if iter == 1 else 100
                        # ============================================================
                        # Evaluate all exceptional datasets and average the results
                        # ============================================================
                        test_loss_list = []
                        test_acc_list = []
                        test_org_logits_list = []
                        test_new_logits_list = []

                        self.model.eval()
                        with torch.no_grad():        
                            for test_generator in test_generators:
                                test_loss_i, test_org_logits_i, test_new_logits_i, test_acc_i = next(test_generator)

                                test_loss_list.append(test_loss_i)
                                test_acc_list.append(test_acc_i)
                                test_org_logits_list.append(test_org_logits_i)
                                test_new_logits_list.append(test_new_logits_i)

                        test_loss = sum(test_loss_list) / len(test_loss_list)
                        test_acc = sum(test_acc_list) / len(test_acc_list)
                        test_org_logits_mean = sum(test_org_logits_list) / len(test_org_logits_list)
                        test_new_logits_mean = sum(test_new_logits_list) / len(test_new_logits_list)

                        self.model.train() 

                        print(f"[Train] e: {epoch}, i: {iter}, train loss: {train_loss_avg / denom: .4f}, train acc: {acc_avg / denom: .4f}, org_logits: {org_logits_avg / denom: .4f}, new_logits: {new_logits_avg / denom: .4f}")
                        print(f"[Test] e: {epoch}, i: {iter}, test loss: {test_loss: .4f}, test acc: {test_acc: .4f}, org_logits: {test_org_logits_mean: .4f}, new_logits: {test_new_logits_mean: .4f}")

                        logging.info(
                            f"e: {epoch}, i: {iter}, train_loss: {train_loss_avg / denom: .4f}, train_acc: {acc_avg / denom: .4f}, train_org_logits: {org_logits_avg / denom: .4f}, train_new_logits: {new_logits_avg / denom: .4f}, "
                            f"test_loss: {test_loss: .4f}, test_acc: {test_acc: .4f}, test_org_logits: {test_org_logits_mean: .4f}, test_new_logits: {test_new_logits_mean: .4f}"
                        )

                        train_loss_avg = 0.
                        org_logits_avg = 0.
                        new_logits_avg = 0.
                        acc_avg = 0.

                    # =================================================
                    # Iteration checkpoint (Up to 4000 iterations)
                    # =================================================
                    end_iter = 4000
                    if self.args.save_iter == 500:
                        end_iter = 2500

                    if self.args.save_iter > 0:
                        if iter % self.args.save_iter == 0 and iter <= end_iter:
                            print(f"Saving model at iteration {iter}...")
                            model_name = f"{self.model_name}_iter_{iter}.pth"
                            torch.save(self.model, join(self.model_path, model_name))

                # =====================================================
                # Epoch checkpoint
                # =====================================================      
                if epoch % self.args.save_step == 0:
                    model_name = f"{self.model_name}_epoch_{epoch}.pth"
                    torch.save(self.model, join(self.model_path, model_name))


        except KeyboardInterrupt as E:
            print("Training interruptted, begin saving...")
            self.model.eval()
            model_name = f"tmp_iter_{iter}.pth"

        self.model.eval()

    def train_gmm(self, gmm_samples, n_comp):
        gmm = GaussianMixture(n_components=n_comp, covariance_type="tied")
        gmm_samples = min(len(self.dataset), gmm_samples)
        lenghts = np.array([len(self.dataset[k]) for k in range(gmm_samples)]).reshape(-1, 1)
        gmm.fit(lenghts)
        self.model.gmm = gmm


    def eval_test_disc_with_adj(self, testloader_A, testloader_new, adj_matrix_new):
        for org_batch, new_batch in cycle(zip(testloader_A, testloader_new)):
            loss, org_logits, new_logits, acc = self.model(org_batch, new_batch, self.dataset.A, adj_matrix_new)
            yield (loss.item(), org_logits.mean().item(), new_logits.mean().item(), acc.item())

    def drop_edges_symmetric(self, A, drop_ratio=0.1):
        A = A.clone().cpu()
        N = A.size(0)

        row_idx, col_idx = torch.triu_indices(N, N, offset=1)
        edge_mask = A[row_idx, col_idx] == 1
        edge_indices = torch.stack([row_idx[edge_mask], col_idx[edge_mask]], dim=0)

        num_edges = edge_indices.size(1)
        num_to_drop = int(num_edges * drop_ratio)

        perm = torch.randperm(num_edges)
        edges_to_drop = edge_indices[:, perm[:num_to_drop]]

        A[edges_to_drop[0], edges_to_drop[1]] = 0
        A[edges_to_drop[1], edges_to_drop[0]] = 0

        return A

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    max_T = 100
    dataset = TrajFastDataset("chengdu", ["20161101"], "./sets_data", device, is_pretrain=True)
    betas = torch.linspace(0.0001, 10, max_T)
    # old beta: 0.01, 15, 50
    
    destroyer = Destroyer(dataset.A, betas, max_T, device)
    eps_model = EPSM(dataset.n_vertex, x_emb_dim=50, hidden_dim=20, dims=[100, 120, 200], device=device, pretrain_path="./sets_data/chengdu_node2vec.pkl")
    restorer = Restorer(eps_model, destroyer, device)
    
    trainer = Trainer(restorer, dataset, device, "./sets_model")
    trainer.train_gmm(gmm_samples=50000, n_comp=5)
    trainer.train(n_epoch=50, batch_size=16, lr=0.0005)
    
    restorer.eval()
    paths = restorer.sample_wo_len(100)
    
    multiple_locs = []
    for path in paths:
        locs = [[wgs84_to_gcj02(dataset.G.nodes[v]["lng"], dataset.G.nodes[v]["lat"])[1], 
                 wgs84_to_gcj02(dataset.G.nodes[v]["lng"], dataset.G.nodes[v]["lat"])[0]] 
                for v in path]
        multiple_locs.append(locs)
        print(locs)
    
    