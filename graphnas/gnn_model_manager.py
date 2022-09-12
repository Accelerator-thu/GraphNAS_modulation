import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import load_data

from graphnas.gnn import GraphNet
from graphnas.utils.model_utils import EarlyStop, TopAverage, process_action


def load(args, save_file=".npy"):
    save_file = args.dataset + save_file
    if os.path.exists(save_file):
        return np.load(save_file).tolist()
    else:
        datas = load_data(args)
        np.save(save_file, datas)
        return datas


def evaluate(output, labels, mask):
    _, indices = torch.max(output, dim=1)
    correct = torch.sum(indices[mask] == labels[mask])
    return correct.item() * 1.0 / mask.sum().item()


# manager the train process of GNN on citation dataset
class CitationGNNManager(object):

    def __init__(self, args):

        self.args = args

        if hasattr(args, 'dataset') and args.dataset in ["cora", "citeseer", "pubmed"]:
            self.data = load(args)
            self.args.in_feats = self.in_feats = self.data.features.shape[1]
            self.args.num_class = self.n_classes = self.data.num_labels

        self.early_stop_manager = EarlyStop(10)
        self.reward_manager = TopAverage(10)

        self.args = args
        self.drop_out = args.in_drop
        self.multi_label = args.multi_label
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.retrain_epochs = args.retrain_epochs
        self.loss_fn = torch.nn.BCELoss()
        self.epochs = args.epochs
        self.train_graph_index = 0
        self.train_set_length = 10

        self.param_file = args.param_file
        self.shared_params = None

        self.loss_fn = torch.nn.functional.nll_loss

    def load_param(self):
        # don't share param
        pass

    def save_param(self, model, update_all=False):
        # don't share param
        pass

    # train from scratch
    def evaluate(self, actions=None, format="two"):
        actions = process_action(actions, format, self.args)
        # print("train action:", actions)

        # create model
        model = self.build_gnn(actions)
        # model1 = self.build_gnn(actions)
        # print("*" * 50)
        # for name, param in model.named_parameters():
        #     print(param)
        # print("*" * 50)
        # for name, param in model1.named_parameters():
        #     print(param)
        # print("*" * 50)

        # print(model.parameters())
        # model = self.build_gnn(actions)
        # print(model.parameters())

        if self.args.cuda:
            model.cuda()

        # use optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        try:
            # print("*" * 20 + "flag here" + "*" * 20)
            model, val_acc, test_acc = self.run_model(model, optimizer, self.loss_fn, self.data, self.epochs,
                                                      cuda=self.args.cuda, return_best=True,
                                                      half_stop_score=max(self.reward_manager.get_top_average() * 0.7,
                                                                          0.4))
            grads = {}
            trials = 1000
            for trial in range(trials):
                model = self.build_gnn(actions)
                # grads = self.run_model(model, optimizer, self.loss_fn, self.data, self.epochs,
                #                                       cuda=self.args.cuda, return_best=True,
                #                                       half_stop_score=max(self.reward_manager.get_top_average() * 0.7,
                #                                                           0.4), opt="wo_train", grads=grads)
                if self.args.cuda:
                    model.cuda()
                # grads = {}
                # for trial in range(1, trials + 1):
                model.train()
                t0 = time.time()
                # forward
                for _ in range(5):  # train for 5 epochs
                    logits = model(self.data.x, self.data.edge_index)
                    logits = F.log_softmax(logits, 1)
                    loss = self.loss_fn(logits[self.data.train_mask], self.data.y[self.data.train_mask])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                logits = model(self.data.x, self.data.edge_index)
                # print(model._parameters)
                # print(vars(model))
                # print(model.named_parameters())
                # for name, parameter in model.named_parameters():
                #     print(f"name: {name}")
                #     print(f"parameter: {parameter}")
                #     print(f"parameter grad: {parameter.grad}")
                #     print(f"parameter grad v-1.data: {parameter.grad.view(-1).data[0:100]}") if parameter.grad is not None else print("None Gradient!")
                #     print(f"len parameter grad v-1.data: {len(parameter.grad.view(-1).data[0:100])}") if parameter.grad is not None else print("None Gradient!")
                logits = F.log_softmax(logits, 1)
                loss = self.loss_fn(logits[self.data.train_mask], self.data.y[self.data.train_mask])
                optimizer.zero_grad()
                loss.backward()
                # if epoch == 1:
                import copy
                # if not grad: return 0, 0, 0,batch_time.sum
                index_grad = 0
                index_name = 0
                for name, param in model.named_parameters():
                    # print(param.grad.view(-1).data)
                    if param.grad is None:
                        # print(name)
                        continue
                    # if param.grad.view(-1)[0] == 0 and param.grad.view(-1)[1] == 0: continue #print(name)
                    # print(i)
                    if index_name > 10: break
                    if len(param.grad.view(-1).data[0:100]) < 50:
                        continue
                    index_grad = name
                    index_name += 1
                    # if index_name > 10: break
                    # index_grad +=
                    if name in grads:
                        grads[name].append(copy.copy(param.grad.view(-1).data[0:100]))
                    else:
                        grads[name] = [copy.copy(param.grad.view(-1).data[0:100])]
                    # print(index_grad)
                # print(f"length of grads[index_grad]: {len(grads[index_grad])}")
                if len(grads[index_grad]) == 50:
                    conv = 0
                    # maxconv = 0
                    # minconv = 0
                    # lower_layer = 1
                    # top_layer = 1
                    para = 0

                    for name in grads:
                        # print(name)
                        '''for i in range(50):
                            grads[name][i] = torch.tensor(grads[name][i], dtype=torch.float)
                            #grads[name][i] = grads[name][i] - grads[name][i].mean()
                            #means += grads[name][i]
                        means = grads[name][0]
                        for i in range(1,50):
                            means += grads[name][i]
                        conv = torch.abs(torch.dot(means, means)/2500)'''
                        for i in range(50):  # nt(self.grads[name][0].size()[0])):
                            # if len(grads[name])!=: print(name)
                            # for j in range(50):
                            # if i == j: continue
                            grad1 = torch.tensor([grads[name][k][i] for k in range(
                                25)])  # torch.tensor(grads[name][j],dtype=torch.float)#torch.tensor([grads[name][j][k] for j in range(25)],dtype=torch.float)
                            grad2 = torch.tensor([grads[name][k][i] for k in range(25,
                                                                                   50)])  # torch.tensor(grads[name][i],dtype=torch.float)#torch.tensor([grads[name][j][k] for j in range(25,50)],dtype=torch.float)
                            grad1 = grad1 - grad1.mean()
                            grad2 = grad2 - grad2.mean()
                            conv += torch.dot(grad1, grad2) / 2500  # torch.tensor(grad1, dtype=torch.float), torch.tensor(grad1,dtype=torch.float))#i#/i1.0*self.grads[name][0].size()[0]
                            para += 1
                    # conv /= para
                    # print("dot product: ", conv.item())
                    with open("/home/yc568/GraphNAS/datacache1.txt", mode='a') as file:
                        file.write(str(conv.item()))
                        file.write('\n')
                        file.close()
                    print(conv.item())
                    # print(conv, maxconv, minconv)# top_layer/lower_layer)
                    # print("endddddddddd")
                    # count += 1
                    break
                #     optimizer.step()
                #     train_loss = loss.item()
                #
                #     # evaluate
                #     model.eval()
                #     logits = model(data.x, data.edge_index)
                #     logits = F.log_softmax(logits, 1)
                #     train_acc = evaluate(logits, data.y, data.train_mask)
                #     dur.append(time.time() - t0)
                #
                #     val_acc = evaluate(logits, data.y, data.val_mask)
                #     test_acc = evaluate(logits, data.y, data.test_mask)
                #
                #     loss = loss_fn(logits[data.val_mask], data.y[data.val_mask])
                #     val_loss = loss.item()
                #     if val_loss < min_val_loss:  # and train_loss < min_train_loss
                #         min_val_loss = val_loss
                #         min_train_loss = train_loss
                #         model_val_acc = val_acc
                #         if test_acc > best_performance:
                #             best_performance = test_acc
                #     if show_info:
                #         print(
                #             "Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f} | acc {:.4f} | val_acc {:.4f} | test_acc {:.4f}".format(
                #                 epoch, loss.item(), np.mean(dur), train_acc, val_acc, test_acc))
                #
                #         end_time = time.time()
                #         print("Each Epoch Cost Time: %f " % ((end_time - begin_time) / epoch))
                # print(f"val_score:{model_val_acc},test_score:{best_performance}")
                # if return_best:
                #     return model, model_val_acc, best_performance
                # else:
                #     return model, model_val_acc

            # print("+" * 20 + "flag here" + "+" * 20)
        except RuntimeError as e:
            if "cuda" in str(e) or "CUDA" in str(e):
                print(e)
                val_acc = 0
                test_acc = 0
            else:
                raise e
        return val_acc, test_acc

    # train from scratch
    def train(self, actions=None, format="two"):
        origin_action = actions
        actions = process_action(actions, format, self.args)
        print("train action:", actions)

        # create model
        model = self.build_gnn(actions)

        try:
            if self.args.cuda:
                model.cuda()
            # use optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            model, val_acc = self.run_model(model, optimizer, self.loss_fn, self.data, self.epochs, cuda=self.args.cuda,
                                            half_stop_score=max(self.reward_manager.get_top_average() * 0.7, 0.4))
        except RuntimeError as e:
            if "cuda" in str(e) or "CUDA" in str(e):
                print(e)
                val_acc = 0
            else:
                raise e
        reward = self.reward_manager.get_reward(val_acc)
        self.save_param(model, update_all=(reward > 0))

        self.record_action_info(origin_action, reward, val_acc)

        return reward, val_acc

    def record_action_info(self, origin_action, reward, val_acc):
        with open(self.args.dataset + "_" + self.args.search_mode + self.args.submanager_log_file, "a") as file:
            # with open(f'{self.args.dataset}_{self.args.search_mode}_{self.args.format}_manager_result.txt', "a") as file:
            file.write(str(origin_action))

            file.write(";")
            file.write(str(reward))

            file.write(";")
            file.write(str(val_acc))
            file.write("\n")

    def build_gnn(self, actions):
        print("-" * 50)
        model = GraphNet(actions, self.in_feats, self.n_classes, drop_out=self.args.in_drop, multi_label=False,
                         batch_normal=False)
        return model

    def retrain(self, actions, format="two"):
        return self.train(actions, format)

    def test_with_param(self, actions=None, format="two", with_retrain=False):
        return self.train(actions, format)

    @staticmethod
    def run_model(model, optimizer, loss_fn, data, epochs, early_stop=5, tmp_model_file="geo_citation.pkl",
                  half_stop_score=0, return_best=False, cuda=True, need_early_stop=False, show_info=False,
                  opt='w_train', grads=None):
        dur = []
        begin_time = time.time()
        best_performance = 0
        min_val_loss = float("inf")
        min_train_loss = float("inf")
        model_val_acc = 0
        features, g, labels, mask, val_mask, test_mask, n_edges = CitationGNNManager.prepare_data(data, cuda)
        print("-" * 20 + "flag here" + "-" * 20)
        for epoch in range(1, epochs + 1):
            model.train()
            t0 = time.time()
            # forward
            logits = model(features, g)
            logits = F.log_softmax(logits, 1)
            loss = loss_fn(logits[mask], labels[mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = loss.item()

            # evaluate
            model.eval()
            logits = model(features, g)
            logits = F.log_softmax(logits, 1)
            train_acc = evaluate(logits, labels, mask)
            dur.append(time.time() - t0)

            val_loss = float(loss_fn(logits[val_mask], labels[val_mask]))
            val_acc = evaluate(logits, labels, val_mask)
            test_acc = evaluate(logits, labels, test_mask)

            if val_loss < min_val_loss:  # and train_loss < min_train_loss
                min_val_loss = val_loss
                min_train_loss = train_loss
                model_val_acc = val_acc
                if test_acc > best_performance:
                    best_performance = test_acc
            if show_info:
                print(
                    "Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f} | acc {:.4f} | val_acc {:.4f} | test_acc {:.4f}".format(
                        epoch, loss.item(), np.mean(dur), train_acc, val_acc, test_acc))

                end_time = time.time()
                print("Each Epoch Cost Time: %f " % ((end_time - begin_time) / epoch))
        # print("#" * 50)
        print(f"val_score:{model_val_acc},test_score:{best_performance}")
        if return_best:
            return model, model_val_acc, best_performance
        else:
            return model, model_val_acc

    # @staticmethod
    # def run_model(model, optimizer, loss_fn, data, epochs, early_stop=5, tmp_model_file="citation_testing_2.pkl",
    #               half_stop_score=0, return_best=False, cuda=True, need_early_stop=False):
    #
    #     early_stop_manager = EarlyStop(early_stop)
    #     # initialize graph
    #     dur = []
    #     begin_time = time.time()
    #     features, g, labels, mask, val_mask, test_mask, n_edges = CitationGNNManager.prepare_data(data, cuda)
    #     saved = False
    #     best_performance = 0
    #     for epoch in range(1, epochs + 1):
    #         should_break = False
    #         t0 = time.time()
    #
    #         model.train()
    #         logits = model(features, g)
    #         logits = F.log_softmax(logits, 1)
    #         loss = loss_fn(logits[mask], labels[mask])
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #         model.eval()
    #         logits = model(features, g)
    #         logits = F.log_softmax(logits, 1)
    #         train_acc = evaluate(logits, labels, mask)
    #         train_loss = float(loss)
    #         dur.append(time.time() - t0)
    #
    #         val_loss = float(loss_fn(logits[val_mask], labels[val_mask]))
    #         val_acc = evaluate(logits, labels, val_mask)
    #         test_acc = evaluate(logits, labels, test_mask)
    #
    #         print(
    #             "Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f} | acc {:.4f} | val_acc {:.4f} | test_acc {:.4f}".format(
    #                 epoch, loss.item(), np.mean(dur), train_acc, val_acc, test_acc))
    #
    #         end_time = time.time()
    #         print("Each Epoch Cost Time: %f " % ((end_time - begin_time) / epoch))
    #         # print("Test Accuracy {:.4f}".format(acc))
    #         if early_stop_manager.should_save(train_loss, train_acc, val_loss, val_acc):
    #             saved = True
    #             torch.save(model.state_dict(), tmp_model_file)
    #             if test_acc > best_performance:
    #                 best_performance = test_acc
    #         if need_early_stop and early_stop_manager.should_stop(train_loss, train_acc, val_loss, val_acc):
    #             should_break = True
    #         if should_break and epoch > 50:
    #             print("early stop")
    #             break
    #         if half_stop_score > 0 and epoch > (epochs / 2) and val_acc < half_stop_score:
    #             print("half_stop")
    #             break
    #     if saved:
    #         model.load_state_dict(torch.load(tmp_model_file))
    #     model.eval()
    #     val_acc = evaluate(model(features, g), labels, val_mask)
    #     print(evaluate(model(features, g), labels, test_mask))
    #     if return_best:
    #         return model, val_acc, best_performance
    #     else:
    #         return model, val_acc

    @staticmethod
    def prepare_data(data, cuda=True):
        features = torch.FloatTensor(data.features)
        labels = torch.LongTensor(data.labels)
        mask = torch.ByteTensor(data.train_mask)
        test_mask = torch.ByteTensor(data.test_mask)
        val_mask = torch.ByteTensor(data.val_mask)
        n_edges = data.graph.number_of_edges()
        # create DGL graph
        g = DGLGraph(data.graph)
        # add self loop
        g.add_edges(g.nodes(), g.nodes())
        degs = g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0

        if cuda:
            features = features.cuda()
            labels = labels.cuda()
            norm = norm.cuda()
        g.ndata['norm'] = norm.unsqueeze(1)
        return features, g, labels, mask, val_mask, test_mask, n_edges
