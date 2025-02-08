import os
import time
import torch
import torch.nn as nn
from de_variant import *  # 33
from tester import Tester
import logging
import datetime


class Trainer:
    def __init__(
        self, dataset, params, model_name, edge_index, edge_type, regularization=None
    ):
        instance_gen = globals()[model_name]
        self.model_name = model_name
        self.model = nn.DataParallel(instance_gen(dataset=dataset, params=params))
        self.dataset = dataset
        self.params = params
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.FileHandler(self.params.lg_train_dir))
        self.device = params.device
        self.edge_index = edge_index
        self.edge_type = edge_type
        # self.reg = regularization
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.params.lr,
            weight_decay=self.params.reg_lambda,
        )  # weight_decay corresponds to L2 regularization

        self.loss_func = nn.CrossEntropyLoss()
        self.start_epoch = params.start_epoch // params.save_each * params.save_each

    def train_and_valid(self, early_stop=False):
        self.logger.info(f"device: {self.device}")
        best_epoch = -1
        best_mrr = -1.0
        best_model = ""
        if self.start_epoch > 0:
            best_epoch, best_mrr, best_model = self.loadModel(self.start_epoch)
            self.logger.info(f"load at epoch: {self.start_epoch}")

        self.model.train()
        round_func = lambda x, y=3: round(x, y)
        for epoch in range(1 + self.start_epoch, self.params.ne + 1):
            last_batch = False
            total_loss = 0.0
            start = time.time()

            while not last_batch:
                heads, rels, tails, years, months, days, intervals = (
                    self.dataset.nextBatch(
                        self.params.bsize, neg_ratio=self.params.neg_ratio
                    )
                )
                last_batch = self.dataset.wasLastBatch()

                scores = self.model(
                    heads,
                    rels,
                    tails,
                    years,
                    months,
                    days,
                    intervals,
                    self.edge_index,
                    self.edge_type,
                )
                ###Added for softmax####
                num_examples = int(heads.shape[0] / (1 + self.params.neg_ratio))
                # print(heads.shape[0], num_examples)
                scores_reshaped = scores.view(num_examples, self.params.neg_ratio + 1)
                #print(scores_reshaped.shape)
                l = torch.zeros(num_examples).long().to(self.device)
                loss = self.loss_func(scores_reshaped, l)

                # loss_reg = self.reg(factors)
                # loss = loss_fit + 0 if not loss_reg else loss_reg

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.cpu().item()

            self.logger.info(
                f"Epoch {epoch}: loss={round_func(total_loss)} time={round_func(time.time()-start)}"
            )

            if epoch % self.params.save_each == 0:
                # save the model
                self.saveModel(epoch, total_loss, best_epoch, best_mrr, best_model)

                # validate the model
                model_path = f"./models/{self.model_name}/id_{self.params.str_()}/{self.dataset.name}_cpt_{epoch}.chkpnt"
                tester = Tester(
                    self.dataset,
                    self.params,
                    model_path,
                    self.model_name,
                    "valid",
                    lg_dir=self.params.lg_valid_dir,
                    epoch=epoch,
                    edge_index=self.edge_index,
                    edge_type=self.edge_type,
                )
                mrr = tester.test()
                if mrr > best_mrr:
                    best_mrr = mrr
                    best_model = model_path
                    best_epoch = epoch

                tester.close_logger()

        return best_model, best_epoch

    def saveModel(self, checkpoint, total_loss, best_epoch, best_mrr, best_model):
        directory = f"./models/{self.model_name}/id_{self.params.str_()}"
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(
            {
                "epoch": checkpoint,
                "best_epoch": best_epoch,
                "best_mrr": best_mrr,
                "best_model": best_model,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": total_loss,
            },
            f"{directory}/{self.dataset.name}_cpt_{checkpoint}.chkpnt",
        )

    def loadModel(self, cpt):
        directory = f"./models/{self.model_name}/id_{self.params.str_()}"
        checkpoint_path = f"{directory}/{self.dataset.name}_cpt_{cpt}.chkpnt"

        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_epoch = checkpoint["best_epoch"]
        best_mrr = checkpoint["best_mrr"]
        best_model = checkpoint["best_model"]
        return best_epoch, best_mrr, best_model
