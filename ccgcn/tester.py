import torch
import numpy as np
from scripts import shredFacts
from de_variant import *
from measure import Measure
import logging


class Tester:
    def __init__(
        self,
        dataset,
        params,
        model_path,
        model_name,
        valid_or_test,
        lg_dir,
        edge_index,
        edge_type,
        epoch=-1,
    ):
        self.model_path = model_path
        instance_gen = globals()[model_name]
        self.model = nn.DataParallel(instance_gen(dataset=dataset, params=params))
        self.loadModel(model_path)
        self.params = params
        self.dataset = dataset
        self.valid_or_test = valid_or_test
        self.measure = Measure()
        self.epoch = epoch
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.FileHandler(lg_dir))
        self.edge_index = edge_index
        self.edge_type = edge_type

    def loadModel(self, model_path):
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def getRank(self, sim_scores):  # assuming the test fact is the first one
        return (sim_scores > sim_scores[0]).sum() + 1

    def replaceAndShred(self, fact, raw_or_fil, head_or_tail):
        head, rel, tail, years, months, days, intervals = fact
        if head_or_tail == "head":
            ret_facts = [
                (i, rel, tail, years, months, days, intervals)
                for i in range(self.dataset.numEnt())
            ]
        if head_or_tail == "tail":
            ret_facts = [
                (head, rel, i, years, months, days, intervals)
                for i in range(self.dataset.numEnt())
            ]

        if raw_or_fil == "raw":
            ret_facts = [tuple(fact)] + ret_facts
        elif raw_or_fil == "fil":
            ret_facts = [tuple(fact)] + list(
                set(ret_facts) - self.dataset.all_facts_as_tuples
            )

        return shredFacts(np.array(ret_facts))

    def test(self):
        # write checkpoint row to validation log file
        if self.valid_or_test == "valid":
            self.logger.info(f"Validation results at epoch: {self.epoch}")
        else:
            self.logger.info(
                f"Test results at best epoch (epoch has maximum mrr in filtering setting): {self.epoch}"
            )

        # compute metrics
        for i, fact in enumerate(self.dataset.data[self.valid_or_test]):
            settings = ["fil"]
            for raw_or_fil in settings:
                for head_or_tail in ["head", "tail"]:
                    heads, rels, tails, years, months, days, intervals = (
                        self.replaceAndShred(fact, raw_or_fil, head_or_tail)
                    )
                    sim_scores = self.model(
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
                    sim_scores = sim_scores.cpu().data.numpy()
                    rank = self.getRank(sim_scores)
                    self.measure.update(rank, raw_or_fil)

        self.measure.normalize(len(self.dataset.data[self.valid_or_test]))

        # write metrics to log file
        round_func = lambda x, y=3: round(x, y)
        for raw_or_fil in ["fil"]:
            self.logger.info(f"Validation {raw_or_fil} setting:")
            self.logger.info(f"\tHit@1 = {round_func(self.measure.hit1[raw_or_fil])}")
            self.logger.info(f"\tHit@3 = {round_func(self.measure.hit3[raw_or_fil])}")
            self.logger.info(f"\tHit@10 = {round_func(self.measure.hit10[raw_or_fil])}")
            self.logger.info(f"\tMR = {round_func(self.measure.mr[raw_or_fil], 0)}")
            self.logger.info(f"\tMRR = {round_func(self.measure.mrr[raw_or_fil])}")
            self.logger.info("#" * 50)

        return self.measure.mrr["fil"]

    def close_logger(self, is_write=False):
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)

    def get_emb_analysis(self, load_path, save_path, model_path):

        for i, fact in enumerate(self.dataset.data[self.valid_or_test]):
            settings = ["fil"]
            for raw_or_fil in settings:
                for head_or_tail in ["head", "tail"]:
                    heads, rels, tails, years, months, days, intervals = (
                        self.replaceAndShred(fact, raw_or_fil, head_or_tail)
                    )
                    sim_scores = self.model(
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
                    sim_scores = sim_scores.cpu().data.numpy()
                    rank = self.getRank(sim_scores)
                    self.measure.update(rank, raw_or_fil)
