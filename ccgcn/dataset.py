import numpy as np
from scripts import shredFacts
import torch
import datetime
from utils import *
from ordered_set import OrderedSet


class Dataset:
    """Implements the specified dataloader"""

    def __init__(self, ds_name):
        """
        Params:
                ds_name : name of the dataset
        """
        self.name = ds_name
        # self.ds_path = "<path-to-dataset>" + ds_name.lower() + "/"
        self.ds_path = "datasets/" + ds_name.lower() + "/"
        self.ent2id = {}
        self.rel2id = {}
        self.rel_set = OrderedSet()
        self.start_date = datetime.datetime.now()
        self.data = {
            "train": self.readFile(self.ds_path + "train.txt"),
            "valid": self.readFile(self.ds_path + "valid.txt"),
            "test": self.readFile(self.ds_path + "test.txt"),
        }

        self.rel2id.update(
            {
                rel + "_reverse": idx + len(self.rel2id)
                for idx, rel in enumerate(self.rel_set)
            }
        )

        self.start_batch = 0
        self.all_facts_as_tuples = None
        self.device = auto_device()

        # self.add_inverse_quadlets()

        self.convertTimes()

        self.all_facts_as_tuples = set(
            [
                tuple(d)
                for d in self.data["train"] + self.data["valid"] + self.data["test"]
            ]
        )

        for spl in ["train", "valid", "test"]:
            self.data[spl] = np.array(self.data[spl])

    def readFile(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            data = f.readlines()

        facts = []
        for line in data:
            elements = line.strip().split("\t")
            head_id = self.getEntID(elements[0])
            rel_id = self.getRelID(elements[1])
            tail_id = self.getEntID(elements[2])
            timestamp = elements[3]
            t = datetime.datetime.strptime(timestamp, "%Y-%m-%d")
            if t < self.start_date:
                self.start_date = t
            self.rel_set.add(elements[1])

            facts.append([head_id, rel_id, tail_id, timestamp])
        return facts

    def add_inverse_quadlets(self):
        inverse_facts = []
        num_rels = self.numRel()
        for d in self.data["train"]:
            head_id, rel_id, tail_id, timestamp = d
            inverse_facts.append([tail_id, rel_id + num_rels, head_id, timestamp])

        # add inverse facts to train
        self.data["train"].extend(inverse_facts)

    def convertTimes(self):
        """
        This function spits the timestamp in the day, date and time.
        """
        for split in ["train", "valid", "test"]:
            for i, fact in enumerate(self.data[split]):
                fact_date = fact[-1]
                self.data[split][i] = self.data[split][i][:-1]
                date = list(map(float, fact_date.split("-")))
                intervals = (
                    datetime.datetime.strptime(fact_date, "%Y-%m-%d") - self.start_date
                )
                date.append(intervals.days)
                self.data[split][i] += date

    def numEnt(self):
        return len(self.ent2id)

    def numRel(self):
        return len(self.rel2id) // 2

    def getEntID(self, ent_name):
        if ent_name in self.ent2id:
            return self.ent2id[ent_name]
        self.ent2id[ent_name] = len(self.ent2id)
        return self.ent2id[ent_name]

    def getRelID(self, rel_name):
        if rel_name in self.rel2id:
            return self.rel2id[rel_name]
        self.rel2id[rel_name] = len(self.rel2id)
        return self.rel2id[rel_name]

    def nextPosBatch(self, batch_size):
        if self.start_batch + batch_size > len(self.data["train"]):
            ret_facts = self.data["train"][self.start_batch :]
            self.start_batch = 0
        else:
            ret_facts = self.data["train"][
                self.start_batch : self.start_batch + batch_size
            ]
            self.start_batch += batch_size
        return ret_facts

    def addNegFacts(self, bp_facts, neg_ratio):
        ex_per_pos = 2 * neg_ratio + 2
        facts = np.repeat(np.copy(bp_facts), ex_per_pos, axis=0)
        for i in range(bp_facts.shape[0]):
            s1 = i * ex_per_pos + 1
            e1 = s1 + neg_ratio
            s2 = e1 + 1
            e2 = s2 + neg_ratio

            facts[s1:e1, 0] = (
                facts[s1:e1, 0]
                + np.random.randint(low=1, high=self.numEnt(), size=neg_ratio)
            ) % self.numEnt()
            facts[s2:e2, 2] = (
                facts[s2:e2, 2]
                + np.random.randint(low=1, high=self.numEnt(), size=neg_ratio)
            ) % self.numEnt()

        return facts

    def addNegFacts2(self, bp_facts, neg_ratio):
        pos_neg_group_size = 1 + neg_ratio
        facts1 = np.repeat(np.copy(bp_facts), pos_neg_group_size, axis=0)
        facts2 = np.copy(facts1)
        rand_nums1 = np.random.randint(low=1, high=self.numEnt(), size=facts1.shape[0])
        rand_nums2 = np.random.randint(low=1, high=self.numEnt(), size=facts2.shape[0])

        for i in range(facts1.shape[0] // pos_neg_group_size):
            rand_nums1[i * pos_neg_group_size] = 0
            rand_nums2[i * pos_neg_group_size] = 0

        facts1[:, 0] = (facts1[:, 0] + rand_nums1) % self.numEnt()
        facts2[:, 2] = (facts2[:, 2] + rand_nums2) % self.numEnt()
        return np.concatenate((facts1, facts2), axis=0)

    def nextBatch(self, batch_size, neg_ratio=1):
        bp_facts = self.nextPosBatch(batch_size)
        batch = shredFacts(self.addNegFacts2(bp_facts, neg_ratio))
        return batch

    def wasLastBatch(self):
        return self.start_batch == 0

    def construct_adj(self):
        edge_index, edge_type = [], []

        for quad in self.data["train"]:
            edge_index.append((quad[0], quad[2]))
            edge_type.append(quad[1])

        # Adding inverse edges
        for quad in self.data["train"]:
            edge_index.append((quad[2], quad[0]))
            edge_type.append(quad[1] + self.numRel())

        edge_index = torch.LongTensor(edge_index).to(self.device).t()
        edge_type = torch.LongTensor(edge_type).to(self.device)

        return edge_index, edge_type
