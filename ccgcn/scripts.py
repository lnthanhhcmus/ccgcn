import torch
from utils import *

device = auto_device()


def shredFacts(facts):  # takes a batch of facts and shreds it into its columns

    heads = torch.tensor(facts[:, 0]).long().to(device)
    rels = torch.tensor(facts[:, 1]).long().to(device)
    tails = torch.tensor(facts[:, 2]).long().to(device)
    years = torch.tensor(facts[:, 3]).float().to(device)
    months = torch.tensor(facts[:, 4]).float().to(device)
    days = torch.tensor(facts[:, 5]).float().to(device)
    intervals = torch.tensor(facts[:, 6]).float().to(device)
    return heads, rels, tails, years, months, days, intervals
