import argparse
from dataset import Dataset
from trainer import Trainer
from tester import Tester
from params import Params
from utils import *
import os
import shutil
import random
import torch
import numpy as np

# set the random seed
RANDOM_SEED = 1299

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


print("load data")
desc = "Temporal Knowledge Graph Completion based Graph Convolutional Networks and Diachronic Embedding"
parser = argparse.ArgumentParser(description=desc)

parser.add_argument(
    "-dataset",
    help="Dataset",
    type=str,
    default="icews14",
    choices=["icews14", "icews05-15", "gdelt"],
)
parser.add_argument(
    "-model",
    help="Model",
    type=str,
    default="CCGCN",
)
parser.add_argument("-ne", help="Number of epochs", type=int, default=1000)
parser.add_argument("-bsize", help="Batch size", type=int, default=512)
parser.add_argument("-lr", help="Learning rate", type=float, default=0.001)
parser.add_argument(
    "-reg_lambda", help="L2 regularization parameter", type=float, default=0.0
)
parser.add_argument(
    "-gcn_inp_dim", help="Initial gcn embedding dimension", type=int, default=100
)
parser.add_argument(
    "-gcn_hid_dim", help="Hidden gcn embedding dimension", type=int, default=150
)
parser.add_argument("-emb_dim", help="Embedding final dimension", type=int, default=100)
parser.add_argument("-tim_dim", help="Embedding time dimension", type=int, default=50)
parser.add_argument("-neg_ratio", help="Negative ratio", type=int, default=500)
parser.add_argument("-dropout", help="Dropout probability", type=float, default=0.4)
parser.add_argument("-inp_drop", help="Input Dropout", type=float, default=0.1)
parser.add_argument(
    "-hid_drop", help="Hidde Dropout in GCN layer", type=float, default=0.3
)
parser.add_argument("-gcn_drop", help="After GCN Dropout", type=float, default=0.1)
parser.add_argument("-fin_drop", help="Final Dropout", type=float, default=0.2)
parser.add_argument(
    "-save_each", help="Save model and validate each K epochs", type=int, default=50
)
parser.add_argument(
    "-se_prop", help="Static embedding proportion", type=float, default=0.2
)
parser.add_argument(
    "-run_id", help="the name of planned strategy row id", type=str, default="0"
)
parser.add_argument("-is_gcn", help="is applying gcn layers", type=int, default=1)
parser.add_argument("-start_epoch", help="start epoch", type=int, default=0)
parser.add_argument("-opn", help="composition operation", type=str, default="mult")

args = parser.parse_args()

dataset = Dataset(args.dataset)
edge_index, edge_type = dataset.construct_adj()
time_dct = get_time_dct()

# create or replace log files to track the training and testing process
log_train_dir = create_logging_file(args, time_dct, "train")
log_valid_dir = create_logging_file(args, time_dct, "valid")
log_test_dir = create_logging_file(args, time_dct, "test")

# remove the old model at run_id if it exists
model_path = f"./models/{args.model}/id_{args.run_id}"
if args.start_epoch == 0:
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    else:
        os.makedirs(model_path)


device = auto_device()
print("used device: ", auto_device())
# create the parameters object
params = Params(
    run_id=args.run_id,
    ne=args.ne,
    bsize=args.bsize,
    lr=args.lr,
    reg_lambda=args.reg_lambda,
    gcn_inp_dim=args.gcn_inp_dim,
    gcn_hid_dim=args.gcn_hid_dim,
    emb_dim=args.emb_dim,
    tim_dim=args.tim_dim,
    neg_ratio=args.neg_ratio,
    dropout=args.dropout,
    save_each=args.save_each,
    se_prop=args.se_prop,
    lg_train_dir=log_train_dir,
    lg_valid_dir=log_valid_dir,
    lg_test_dir=log_test_dir,
    time_dct=time_dct,
    inp_drop=args.inp_drop,
    hid_drop=args.hid_drop,
    gcn_drop=args.gcn_drop,
    fin_drop=args.fin_drop,
    is_gcn=args.is_gcn,
    device=device,
    start_epoch=args.start_epoch,
    opn=args.opn,
)
# reg = N3(weight=1e-2)
# training the model and validating it on the validation set
trainer = Trainer(dataset, params, args.model, edge_index, edge_type)
best_model_path, best_epoch = trainer.train_and_valid()

# testing the best chosen model on the test set
print(f"Best epoch: {best_epoch}")
tester = Tester(
    dataset,
    params,
    best_model_path,
    args.model,
    "test",
    log_test_dir,
    edge_index,
    edge_type,
    epoch=best_epoch,
)
tester.test()
tester.close_logger(is_write=True)
