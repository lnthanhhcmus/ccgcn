import logging
import datetime
import os
import numpy as np, sys, os, json
import logging, logging.config
import torch
from torch.nn.init import xavier_normal_, uniform_, xavier_uniform_
from torch.nn import Parameter

np.set_printoptions(precision=4)


def get_time_dct():
    result = {}
    current_time = datetime.datetime.now() + datetime.timedelta(hours=7)
    result["date"] = current_time.strftime("%Y%m%d")
    result["time"] = current_time.strftime("%H%M00")
    result["timestamp"] = current_time.strftime("%Y-%m-%d\t%H:%M:%S")
    return result


def create_logging_file(args, time_dct: dict, run_type="train"):
    try_flag = "try_" if args.ne <= 10 else ""
    folder_name = f"./logs/{try_flag}{args.model}/id_{args.run_id}"

    file_dir = f"{folder_name}/{run_type}_{args.dataset}.txt"

    if args.start_epoch > 0:
        return file_dir
    else:
        if os.path.exists(file_dir):
            os.remove(file_dir)
            open(file_dir, "a").close()

        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)

    return file_dir


def set_gpu(gpus):
    """
    Sets the GPU to be used for the run

    Parameters
    ----------
    gpus:           List of GPUs to be used for the run

    Returns
    -------

    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def get_logger(name, log_dir, config_dir):
    """
    Creates a logger object

    Parameters
    ----------
    name:           Name of the logger file
    log_dir:        Directory where logger file needs to be stored
    config_dir:     Directory from where log_config.json needs to be read

    Returns
    -------
    A logger object which writes to both file and stdout

    """
    config_dict = json.load(open(config_dir + "log_config.json"))
    config_dict["handlers"]["file_handler"]["filename"] = log_dir + name.replace(
        "/", "-"
    )
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(name)

    std_out_format = "%(asctime)s - [%(levelname)s] - %(message)s"
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger


def get_combined_results(left_results, right_results):
    results = {}
    count = float(left_results["count"])

    results["left_mr"] = round(left_results["mr"] / count, 5)
    results["left_mrr"] = round(left_results["mrr"] / count, 5)
    results["right_mr"] = round(right_results["mr"] / count, 5)
    results["right_mrr"] = round(right_results["mrr"] / count, 5)
    results["mr"] = round((left_results["mr"] + right_results["mr"]) / (2 * count), 5)
    results["mrr"] = round(
        (left_results["mrr"] + right_results["mrr"]) / (2 * count), 5
    )

    for k in range(10):
        results["left_hits@{}".format(k + 1)] = round(
            left_results["hits@{}".format(k + 1)] / count, 5
        )
        results["right_hits@{}".format(k + 1)] = round(
            right_results["hits@{}".format(k + 1)] / count, 5
        )
        results["hits@{}".format(k + 1)] = round(
            (
                left_results["hits@{}".format(k + 1)]
                + right_results["hits@{}".format(k + 1)]
            )
            / (2 * count),
            5,
        )
    return results


def com_mult(a, b):
    r1, i1 = torch.real(a), torch.imag(a)
    r2, i2 = torch.real(b), torch.imag(b)
    return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)


def cconv(a, b):
    return torch.fft.irfft(
        com_mult(torch.fft.rfft(a, 1), torch.fft.rfft(b, 1)),
        1,
        signal_sizes=(a.shape[-1],),
    )


def ccorr(a, b):
    # print(a.shape[-1])
    return torch.fft.irfft(
        com_mult(torch.conj(torch.fft.rfft(a, 1)), torch.fft.rfft(b, 1)), n=a.shape[-1]
    ).squeeze(1)


def complex_ccorr(a, b):
    re_a, im_a = a
    re_b, im_b = b
    re_c = ccorr(re_a, re_b)
    im_c = ccorr(im_a, im_b)
    # print(re_c.shape, a[0].shape)
    return (re_c, im_c)


def complex_add(a, b):
    re_a, im_a = a
    re_b, im_b = b

    re_c = re_a + re_b
    im_c = im_a + im_b

    return (re_c, im_c)


def complex_sub(a, b):
    re_a, im_a = a
    re_b, im_b = b

    re_c = re_a - re_b
    im_c = im_a - im_b

    return (re_c, im_c)


def complex_mult(a, b):
    re_a, im_a = a
    re_b, im_b = b

    re_c = re_a * re_b - im_a * im_b
    im_c = re_a * im_b + im_a * re_b

    return (re_c, im_c)


def complex_hermitian(a, b):  # hermitian
    re_a, im_a = a
    re_b, im_b = b

    re_c = re_a * re_b + im_a * im_b
    im_c = re_a * im_b - im_a * re_b

    return (re_c, im_c)


def complex_conjugate(a):
    re_a, im_a = a
    re_c, im_c = re_a, -im_a
    return (re_c, im_c)


def complex_simple(a, b):
    conj_b = complex_conjugate(b)

    x1 = complex_mult(a, b)
    x2 = complex_mult(a, conj_b)

    re_c, im_c = complex_add(x1, x2)
    re_c, im_c = re_c / 2.0, im_c / 2.0

    return (re_c, im_c)


def auto_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    return device


def get_param(shape, is_rel=False, device=auto_device()):
    param = Parameter(torch.Tensor(*shape)).to(auto_device())
    xavier_uniform_(param.data, gain=0.7071067811865475)
    return param


def get_embeding(n, dim, device=auto_device()):
    embedding = torch.nn.Embedding(n, dim).to(device)
    xavier_uniform_(embedding.weight, gain=0.7071067811865475)
    return embedding
