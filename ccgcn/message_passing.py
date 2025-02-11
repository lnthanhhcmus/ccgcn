import inspect, torch
from torch_scatter import scatter


def scatter_(name, src, index, dim_size=None):
    r"""Aggregates all values from the :attr:`src` tensor at the indices
    specified in the :attr:`index` tensor along the first dimension.
    If multiple indices reference the same location, their contributions
    are aggregated according to :attr:`name` (either :obj:`"add"`,
    :obj:`"mean"` or :obj:`"max"`).

    Args:
            name (string): The aggregation to use (:obj:`"add"`, :obj:`"mean"`,
                    :obj:`"max"`).
            src (Tensor): The source tensor.
            index (LongTensor): The indices of elements to scatter.
            dim_size (int, optional): Automatically create output tensor with size
                    :attr:`dim_size` in the first dimension. If set to :attr:`None`, a
                    minimal sized output tensor is returned. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    if name == "add":
        name = "sum"
    assert name in ["sum", "mean", "max"]
    out = scatter(src, index, dim=0, out=None, dim_size=dim_size, reduce=name)
    return out[0] if isinstance(out, tuple) else out


class MessagePassing(torch.nn.Module):
    The code will be made available after the first round of review.
