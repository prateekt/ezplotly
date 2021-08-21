from typing import Tuple, List
import numpy as np
from sklearn.neighbors import KernelDensity
import scipy.signal as ss


def roc(pred: np.array, gt: np.array) -> Tuple[np.array, np.array]:
    """
    Generates calculations for a Receiver Operating Characteristic (ROC) curve.

    :param pred: Predictor vector as `np.array[float]`
    :param gt: Ground truth labels vector as `np.array[float]`
    :return:
        fpr: False Positive Rate (FPR) data vector of ROC curve as `np.array[float]`
        tpr: True Positive Rate (TPR) data vector of ROC curve as `np.array[float]`
    """

    # make sure pred /gt are correct shape
    if (pred.ndim == 2) and (pred.shape[1] > pred.shape[0]):
        pred = pred.T
    elif pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    if (gt.ndim == 2) and (gt.shape[1] > gt.shape[0]):
        gt = gt.T
    elif gt.ndim == 1:
        gt = gt.reshape(-1, 1)

    # make results table
    results = np.concatenate([pred, gt], axis=1)

    # compute step sizes
    a = results.shape[0]
    num_pos = np.sum(results[:, 1])
    num_neg = a - num_pos
    step_x = 1.0 / num_neg
    step_y = 1.0 / num_pos
    current_x = 0.0
    current_y = 0.0

    # sort results
    ind = np.lexsort((results[:, 1], -1 * results[:, 0]))
    results = results[ind]

    # roc compute
    fpr = np.zeros((a + 1,))
    tpr = np.zeros((a + 1,))
    cnt = 1
    for i in range(0, a):
        if results[i, 1] == 1:
            current_y = current_y + step_y
            fpr[cnt] = current_x
            tpr[cnt] = current_y
        else:
            current_x = current_x + step_x
            fpr[cnt] = current_x
            tpr[cnt] = current_y
        cnt = cnt + 1

    # resize and return
    tpr = tpr[0:cnt]
    fpr = fpr[0:cnt]
    return fpr, tpr


def auc(x, y) -> float:
    """
    Calculates AUC of a roc curve. Inputs 1 ROC curve and outputs the AUC of it.

    :param x: FPR data vector of roc curve as `np.array[float]`
    :param y: TPR data vector of roc curve as `np.array[float]`
    :return:
        AUC: The Area Under the Curve (AUC) as `float`
    """
    # align to reference binning
    ref_binning = np.arange(0, 1.1, 0.01)
    bin_ind = 0
    auc_val = 0
    for i in range(0, len(ref_binning)):
        # find appropriate bin to draw
        while (bin_ind < (len(x) - 1)) and (ref_binning[i] > x[bin_ind]):
            bin_ind = bin_ind + 1
        auc_val = auc_val + y[bin_ind]
    auc_val = auc_val / len(ref_binning)
    return auc_val


def compute_p_value(
    data_sample: np.array, observed: float, tailed: str = "two-tailed"
) -> float:
    """
    Computes a p-value of an observed point relative to a data distribution based on rank of
    an observed point relative to a data sample. Computes either a left, right, or two-tailed p-value.

    :param data_sample: The data sample of the original distribution as `np.array[float]`
    :param observed: The observed point as `float`
    :param tailed: Whether to produce a right, left, or two-tailed p-value as `str`
    :return:
        p_val: The computed p-value as `float`
    """
    # combine data and sort
    combined_data = np.append(data_sample, observed)
    combined_data.sort()
    n = len(combined_data)

    # identify ranks
    rank = np.where(combined_data == observed)
    lrank = rank[0][0]
    rrank = n - rank[0][-1] - 1

    # tails
    lp = lrank / n
    rp = rrank / n
    if tailed == "left":
        p_val = lp
    elif tailed == "right":
        p_val = rp
    elif tailed == "two-tailed":
        p_val = np.min([lp, rp])
    else:
        raise ValueError("tailed parameter be must {left, right, two-tailed}.")
    return p_val


def nonparametric_ci(
    data: np.array, conf: float = 0.95
) -> Tuple[np.array, np.array, np.array]:
    """
    Compute nonparametric confidence intervals on data.

    :param data: The data sample as `np.array[float].`
        data is a nxd (# samples by data points) matrix.
    :param conf: The % confidence interval as `float`
    :return:
        m: sample mean as `np.array`
        ll: lower confidence limit as `np.array`
        ul: upper confidence limit as `np.array`
    """
    # shapes
    n = data.shape[0]
    d = data.shape[1]
    m = np.zeros((d,))
    ul = np.zeros((d,))
    ll = np.zeros((d,))

    # indices
    ul_ind = int(np.floor(conf * n))
    ll_ind = int(np.floor((1 - conf) * n))

    # loop
    for d_cnt in range(0, d):
        samples = np.sort(data[:, d_cnt])
        ul[d_cnt] = samples[ul_ind]
        ll[d_cnt] = samples[ll_ind]
        m[d_cnt] = np.mean(samples)

    # return
    return m, ll, ul


def pdf(
    data: np.array, bin_edges: np.array, method: str = "KDE"
) -> Tuple[np.array, np.array]:
    """
    Generates probability distribution function (PDF) of data. Allows histogram and Kernel Density Estimation (KDE)
    methods of producing PDFs.

    :param data: The data as `np.array[float]`
    :param bin_edges: Bin edges list as [left_bin_edge, right_bin_edge] for estimation as `np.array[float]`
    :param method: Estimation method as `str` (either hist or KDE).
    :return:
        distr: The estimated pdf as `np.array`
        bin_centers: The bin centers as `np.array`
    """
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges + bin_width / 2.0
    bin_centers = bin_centers[:-1]
    if method == "KDE":
        kde = KernelDensity(kernel="gaussian", bandwidth=1.0).fit(data)
        distr = kde.score_samples(bin_edges)
        distr = np.exp(distr)
    elif method == "hist":
        distr = np.histogram(data, bins=bin_edges.squeeze())[0]
    else:
        raise ValueError("method parameter must be hist or KDE.")
    distr = distr / np.sum(distr)
    bin_centers = bin_centers.squeeze()
    return distr, bin_centers


def find_modes(
    data: np.array, bin_edges: np.array, method: str = "KDE"
) -> Tuple[np.array, np.array]:
    """
    Find extrema in data.

    :param data: Data sample as `np.array[float]`
    :param bin_edges: The bin the edges for pdf as `np.array[float]`
    :param method: The method of pdf estimation as `str`
    :return:
        x_maxima: The x-dimension of maxima as `np.array[float]`
        y_maxima: The y-dimension of maxima as `np.array[float]`
    """

    # get data distribution
    distr, bin_centers = pdf(data=data, bin_edges=bin_edges, method=method)

    # find relative maxima
    maxima_inds = ss.argrelextrema(distr, np.greater)[0]
    x_maxima = bin_edges[maxima_inds].squeeze()
    y_maxima = distr[maxima_inds].squeeze()
    return x_maxima, y_maxima
