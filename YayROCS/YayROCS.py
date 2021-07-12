import numpy as np
from sklearn.neighbors import KernelDensity
import scipy.signal as ss


# input = predictor vector, ground truth vector
def roc(pred, gt):
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


# input 1 roc (X,Y)
def auc(x, y):
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


# compute p-value based on distribution and observed point
def compute_p_value(distr, observed, tailed='two-tailed'):
    # combine data and sort
    combined_data = np.append(distr, observed)
    combined_data.sort()
    n = len(combined_data)

    # identify ranks
    rank = np.where(combined_data == observed)
    lrank = rank[0][0]
    rrank = n - rank[0][-1] - 1

    # tails
    lp = lrank / n
    rp = rrank / n
    if tailed == 'left':
        p_val = lp
    elif tailed == 'right':
        p_val = rp
    elif tailed == 'two-tailed':
        p_val = np.min([lp, rp])
    return p_val


# compute nonparametric confidence intervals
# data is nxd (# samples by data points)
def nonparametric_ci(data, conf=0.95):
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


# generate pdf
def pdf(data, bin_edges, method='KDE'):
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges + bin_width / 2.0
    bin_centers = bin_centers[:-1]
    if method == 'KDE':
        kde = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(data)
        distr = kde.score_samples(bin_edges)
        distr = np.exp(distr)
    elif method == 'hist':
        distr = np.histogram(data, bins=bin_edges.squeeze())[0]
    distr = distr / np.sum(distr)
    bin_centers = bin_centers.squeeze()
    return distr, bin_centers


# find modes
def find_modes(data, bin_edges, method='KDE'):
    # get data distribution
    distr, bin_centers = pdf(data=data, bin_edges=bin_edges, method=method)

    # find relative maxima
    maxima_inds = ss.argrelextrema(distr, np.greater)[0]
    x_maxima = bin_edges[maxima_inds].squeeze()
    y_maxima = distr[maxima_inds].squeeze()
    return x_maxima, y_maxima
