import EZPlotly as EP
import YayROCS as YP
import pandas as pd
import scipy.stats as sc
import numpy as np
import sklearn.metrics


def manhattan_plot(df, pos_col_name, pvalue_col_name, title=None, height=None):
    # plot
    running_pos = 0
    cnt = 0
    h = [None] * 22
    for chrs in range(1, 23):
        chr_name = 'chr' + str(chrs)
        chr_df = df[df.Chr == chr_name]
        max_pos = np.max(chr_df[pos_col_name].values)
        pts_x = chrDF[pos_col_name].values + running_pos
        pts_y = -1 * np.log10(chr_df[pvalue_col_name].values)
        h[chrs - 1] = EP.scattergl(pts_x, pts_y, name=chr_name, xlabel='Chromosome', ylabel='-log10(p)', title=title,
                                   marker_size=3)
        cnt = cnt + len(chrDF)
        running_pos = running_pos + max_pos
    EP.plot_all(h, panels=np.ones((len(h),), dtype=int).tolist(), numcols=1, showlegend=True, height=height)


def chr_rolling_median(chr_pos_df, chr_val_df, rollwinsize, ylabel=None, title=None, withhold=False, xlim=None,
                       ylim=None):
    # plot
    x = chr_pos_df.values
    y = chr_val_df.rolling(rollwinsize).median()
    line_plot = EP.line(x=x, y=y, title=title, xlabel='Chr Position', ylabel=ylabel, xlim=xlim, ylim=ylim)
    if withhold:
        return line_plot
    else:
        EP.plot_all([linePlot])


def chr_count(bool_vals, chr_df, title=None, withhold=False):
    # extract unique chr vals
    uniq_vals = chr_df.unique()

    # make chromosome-level counts
    counts = np.zeros((len(uniq_vals),))
    cnt = 0
    for u in uniq_vals:
        counts[cnt] = np.sum(bool_vals[chr_df == u])
        cnt = cnt + 1

    # plot
    bar_plot = EP.bar(counts, x=uniq_vals, title=title, xlabel='Chromosome', ylabel='Count')
    if withhold:
        return bar_plot
    else:
        EP.plot_all([bar_plot])


def chr_distr(data, chr_df, distr_name, title=None, withhold=False):
    # extract unique chr vals
    uniq_vals = chr_df.unique()

    # make chromosome-level counts
    pvals = np.zeros((len(uniq_vals),))
    cnt = 0
    for u in uniq_vals:
        chr_dat = data[chr_df == u]
        res = sc.kstest(chr_dat, distr_name)
        pvals[cnt] = res[1]
        cnt = cnt + 1

    # plot
    bar_plot = EP.bar(pvals, x=uniq_vals, title=title, xlabel='Chromosome',
                      ylabel='P-value (Null: Data is ' + distr_name + ')', ylim=[0, 1])
    if withhold:
        return barPlot
    else:
        EP.plot_all([bar_plot])


def chr_count_distr(bool_vals, chr_df, title=None, withhold=False):
    # bool vals = N pos x M replicates

    # extract unique chr vals
    uniq_vals = chr_df.unique()

    # make chromosome-level counts
    means = np.zeros((len(uniq_vals),))
    stds = np.zeros((len(uniq_vals),))
    cnt = 0
    for u in uniq_vals:
        reps = bool_vals[chr_df == u]
        sums = np.sum(reps, axis=0)
        means[cnt] = np.mean(sums)
        stds[cnt] = np.std(sums)
        cnt = cnt + 1

    # plot
    bar_plot = EP.bar(y=means, x=uniq_vals, error_y=stds, title=title, xlabel='Chromosome', ylabel='Count')
    if withhold:
        return bar_plot
    else:
        EP.plot_all([bar_plot])


def chr_hist(df, chr_col, col_name, minbin=None, maxbin=None, binsize=None, title=None, xlabel=None, ylabel=None,
             histnorm=None, x_dtick=None, y_dtick=None, outfile=None):
    # labels
    if xlabel is None:
        xlabel = colName
    if ylabel is None:
        ylabel = 'Count'

    # chrs
    chrs_list = list()
    for i in range(1, 23):
        chrs_list.append('chr' + str(i))
    chrs_list.append('chrX')
    chrs_list.append('chrY')

    # extract unique chr values
    uniq_vals = df.iloc[:, chr_col].unique()

    # make chromosome level hists
    hists = list()
    for u_chr in chrs_list:
        if u_chr in uniq_vals:
            data = df[col_name][df.iloc[:, chr_col] == u_chr]
            EP_hist = EP.hist(data, title=u_chr, xlabel=xlabel, minbin=minbin, maxbin=maxbin, binsize=binsize,
                              ylabel=ylabel, color='#1ad1ff', histnorm=histnorm, x_dtick=x_dtick, y_dtick=y_dtick)
            hists.append(EP_hist)

    # make plot
    EP.plot_all(hists, numcols=5, title=title, chrpacked=True, outfile=outfile)


def chr_qq(df, chr_col, col_name, sparams=(), dist='norm', title=None, outfile=None):
    # chrs
    chrs_list = list()
    for i in range(1, 23):
        chrs_list.append('chr' + str(i))
    chrs_list.append('chrX')
    chrs_list.append('chrY')

    # extract unique chr values
    uniq_vals = df.iloc[:, chr_col].unique()

    # make chromosome level qq-plots
    plots = list()
    panels = list()
    panel_index = 1
    for u_chr in chrs_list:
        if u_chr in uniq_vals:
            data = df[col_name][df.iloc[:, chr_col] == u_chr].values
            qq = qqplot(data, sparams=sparams, dist=dist, title=u_chr)
            plots.append(qq[0])
            plots.append(qq[1])
            panels.append(panel_index)
            panels.append(panel_index)
            panel_index = panel_index + 1

    # make plot
    EP.plot_all(plots, panels=panels, numcols=5, height=1000, title=title, chrpacked=True, outfile=outfile)


def qqplot(data, sparams=(), dist='norm', title=None, name=None, marker_color='blue', line_color='red'):
    qq = sc.probplot(data, dist=dist, sparams=sparams)
    x = np.array([qq[0][0][0], qq[0][0][-1]])
    pts_scatter = EP.scattergl(x=qq[0][0], y=qq[0][1], title=title, xlabel='Expected', ylabel='Observed', markersize=5,
                               marker_color=marker_color, name=name)
    if name is None:
        name = ''
    line_plot = EP.line(x=x, y=qq[1][1] + qq[1][0] * x, width=3, color=line_color, title=title,
                        name=(name + ' (distribution=' + dist + ')'))
    return pts_scatter, line_plot


def roc(preds, gt, panel=1, names=None, title=None, xscale=None, yscale=None):
    # preds = N-ROC x L
    # gt = (L,)

    # structure
    plots = list()
    panels = list()

    # add chance curve
    p = EP.line(x=np.arange(0.0, 1.01, 0.01), y=np.arange(0.0, 1.01, 0.01), width=2, name='Chance Curve', color='black',
                xlabel='False Positive Rate', ylabel='True Positive Rate', title=title, xlim=[0, 1.01], ylim=[0, 1.01],
                xscale=xscale, yscale=yscale, x_dtick=0.1, y_dtick=0.1)
    plots.append(p)
    panels.append(panel)

    # for each predictor compute roc curve
    for i in range(0, preds.shape[0]):
        fpr, tpr = YP.roc(preds[i, :], gt)
        #		fpr,tpr,_ = sklearn.metrics.roc_curve(gt,preds[:,i])
        auc = np.round(YP.auc(fpr, tpr), 3)
        name = 'AUC=' + str(auc)
        if names is not None:
            name = names[i] + '(' + 'AUC=' + str(auc) + ')'
        p = EP.line(x=fpr, y=tpr, width=2, name=name, xlim=[0, 1.0], ylim=[0, 1.01], xscale=xscale, yscale=yscale,
                    x_dtick=0.1, y_dtick=0.1)
        plots.append(p)
        panels.append(panel)

    # return
    return plots, panels


def ecdf(data, minbin=None, maxbin=None, binsize=None, title=None, xlabel=None, ylabel=None, norm=False, name=None,
         xscale=None, yscale=None, x_dtick=None, y_dtick=None, color=None):
    # default binning
    if minbin is None:
        minbin = np.nanmin(data)
    if maxbin is None:
        maxbin = np.nanmax(data)
    if binsize is None:
        binsize = 1

    # histogram data and produce cdf
    counts, bin_edges = np.histogram(data, bins=np.arange(minbin, maxbin + binsize, binsize))
    counts_sum = np.sum(counts)
    counts = counts / counts_sum
    cdf = np.cumsum(counts)

    # plot
    if not norm:
        if ylabel is None:
            ylabel = 'Cum Freq'
        cdf_line = EP.line(x=bin_edges[:-1], y=counts_sum * cdf, title=title, xlabel=xlabel, ylabel=ylabel,
                           xlim=[minbin, maxbin + binsize], name=name, xscale=xscale, yscale=yscale, x_dtick=x_dtick,
                           y_dtick=y_dtick, color=color)
    else:
        if ylabel is None:
            ylabel = 'CDF'
        cdf_line = EP.line(x=bin_edges[:-1], y=cdf, title=title, xlabel=xlabel, ylabel=ylabel,
                           xlim=[minbin, maxbin + binsize], ylim=[0, 1.0], name=name, xscale=xScale, yscale=yscale,
                           x_dtick=x_dtick, y_dtick=y_dtick, color=color)
    return cdf_line


def rcdf(data, minbin=None, maxbin=None, binsize=None, title=None, xlabel=None, ylabel=None, norm=False, name=None,
         xscale=None,
         yscale=None, x_dtick=None, y_dtick=None, color=None):
    # default binning
    if minbin is None:
        minbin = np.nanmin(data)
    if maxbin is None:
        maxbin = np.nanmax(data)
    if binsize is None:
        binsize = 1

    # histogram data and produce cdf
    counts, bin_edges = np.histogram(data, bins=np.arange(minbin, maxbin + binsize, binsize))
    counts_sum = np.sum(counts)
    counts = counts / counts_sum
    cdf = np.cumsum(counts)

    # plot
    if not norm:
        if ylabel is None:
            ylabel = 'Reverse Cum Freq'
        cdf_line = EP.line(x=bin_edges[:-1], y=np.round(counts_sum * (1.0 - cdf), 5), title=title, xlabel=xlabel,
                           ylabel=ylabel, xlim=[minbin, maxbin + binsize], name=name, xscale=xscale, yscale=yscale,
                           x_dtick=x_dtick, y_dtick=y_dtick, color=color)
    else:
        if ylabel is None:
            ylabel = 'CDF'
        cdf_line = EP.line(x=bin_edges[:-1], y=np.round(1.0 - cdf, 5), title=title, xlabel=xlabel, ylabel=ylabel,
                           xlim=[minbin, maxbin + binsize], ylim=[0, 1.0], name=name, xscale=xscale, yscale=yscale,
                           x_dtick=x_dtick, y_dtick=y_dtick, color=color)
    return cdf_line


def corr_plot(x, y, error_y=None, xlabel=None, ylabel=None, title=None, name=None, xscale=None, yscale=None, x_dtick=None,
              y_dtick=None, xlim=None, ylim=None,plot_type='scatter'):
    corr_val = pd.core.nanops.nancorr(x, y)
    if name is None:
        name = 'corr=' + str(np.round(corr_val, 3))
    else:
        name = name + ' (' + 'corr=' + str(np.round(corr_val, 3)) + ')'
    if plot_type == 'scatter':
        corr_plot = EP.scattergl(x=x, y=y, error_y=error_y,xlabel=xlabel, ylabel=ylabel, title=title, name=name, xscale=xscale,
                                    yscale=yscale, x_dtick=x_dtick, y_dtick=y_dtick, xlim=xlim, ylim=ylim)
    elif plot_type == 'line':
        corr_plot = EP.line(x=x, y=y, error_y=error_y,xlabel=xlabel, ylabel=ylabel, title=title, name=name, xscale=xscale,
                                    yscale=yscale, x_dtick=x_dtick, y_dtick=y_dtick, xlim=xlim, ylim=ylim)
    return corr_plot


def nonparametric_ci(x, y_data, conf=0.95, ci_plot_type='line', color=None, xlabel=None, ylabel=None, title=None,
                     name=None, xscale=None, yscale=None, x_dtick=None, y_dtick=None, xlim=None, ylim=None):
    # compute confidence intervals
    m, ll, ul = YP.nonparametric_ci(data=y_data, conf=conf)

    # generate plots
    if ci_plot_type == 'line':
        if name is None:
            ci_name = ' (CI=' + str(conf * 100.0) + '%' + ')'
        else:
            ci_name = name + ' (CI=' + str(conf * 100.0) + '%' + ')'
        ll_pl = EP.line(x=x, y=ll, color=color, dash='dot', name=ci_name)
        ul_pl = EP.line(x=x, y=ul, color=color, dash='dot', name=ci_name)
        mean_pl = EP.line(x=x, y=m, color=color, name=name, xlabel=xlabel, ylabel=ylabel, title=title, xscale=xscale,
                          yscale=yscale, x_dtick=x_dtick, y_dtick=y_dtick, xlim=xlim, ylim=ylim)
        return mean_pl, ll_pl, ul_pl
    elif ci_plot_type == 'point':
        return EP.line(x=x, y=m, ucl=np.abs(ul - m), lcl=np.abs(ll - m), color=color, name=name, xlabel=xlabel,
                       ylabel=ylabel, title=title,
                       xscale=xscale, yscale=yscale, x_dtick=x_dtick, y_dtick=y_dtick, xlim=xlim, ylim=ylim)