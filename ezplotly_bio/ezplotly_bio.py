from typing import Optional, List, Any, Sequence, Tuple, Union
from ezplotly import EZPlotlyPlot
import pandas as pd
import scipy.stats as sc
import numpy as np
import ezplotly as ep
import yayrocs as yr


def manhattan_plot(
    df: pd.DataFrame,
    title: Optional[str] = None,
    height: Optional[int] = None,
    withhold: bool = False,
    outfile: Optional[str] = None,
) -> Optional[List[EZPlotlyPlot]]:
    """
    Plot a manhattan plot of p-values from a pandas dataframe containing chromosome position p-values.
    Makes a plot separately for each chromosome specified in dataframe.

    :param df: A `pd.DataFrame` containing columns (chr, pos, pval).
    :param title: The title of the overall plot as `Optional[str]`
    :param height: The height of the plot as `Optional[int]`
    :param withhold: Whether to plot the plot or withhold and return the plot to the user as `bool`
    :param outfile: The file to write plot to as `Optional[str]`
    :return:
        figs_list: `Optional[List[EZPlotlyPlot]]` of figures containing produced EZPlotlyPlot set
    """

    # prepare manhattan plots per chromosome
    running_pos = 0
    cnt = 0
    figs_list = list()
    panels = [1 for _ in range(1, 25)]
    for chrs in range(1, 25):
        if chrs <= 22:
            chr_name = "chr" + str(chrs)
        elif chrs == 23:
            chr_name = "chrX"
        elif chrs == 24:
            chr_name = "chrY"
        else:
            raise ValueError("Invalid state.")
        chr_df = df[df.chr == chr_name]
        if len(chr_df) == 0:
            continue
        max_pos = np.max(chr_df["pos"].values)
        pts_x = chr_df["pos"].values + running_pos
        pts_y = -1 * np.log10(chr_df["pval"].values)
        figs_list.append(
            ep.scattergl(
                pts_x,
                pts_y,
                name=chr_name,
                ylabel="-log10(p)",
                title=title,
                marker_size=3,
            )
        )
        cnt = cnt + len(chr_df)
        running_pos = running_pos + max_pos

    # withhold (return) or plot
    if withhold:
        return figs_list
    else:
        ep.plot_all(
            figs_list,
            panels=panels,
            numcols=1,
            showlegend=True,
            height=height,
            outfile=outfile,
        )


def chr_rolling_median(
    chr_pos_df: pd.DataFrame,
    chr_val_df: pd.DataFrame,
    rollwinsize: int,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    xlim: Optional[List[int]] = None,
    ylim: Optional[List[int]] = None,
    withhold: bool = False,
    outfile: Optional[str] = None,
) -> Optional[EZPlotlyPlot]:
    """
    Makes a chromosome rolling median plot.

    :param chr_pos_df: Chromosome positions in a dataframe as `pd.DataFrame`
    :param chr_val_df: Chromosome values in a dataframe as `pd.DataFrame`
    :param rollwinsize: The size of the rolling window as `int`
    :param ylabel: The y-axis label of the plot as `Optional[str]`
    :param title: The title of the plot as `Optional[str]`
    :param xlim: The x-axis limits [x_left_lim, x_right_lim] as `Optional[List[float]]`
    :param ylim: The y-axis limits [y_left_lim, y_right_lim] as `Optional[List[float]]`
    :param withhold: Whether to plot the plot or withhold and return the plot to the user as `bool`
    :param outfile: The file to write plot to as `Optional[str]`
    :return:
        `Optional[EZPlotlyPlot]` representing rolling median plot
    """
    # prepare plot
    x = chr_pos_df.values
    y = chr_val_df.rolling(rollwinsize).median()
    line_plot = ep.line(
        x=x,
        y=y,
        title=title,
        xlabel="Chr Position",
        ylabel=ylabel,
        xlim=xlim,
        ylim=ylim,
    )

    # withhold (return) or plot
    if withhold:
        return line_plot
    else:
        ep.plot_all([line_plot], outfile=outfile)


def chr_count(
    bool_vals: pd.DataFrame,
    chr_df: pd.DataFrame,
    title: Optional[str] = None,
    withhold: bool = False,
    outfile: Optional[str] = None,
) -> Optional[EZPlotlyPlot]:
    """

    Makes a chr bar plot, 1 bar for each chr specified tabulating a dataframe of booleans.

    :param bool_vals: Dataframe of booleans as `pd.DataFrame`
    :param chr_df: Chrs specified in a `pd.DataFrame`
    :param title: The title of the plot as `Optional[str]`
    :param withhold: Whether to plot the plot or withhold and return the plot to the user as `bool`
    :param outfile: The file to write plot to as `Optional[str]`
    :return:
        `Optional[EZPlotlyPlot]` representing chr bar plot.

    """
    # extract unique chr vals
    uniq_vals = chr_df.unique()

    # make chromosome-level counts
    counts = np.zeros((len(uniq_vals),))
    cnt = 0
    for u in uniq_vals:
        counts[cnt] = np.sum(bool_vals[chr_df == u])
        cnt = cnt + 1

    # plot
    bar_plot = ep.bar(
        counts, x=uniq_vals, title=title, xlabel="Chromosome", ylabel="Count"
    )
    if withhold:
        return bar_plot
    else:
        ep.plot_all([bar_plot], outfile=outfile)


def chr_distr(
    data: pd.DataFrame,
    chr_df: pd.DataFrame,
    distr_name: str,
    title: Optional[str] = None,
    withhold: bool = False,
    outfile: Optional[str] = None,
) -> Optional[EZPlotlyPlot]:
    """

    Makes a p-value plot testing whether data for each chromosome follows a particular distribution.
    Plots a bar, one for each chr, for the associated p-value of fit to the specified distribution.

    :param data: The data in a `pd.DataFrame`
    :param chr_df: Chrs specified in a `pd.DataFrame`
    :param distr_name: The distribution name as `str`
    :param title: The title of the plot as `Optional[str]`
    :param withhold: Whether to plot the plot or withhold and return the plot to the user as `bool`
    :param outfile: The file to write plot to as `Optional[str]`
    :return:
        `Optional[EZPlotlyPlot]` representing plot
    """

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
    bar_plot = ep.bar(
        pvals,
        x=uniq_vals,
        title=title,
        xlabel="Chromosome",
        ylabel="P-value (Null: Data is " + distr_name + ")",
        ylim=[0, 1],
    )
    if withhold:
        return bar_plot
    else:
        ep.plot_all([bar_plot], outfile=outfile)


def chr_hist(
    df: pd.DataFrame,
    data_col_name: str,
    min_bin: Optional[float] = None,
    max_bin: Optional[float] = None,
    bin_size: Optional[int] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    histnorm: Optional[str] = None,
    x_dtick: Optional[List[int]] = None,
    y_dtick: Optional[List[int]] = None,
    withhold: bool = False,
    outfile: Optional[str] = None,
) -> Optional[List[EZPlotlyPlot]]:
    """
    Makes 24 histograms of data, 1 for each chromosome. Data is passed in as a pandas dataframe
    with columns (chr, data_col_name).

    :param df: The data frame containing columns (chr, data_col_name) as `pd.dataFrame`
    :param data_col_name: The data column to histogram as `str`
    :param min_bin: The left bin edge of the histogram as `Optional[float]`
    :param max_bin: The right bin edge of the histogram as `Optional[float]`
    :param bin_size: The size of a histogram bin as `Optional[float]`
    :param title: The title of the plot as `Optional[str]`
    :param xlabel: The x-axis label as `Optional[str]`
    :param ylabel: The y-axis label as `Optional[str]`
    :param histnorm: The normalization scheme to use as `Optional[str]`
    :param x_dtick: The plotting delta tick (i.e. tick length) of the x-axis as `Optional[float]`
    :param y_dtick: The plotting delta tick (i.e. tick length) of the y-axis as `Optional[float]`
    :param withhold: Whether to plot the plot or withhold and return the plot to the user as `bool`
    :param outfile: If specified, save to this file as `Optional[str]`
    :return:
        `Optional[List[EZPlotlyPlot]]` representing figures list of histograms
    """

    # labels
    if xlabel is None:
        xlabel = data_col_name
    if ylabel is None:
        ylabel = "Count"

    # chrs
    chrs_list = list()
    for i in range(1, 23):
        chrs_list.append("chr" + str(i))
    chrs_list.append("chrX")
    chrs_list.append("chrY")

    # extract unique chr values
    uniq_vals = df["chr"].unique()

    # make chromosome level hists
    hists = list()
    for u_chr in chrs_list:
        if u_chr in uniq_vals:
            data = df[data_col_name][df["chr"] == u_chr]
            ep_hist = ep.hist(
                data,
                title=u_chr,
                xlabel=xlabel,
                min_bin=min_bin,
                max_bin=max_bin,
                bin_size=bin_size,
                ylabel=ylabel,
                color="#1ad1ff",
                histnorm=histnorm,
                x_dtick=x_dtick,
                y_dtick=y_dtick,
            )
            hists.append(ep_hist)

    # make plot (or withhold)
    if withhold:
        return hists
    else:
        ep.plot_all(hists, numcols=5, title=title, chrpacked=True, outfile=outfile)


def chr_qq(
    df: pd.DataFrame,
    data_col_name: str,
    distr: str = "norm",
    sparams: List = (),
    title: Optional[str] = None,
    withhold: bool = False,
    outfile: Optional[str] = None,
) -> Optional[List[EZPlotlyPlot]]:
    """
    Makes 24 qq plots for data, 1 for each chromosome. Specify a pandas dataframe with columns (chr, data_col_name).
    Specify what the hypothetical distribution is using distr and sparams.

    :param df: `pd.DataFrame` of data containing columns (chr, data_col_name)
    :param data_col_name: The data column name in df as `str`
    :param distr: The hypothetical null distribution as `str`
    :param sparams: Distributional params as `List`
    :param title: The title of the plot as `Optional[str]`
    :param withhold: Whether to plot the plot or withhold and return the plot to the user as `bool`
    :param outfile: If specified, save to this file as `str`
    :return:
        `Optional[List[EZPlotlyPlot]]` of plots representing each chromosome qq-plot
    """

    # chrs
    chrs_list = ["chr" + str(i) for i in range(1, 23)]
    chrs_list.append("chrX")
    chrs_list.append("chrY")

    # extract unique chr values
    uniq_vals = df["chr"].unique()

    # make chromosome level qq-plots
    plots = list()
    panels = list()
    panel_index = 1
    for u_chr in chrs_list:
        if u_chr in uniq_vals:
            data = df[data_col_name][df["chr"] == u_chr].values
            qq = qqplot(data, sparams=sparams, distr=distr, title=u_chr, withhold=True)
            plots.append(qq[0])
            plots.append(qq[1])
            panels.append(panel_index)
            panels.append(panel_index)
            panel_index = panel_index + 1

    # make plot (or withhold)
    if withhold:
        return plots
    else:
        ep.plot_all(
            plots,
            panels=panels,
            numcols=5,
            height=1000,
            title=title,
            chrpacked=True,
            outfile=outfile,
        )


def qqplot(
    data: Sequence[float],
    distr: str = "norm",
    sparams: List[Any] = (),
    title: Optional[str] = None,
    name: Optional[str] = None,
    marker_color: str = "blue",
    line_color: str = "red",
    withhold: bool = False,
    outfile: Optional[str] = None,
) -> Optional[Tuple[EZPlotlyPlot, EZPlotlyPlot]]:
    """

    Make qq-plot of data with respect to a hypothesis distribution.

    :param data: The data to use as `Sequence[float]`
    :param distr: The hypothetical null distribution as `str`
    :param sparams: Distributional params as `List[Any]`
    :param title: The title of the plot as `Optional[str]`
    :param name: The name of the series as `Optional[str]
    :param marker_color: The color of a marker as `Optional[str]`
    :param line_color: The color of the line as `str`
    :param withhold: Whether to plot the plot or withhold and return the plot to the user as `bool`
    :param outfile: If specified, save to this file as `Optional[str]`
    :return:
        `Optional[Tuple[EZPlotlyPlot, EZPlotlyPlot]]` of scatter plot and line plot representing qqplot
    """
    qq = sc.probplot(data, dist=distr, sparams=sparams)
    x = np.array([qq[0][0][0], qq[0][0][-1]])
    pts_scatter = ep.scattergl(
        x=qq[0][0],
        y=qq[0][1],
        title=title,
        xlabel="Expected",
        ylabel="Observed",
        marker_size=5,
        marker_color=marker_color,
        name=name,
    )
    if name is None:
        name = ""
    line_plot = ep.line(
        x=x,
        y=qq[1][1] + qq[1][0] * x,
        width=3,
        color=line_color,
        title=title,
        name=(name + " (distribution=" + distr + ")"),
    )

    # make plot (or withhold)
    if withhold:
        return pts_scatter, line_plot
    else:
        ep.plot_all(plots=[pts_scatter, line_plot], panels=[1, 1], outfile=outfile)


def roc(
    preds: np.array,
    gt: np.array,
    panel: int = 1,
    names: Optional[List[str]] = None,
    title: Optional[str] = None,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    withhold: bool = False,
    outfile: Optional[str] = None,
) -> Optional[Tuple[List[EZPlotlyPlot], List[int]]]:
    """
    Make Receiver Operating Characteristic (ROC) Curve.

    :param preds: The raw prediction values as an N x L matrix np.array[float] where N is the number of methods and L
        is the number of labels.
    :param gt: The ground truth labels as an (L,) np.array[int]
    :param panel: The panel to plot the ROC curve as `int`
    :param names: The names of the methods as `Optional[List[str]]`
    :param title: The title of the plot as `Optional[str]`
    :param xscale: The scale of the x-axis ('log', 'linear') as `Optional[str]`
    :param yscale: The scale of the y-axis ('log', 'linear') as `Optional[str]`
    :param withhold: Whether to plot the plot or withhold and return the plot to the user as `bool`
    :param outfile: If specified, save to this file as `Optional[str]`
    :return:
        plots: List[EZPlotlyPlot] of generated plots
        panels: List[int] of panels for plots
    """

    # structure
    plots = list()
    panels = list()

    # add chance curve
    p = ep.line(
        x=np.arange(0.0, 1.01, 0.01),
        y=np.arange(0.0, 1.01, 0.01),
        width=2,
        name="Chance Curve",
        color="black",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=title,
        xlim=[0, 1.01],
        ylim=[0, 1.01],
        xscale=xscale,
        yscale=yscale,
        x_dtick=0.1,
        y_dtick=0.1,
    )
    plots.append(p)
    panels.append(panel)

    # for each predictor compute roc curve
    for i in range(0, preds.shape[0]):
        fpr, tpr = yr.roc(preds[i, :], gt)
        auc = np.round(yr.auc(fpr, tpr), 3)
        name = "AUC=" + str(auc)
        if names is not None:
            name = names[i] + "(" + "AUC=" + str(auc) + ")"
        p = ep.line(
            x=fpr,
            y=tpr,
            width=2,
            name=name,
            xlim=[0, 1.0],
            ylim=[0, 1.01],
            xscale=xscale,
            yscale=yscale,
            x_dtick=0.1,
            y_dtick=0.1,
        )
        plots.append(p)
        panels.append(panel)

    # make plot (or withhold)
    if withhold:
        return plots, panels
    else:
        ep.plot_all(plots=plots, panels=panels, outfile=outfile, showlegend=True)


def ecdf(
    data: np.array,
    min_bin: Optional[float] = None,
    max_bin: Optional[float] = None,
    bin_size: Optional[float] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    norm: bool = False,
    name: Optional[str] = None,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    x_dtick: Optional[List[int]] = None,
    y_dtick: Optional[List[int]] = None,
    color: Optional[str] = None,
    withhold: bool = False,
    outfile: Optional[str] = None,
) -> Optional[EZPlotlyPlot]:
    """
    Plot empirical cumulative distribution function (CDF) of data. Function allows specifying of histogram binning of
    data for the probability distribution function (PDF) which is integrated to estimate the CDF.

    :param data: The data to plot the empirical CDF of as `np.array[float]`
    :param min_bin: The left bin edge of the estimating histogram as `Optional[float]`
    :param max_bin: The right bin edge of the estimating histogram as `Optional[float]`
    :param bin_size: The bin size of a bin in the estimating histogram as `Optional[float]`
    :param title: The title of the plot as `Optional[str]`
    :param xlabel: The x-axis label as `Optional[str]`
    :param ylabel: The y-axis label as `Optional[str]`
    :param norm: Whether to normalize the CDF or not as `bool`
    :param name: The name of the series as `Optional[str]`
    :param xscale: The scale of the x-axis ('log', 'linear') as `Optional[str]`
    :param yscale: The scale of the y-axis ('log', 'linear') as `Optional[str]`
    :param x_dtick: The plotting delta tick (i.e. tick length) of the x-axis as `Optional[float]`
    :param y_dtick: The plotting delta tick (i.e. tick length) of the y-axis as `Optional[float]`
    :param color: The color of the line as `Optional[str]`
    :param withhold: Whether to plot the plot or withhold and return the plot to the user as `bool`
    :param outfile: If specified, save to this file as `Optional[str]`
    :return:
        `Optional[EZPlotlyPlot]` of ecdf
    """

    # default binning
    if min_bin is None:
        min_bin = np.nanmin(data)
    if max_bin is None:
        max_bin = np.nanmax(data)
    if bin_size is None:
        bin_size = 1

    # histogram data and produce cdf
    counts, bin_edges = np.histogram(
        data, bins=np.arange(min_bin, max_bin + bin_size, bin_size)
    )
    counts_sum = np.sum(counts)
    counts = counts / counts_sum
    cdf = np.cumsum(counts)

    # plot
    if not norm:
        if ylabel is None:
            ylabel = "Cum Freq"
        cdf_line = ep.line(
            x=bin_edges[:-1],
            y=counts_sum * cdf,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=[min_bin, max_bin + bin_size],
            name=name,
            xscale=xscale,
            yscale=yscale,
            x_dtick=x_dtick,
            y_dtick=y_dtick,
            color=color,
        )
    else:
        if ylabel is None:
            ylabel = "CDF"
        cdf_line = ep.line(
            x=bin_edges[:-1],
            y=cdf,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=[min_bin, max_bin + bin_size],
            ylim=[0, 1.0],
            name=name,
            xscale=xscale,
            yscale=yscale,
            x_dtick=x_dtick,
            y_dtick=y_dtick,
            color=color,
        )

    # make plot (or withhold)
    if withhold:
        return cdf_line
    else:
        ep.plot_all(plots=[cdf_line], outfile=outfile)


def rcdf(
    data: np.array,
    min_bin: Optional[float] = None,
    max_bin: Optional[float] = None,
    bin_size: Optional[float] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    norm: bool = False,
    name: Optional[str] = None,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    x_dtick: Optional[List[int]] = None,
    y_dtick: Optional[List[int]] = None,
    color: Optional[str] = None,
    withhold: bool = False,
    outfile: Optional[str] = None,
) -> Optional[EZPlotlyPlot]:
    """
    Plot of reverse cumulative frequency distribution.

    :param data: The data to plot as `np.array`
    :param min_bin: The left bin edge of the estimating histogram as `Optional[float]`
    :param max_bin: The right bin edge of the estimating histogram as `Optional[float]`
    :param bin_size: The bin size of a bin in the estimating histogram as `Optional[float]`
    :param title: The title of the plot as `Optional[str]`
    :param xlabel: The x-axis label as `Optional[str]`
    :param ylabel: The y-axis label as `Optional[str]`
    :param norm: Whether to normalize the CDF or not as `bool`
    :param name: The name of the series as `Optional[str]`
    :param xscale: The scale of the x-axis ('log', 'linear') as `Optional[str]`
    :param yscale: The scale of the y-axis ('log', 'linear') as `Optional[str]`
    :param x_dtick: The plotting delta tick (i.e. tick length) of the x-axis as `Optional[float]`
    :param y_dtick: The plotting delta tick (i.e. tick length) of the y-axis as `Optional[float]`
    :param color: The color of the line as `Optional[str]`
    :param withhold: Whether to plot the plot or withhold and return the plot to the user as `bool`
    :param outfile: If specified, save to this file as `Optional[str]`
    :return:
        `Optional[EZPlotlyPlot]` represent rcdf
    """

    # default binning
    if min_bin is None:
        min_bin = np.nanmin(data)
    if max_bin is None:
        max_bin = np.nanmax(data)
    if bin_size is None:
        bin_size = 1

    # histogram data and produce cdf
    counts, bin_edges = np.histogram(
        data, bins=np.arange(min_bin, max_bin + bin_size, bin_size)
    )
    counts_sum = np.sum(counts)
    counts = counts / counts_sum
    cdf = np.cumsum(counts)

    # plot
    if not norm:
        if ylabel is None:
            ylabel = "Reverse Cum Freq"
        cdf_line = ep.line(
            x=bin_edges[:-1],
            y=np.round(counts_sum * (1.0 - cdf), 5),
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=[min_bin, max_bin + bin_size],
            name=name,
            xscale=xscale,
            yscale=yscale,
            x_dtick=x_dtick,
            y_dtick=y_dtick,
            color=color,
        )
    else:
        if ylabel is None:
            ylabel = "CDF"
        cdf_line = ep.line(
            x=bin_edges[:-1],
            y=np.round(1.0 - cdf, 5),
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=[min_bin, max_bin + bin_size],
            ylim=[0, 1.0],
            name=name,
            xscale=xscale,
            yscale=yscale,
            x_dtick=x_dtick,
            y_dtick=y_dtick,
            color=color,
        )

    # make plot (or withhold)
    if withhold:
        return cdf_line
    else:
        ep.plot_all(plots=[cdf_line], outfile=outfile)


def corr_plot(
    x: np.array,
    y: np.array,
    error_y: Optional[Sequence[float]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    name: Optional[str] = None,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    x_dtick: Optional[float] = None,
    y_dtick: Optional[float] = None,
    xlim: Optional[List[float]] = None,
    ylim: Optional[List[float]] = None,
    plot_type: str = "scatter",
    withhold: bool = False,
    outfile: Optional[str] = None,
) -> Optional[EZPlotlyPlot]:
    """
    Make scatterplot plot of data of two data vectors x and y. Displays correlation of data vectors in legend.

    :param x: First data vector as 1-D `np.array`
    :param y: Second data vector as 1-D `np.array`
    :param error_y: Error bar length as Sequence[float]`
    :param xlabel: X-axis label as `Optional[str]`
    :param ylabel: Y-axis label as `Optional[str]`
    :param title: Plot title as `Optional[str]`
    :param name: The name of the scatter plot as `Optional[str]` (useful for plotting series)
    :param xscale: The scale of the x-axis ('log', 'linear') as `Optional[str]`
    :param yscale: The scale of the y-axis ('log', 'linear') as `Optional[str]`
    :param x_dtick: The plotting delta tick (i.e. tick length) of the x-axis as `Optional[float]`
    :param y_dtick: The plotting delta tick (i.e. tick length) of the y-axis as `Optional[float]`
    :param xlim: The x-axis limits [x_left_lim, x_right_lim] as `Optional[List[float]]`
    :param ylim: The y-axis limits [y_left_lim, y_right_lim] as `Optional[List[float]]`
    :param plot_type: The plot type as `str` (either scatter or line)
    :param withhold: Whether to plot the plot or withhold and return the plot to the user as `bool`
    :param outfile: If specified, save to this file as `Optional[str]`
    :return:
        `Optional[EZPlotlyPlot]` representing correlation plot
    """

    # compute correlation value
    corr_val = pd.core.nanops.nancorr(x, y)

    # get name of plot
    if name is None:
        name = "corr=" + str(np.round(corr_val, 3))
    else:
        name = name + " (" + "corr=" + str(np.round(corr_val, 3)) + ")"

    # make EZPlotlyPlot
    if plot_type == "scatter":
        plot = ep.scattergl(
            x=x,
            y=y,
            error_y=error_y,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            name=name,
            xscale=xscale,
            yscale=yscale,
            x_dtick=x_dtick,
            y_dtick=y_dtick,
            xlim=xlim,
            ylim=ylim,
        )
    elif plot_type == "line":
        plot = ep.line(
            x=x,
            y=y,
            error_y=error_y,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            name=name,
            xscale=xscale,
            yscale=yscale,
            x_dtick=x_dtick,
            y_dtick=y_dtick,
            xlim=xlim,
            ylim=ylim,
        )
    else:
        raise ValueError("plot_type must be {scatter, line}.")

    # make plot (or withhold)
    if withhold:
        return plot
    else:
        ep.plot_all(plots=[plot], outfile=outfile, showlegend=True)


def nonparametric_ci(
    x: np.array,
    y_data: np.array,
    conf: float = 0.95,
    ci_plot_type: str = "line",
    color: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    name: Optional[str] = None,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    x_dtick: Optional[float] = None,
    y_dtick: Optional[float] = None,
    xlim: Optional[List[float]] = None,
    ylim: Optional[List[float]] = None,
    withhold: bool = False,
    outfile: Optional[str] = None,
) -> Union[List[EZPlotlyPlot], EZPlotlyPlot]:
    """
    Plots a data plot with non-parametric (i.e. N%) confidence intervals.

    :param x: x data series as `np.array[float]`
    :param y_data: y data series as `np.array[float]`
    :param conf: The % confidence interval as `float`
    :param ci_plot_type: The confidence interval plot type as `str` (either line or point)
    :param color: The line or marker color as `str`
    :param xlabel: X-axis label as `Optional[str]`
    :param ylabel: Y-axis label as `Optional[str]`
    :param title: Plot title as `Optional[str]`
    :param name: The name of the scatter plot as `Optional[str]` (useful for plotting series)
    :param xscale: The scale of the x-axis ('log', 'linear') as `Optional[str]`
    :param yscale: The scale of the y-axis ('log', 'linear') as `Optional[str]`
    :param x_dtick: The plotting delta tick (i.e. tick length) of the x-axis as `Optional[float]`
    :param y_dtick: The plotting delta tick (i.e. tick length) of the y-axis as `Optional[float]`
    :param xlim: The x-axis limits [x_left_lim, x_right_lim] as `Optional[List[float]]`
    :param ylim: The y-axis limits [y_left_lim, y_right_lim] as `Optional[List[float]]`
    :param withhold: Whether to plot the plot or withhold and return the plot to the user as `bool`
    :param outfile: If specified, save to this file as `Optional[str]`
    :return:
        `Union[List[EZPlotlyPlot], EZPlotlyPlot]` representing plots
    """
    # compute confidence intervals
    m, ll, ul = yr.nonparametric_ci(data=y_data, conf=conf)

    # generate plots
    if ci_plot_type == "line":

        # get name
        if name is None:
            ci_name = " (CI=" + str(conf * 100.0) + "%" + ")"
        else:
            ci_name = name + " (CI=" + str(conf * 100.0) + "%" + ")"

        # make line EZPlotlyPlots
        ll_pl = ep.line(x=x, y=ll, color=color, dash="dot", name=ci_name)
        ul_pl = ep.line(x=x, y=ul, color=color, dash="dot", name=ci_name)
        mean_pl = ep.line(
            x=x,
            y=m,
            color=color,
            name=name,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            xscale=xscale,
            yscale=yscale,
            x_dtick=x_dtick,
            y_dtick=y_dtick,
            xlim=xlim,
            ylim=ylim,
        )
        figs_list = [mean_pl, ll_pl, ul_pl]

        # make plot (or withhold)
        if withhold:
            return figs_list
        else:
            ep.plot_all(figs_list, panels=[1, 1, 1], outfile=outfile)
    elif ci_plot_type == "point":

        # gen EZPlotlyPlot
        plot = ep.line(
            x=x,
            y=m,
            ucl=np.abs(ul - m),
            lcl=np.abs(ll - m),
            color=color,
            name=name,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            xscale=xscale,
            yscale=yscale,
            x_dtick=x_dtick,
            y_dtick=y_dtick,
            xlim=xlim,
            ylim=ylim,
        )

        # make plot (or withhold)
        if withhold:
            return plot
        else:
            ep.plot_all([plot], outfile=outfile)
    else:
        raise ValueError("ci_plot_type must be line or point.")
