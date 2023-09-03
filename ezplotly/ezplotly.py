from typing import Sequence, List, Optional, Any, Dict, NamedTuple, Union

import numpy as np
import plotly
import plotly.graph_objs as go
import ezplotly.settings as plot_settings
from plotly.subplots import make_subplots


class EZPlotlyPlot(NamedTuple):
    plot_type: str
    title: Optional[str]
    xlabel: Optional[str]
    ylabel: Optional[str]
    plot_obj: Any
    xlim: Optional[List[float]]
    ylim: Optional[List[float]]
    xscale: Optional[str]
    yscale: Optional[str]
    x_dtick: Optional[float]
    y_dtick: Optional[float]


def _prepare_error_bars(
    error_y: Optional[Union[Sequence[float], np.array]], y: np.array
) -> Optional[Dict[str, Any]]:
    """
    Prepares error bar data for plotting.
    :param error_y: The error bar values as `Optional[Sequence[float], Optional[np.array]]`
    :return:
        A dictionary containing the error bar data as `Dict[str, Any]`
    """
    if error_y is not None:
        if isinstance(error_y, np.ndarray):
            if not error_y.shape == (2, len(y)):
                raise ValueError(
                    "Error bar array must have shape (2, len(y)) if specified as numpy array."
                )
            lower_error = error_y[0, :]
            higher_error = error_y[1, :]
            error_y_dict = dict(
                type="data", array=higher_error, arrayminus=lower_error, visible=True
            )
        elif isinstance(error_y, list):
            error_y_dict = dict(type="data", array=error_y, visible=True)
        else:
            raise TypeError(
                "Error bar values must be a numpy array or a list of floats."
            )
    else:
        error_y_dict = dict(type="data", array=None, visible=False)
    return error_y_dict


def hist(
    data: Sequence[Any],
    min_bin: Optional[float] = None,
    max_bin: Optional[float] = None,
    bin_size: Optional[float] = None,
    histnorm: str = "",
    title: Optional[str] = None,
    name: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    color: Optional[str] = None,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    x_dtick: Optional[float] = None,
    y_dtick: Optional[float] = None,
    xlim: Optional[List[float]] = None,
    ylim: Optional[List[float]] = None,
) -> EZPlotlyPlot:
    """
    Makes a 1-D histogram of data.

    :param data: The data to plot as `Sequence[Any]`
    :param min_bin: The left bin edge of the histogram as `Optional[float]`
    :param max_bin: The right bin edge of the histogram as `Optional[float]`
    :param bin_size: The size of a histogram bin as `Optional[float]`
    :param histnorm: The normalization scheme to use as `str`
    :param title: Plot title as `Optional[str]`
    :param name: The name of the histogram as `str` (useful for plotting a series of histograms)
    :param xlabel: The x-axis label as `Optional[str]`
    :param ylabel: The y-axis label as `Optional[str]`
    :param color: The color of the histogram as `Optional[str]`
    :param xscale: The scale of the x-axis ('log', 'linear') as `Optional[str]`
    :param yscale: The scale of the y-axis ('log', 'linear') as `Optional[str]`
    :param x_dtick: The plotting delta tick (i.e. tick length) of the x-axis as `Optional[float]`
    :param y_dtick: The plotting delta tick (i.e. tick length) of the y-axis as `Optional[float]`
    :param xlim: The x-axis limits [x_left_lim, x_right_lim] as `Optional[List[float]]`
    :param ylim: The y-axis limits [y_left_lim, y_right_lim] as `Optional[List[float]]`
    :return:
        `EZPlotlyPlot` object representing histogram
    """

    # plot type
    plot_type = "histogram"

    # y-label auto-setting
    if ylabel is None:
        if histnorm.lower() == "probability":
            ylabel = "Probability"
        elif histnorm == "":
            ylabel = "Frequency"

    # xbin properties
    if min_bin is None or max_bin is None or bin_size is None:
        xbins = None
    else:
        if xlim is None:
            xlim = [min_bin, max_bin]
        xbins = dict(start=min_bin, end=max_bin, size=bin_size)

    # assemble marker properties
    marker = dict()
    if color is not None:
        marker["color"] = color

    # legend properties
    showlegend = name is not None

    # assemble hist object
    hist_obj = go.Histogram(
        x=data,
        name=name,
        xbins=xbins,
        marker=marker,
        histnorm=histnorm,
        showlegend=showlegend,
    )

    # return plot
    return EZPlotlyPlot(
        plot_type=plot_type,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        plot_obj=hist_obj,
        xlim=xlim,
        ylim=ylim,
        xscale=xscale,
        yscale=yscale,
        x_dtick=x_dtick,
        y_dtick=y_dtick,
    )


def bar(
    y: Sequence[Any],
    x: Sequence[Any] = (),
    error_y: Optional[Union[Sequence[float], np.array]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlim: Optional[List[float]] = None,
    ylim: Optional[List[float]] = None,
    name: Optional[str] = None,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    x_dtick: Optional[float] = None,
    y_dtick: Optional[float] = None,
    text: Optional[Union[Sequence[str], str]] = None,
    textsize: Optional[Union[Sequence[int], int]] = None,
    textposition: Optional[Union[Sequence[str], str]] = None,
    hovertext: Optional[Union[Sequence[str], str]] = None,
    color: Optional[str] = None,
) -> EZPlotlyPlot:
    """
    Makes a 1-D bar plot.

    :param y: The data for x-axis values as `Sequence[Any]`
    :param x: The x-axis bar labels as `Sequence[Any]`
    :param error_y: The error on the y-axis values as `Optional[Union[Sequence[float], np.ndarray]]`
        If a sequence, must be the same length as `y`
        If `error_y` is a numpy array of shape (2, N), where N is the length of `y`,
        where the first row is the lower error and the second row is the upper error.
    :param title: Plot title as `Optional[str]`
    :param xlabel: X-axis label as `Optional[str]`
    :param ylabel: Y-axis label as `Optional[str]`
    :param xlim: The x-axis limits [x_left_lim, x_right_lim] as `Optional[List[float]]`
    :param ylim: The y-axis limits [y_left_lim, y_right_lim] as `Optional[List[float]]`
    :param name: The name of the bar plot as `Optional[str]` (useful if plotting a series of bar plots)
    :param xscale: The scale of the x-axis ('log', 'linear') as `Optional[str]`
    :param yscale: The scale of the y-axis ('log', 'linear') as `Optional[str]`
    :param x_dtick: The plotting delta tick (i.e. tick length) of the x-axis as `Optional[float]`
    :param y_dtick: The plotting delta tick (i.e. tick length) of the y-axis as `Optional[float]`
    :param text: Bar text for each bar as `Optional[Union[Sequence[str], str]]`
    :param textsize: Text size for each bar as `Optional[Union[Sequence[int], int]]`
    :param textposition:  Bar text position as `Optional[Union[Sequence[str], str]]`
    :param hovertext: Hover text associated with bars as `Optional[Union[Sequence[str], str]]`
    :param color: The color of the bar series as `Optional[str]`
    :return:
        `EZPlotlyPlot` object represent bar chart
    """

    # plot type
    plot_type = "bar"

    # define x range
    if len(x) == 0:
        x = [a for a in range(0, len(y))]

    # legend properties
    showlegend = name is not None

    # assemble bar object
    bar_obj = go.Bar(
        name=name,
        x=x,
        y=y,
        error_y=_prepare_error_bars(error_y=error_y, y=y),
        text=text,
        textfont=dict(size=textsize),
        textposition=textposition,
        hovertext=hovertext,
        marker_color=color,
        showlegend=showlegend,
    )

    # return
    return EZPlotlyPlot(
        plot_type=plot_type,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        plot_obj=bar_obj,
        xlim=xlim,
        ylim=ylim,
        xscale=xscale,
        yscale=yscale,
        x_dtick=x_dtick,
        y_dtick=y_dtick,
    )


def bar_hist(
    data: Sequence[Any],
    min_bin: Optional[float] = None,
    max_bin: Optional[float] = None,
    bin_size: Optional[float] = None,
    histnorm: str = "",
    title: Optional[str] = None,
    name: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    color: Optional[str] = None,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    x_dtick: Optional[float] = None,
    y_dtick: Optional[float] = None,
    xlim: Optional[List[float]] = None,
    ylim: Optional[List[float]] = None,
    show_text: bool = False,
    textsize: Optional[Union[Sequence[int], int]] = None,
    textposition: Optional[Union[Sequence[str], str]] = None,
    hovertext: Optional[Union[Sequence[str], str]] = None,
) -> EZPlotlyPlot:
    """

    Makes histogram using numpy and then plot using bar chart. Faster for larger
    data than hist function.

    :param data: The data to plot as `Sequence[float]`
    :param min_bin: The left bin edge of the histogram as `Optional[float]`
    :param max_bin: The right bin edge of the histogram as `Optional[float]`
    :param bin_size: The size of a histogram bin as `Optional[float]`
    :param histnorm: The normalization scheme to use as `Optional[str]`
    :param title: Plot title as `Optional[str]`
    :param name: The name of the histogram as `str` (useful for plotting a series of histograms)
    :param xlabel: The x-axis label as `Optional[str]`
    :param ylabel: The y-axis label as `Optional[str]`
    :param color: The color of the histogram as `Optional[str]`
    :param xscale: The scale of the x-axis ('log', 'linear') as `Optional[str]`
    :param yscale: The scale of the y-axis ('log', 'linear') as `Optional[str]`
    :param x_dtick: The plotting delta tick (i.e. tick length) of the x-axis as `Optional[float]`
    :param y_dtick: The plotting delta tick (i.e. tick length) of the y-axis as `Optional[float]`
    :param xlim: The x-axis limits [x_left_lim, x_right_lim] as `Optional[List[float]]`
    :param ylim: The y-axis limits [y_left_lim, y_right_lim] as `Optional[List[float]]`
    :param show_text: Whether to show test labels for each bar
    :param textsize: Text size for each bar as `Optional[Union[Sequence[int], int]]]`
    :param textposition:  Bar text position as `Optional[Union[Sequence[str], str]]`
    :param hovertext: Hover text associated with bars as `Optional[Union[Sequence[str], str]]`
    :return:
        `EZPlotlyPlot` object representing histogram
    """
    # histogram using numpy
    bin_range = np.arange(min_bin, max_bin + bin_size, bin_size)
    hist_vals, bin_edges = np.histogram(data, bin_range)

    # normalize hist_vals if needed
    if histnorm == "probability":
        hist_vals = hist_vals / np.sum(hist_vals)

    # xlim auto setting
    if xlim is None:
        xlim = [min_bin, max_bin]

    # y-label amd y-lim auto-setting
    if ylabel is None:
        if histnorm == "probability":
            ylabel = "Probability"
            if ylim is None:
                ylim = [0.0, 1.0]
        elif histnorm == "":
            ylabel = "Frequency"

    # text labels
    if show_text:
        text: Optional[List[str]] = [str(hist_val) for hist_val in hist_vals]
    else:
        text: Optional[List[str]] = None

    # plot using bar plot
    return bar(
        y=hist_vals,
        x=bin_range + bin_size / 2.0,
        title=title,
        name=name,
        xlabel=xlabel,
        ylabel=ylabel,
        color=color,
        xscale=xscale,
        yscale=yscale,
        x_dtick=x_dtick,
        y_dtick=y_dtick,
        xlim=xlim,
        ylim=ylim,
        text=text,
        textsize=textsize,
        textposition=textposition,
        hovertext=hovertext,
    )


def scatter(
    x: Sequence[Any],
    y: Sequence[Any],
    error_y: Optional[Union[Sequence[float], np.array]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlim: Optional[List[float]] = None,
    ylim: Optional[List[float]] = None,
    marker_type: str = "circle",
    marker_size: Optional[float] = None,
    marker_color: Optional[str] = None,
    name: Optional[str] = None,
    point_anno: Optional[Union[List[str], str]] = None,
    textsize: Optional[Union[List[int], int]] = None,
    textposition: Optional[Union[List[str], str]] = None,
    hovertext: Optional[Union[List[str], str]] = None,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    x_dtick: Optional[float] = None,
    y_dtick: Optional[float] = None,
) -> EZPlotlyPlot:
    """
    Makes a scatter plot.

    :param x: The x-data to plot as `Sequence[Any]`
    :param y: The y-data to plot as `Sequence[Any]`
    :param error_y: The y-error data to plot as `Optional[Union[Sequence[float], np.ndarray]]`
        If `error_y` is a sequence, it must be the same length as `y`.
        If `error_y` is a numpy array of shape (2, N), where N is the length of `y`,
         where the first row is the lower error and the second row is the upper error.
    :param title: Plot title as `Optional[str]`
    :param xlabel: X-axis label as `Optional[str]`
    :param ylabel: Y-axis label as `Optional[str]`
    :param xlim: The x-axis limits [x_left_lim, x_right_lim] as `Optional[List[float]]`
    :param ylim: The y-axis limits [y_left_lim, y_right_lim] as `Optional[List[float]]`
    :param marker_type The marker symbol type as `str`
    :param marker_size: The size of a marker as `Optional[float]`
    :param marker_color: The color of a marker as `Optional[str]`
    :param name: The name of the scatter plot as `Optional[str]` (useful for plotting series)
    :param point_anno: Text annotations for each point as `Optional[Union[Sequence[str], str]]`
    :param textsize: Text size for each bar as `Optional[Union[Sequence[int], int]]`
    :param textposition:  Bar text position as `Optional[Union[Sequence[str], str]]`
    :param hovertext: Hover text associated with bars as `Optional[Union[Sequence[str], str]]`
    :param xscale: The scale of the x-axis ('log', 'linear') as `Optional[str]`
    :param yscale: The scale of the y-axis ('log', 'linear') as `Optional[str]`
    :param x_dtick: The plotting delta tick (i.e. tick length) of the x-axis as `Optional[float]`
    :param y_dtick: The plotting delta tick (i.e. tick length) of the y-axis as `Optional[float]`
    :return:
        `EZPlotlyPlot` representing scatter plot
    """

    # plot type
    plot_type = "scatter"

    # assemble marker information
    marker = dict()
    marker["symbol"] = marker_type
    if marker_size is not None:
        marker["size"] = marker_size
    if marker_color is not None:
        marker["color"] = marker_color

    # annotation mode
    if point_anno is not None:
        mode = "markers+text"
    else:
        mode = "markers"

    # legend properties
    showlegend = name is not None

    # make scatter gl object
    scatter_obj = go.Scatter(
        name=name,
        x=x,
        y=y,
        error_y=_prepare_error_bars(error_y=error_y, y=y),
        mode=mode,
        visible=True,
        marker=marker,
        text=point_anno,
        textfont=dict(size=textsize),
        textposition=textposition,
        hovertext=hovertext,
        showlegend=showlegend,
    )

    # return
    return EZPlotlyPlot(
        plot_type=plot_type,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        plot_obj=scatter_obj,
        xlim=xlim,
        ylim=ylim,
        xscale=xscale,
        yscale=yscale,
        x_dtick=x_dtick,
        y_dtick=y_dtick,
    )


def scattergl(
    x: Sequence[Any],
    y: Sequence[Any],
    error_y: Optional[Union[Sequence[float], np.array]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlim: Optional[List[float]] = None,
    ylim: Optional[List[float]] = None,
    marker_type: str = "circle",
    marker_size: Optional[float] = None,
    marker_color: Optional[str] = None,
    name: Optional[str] = None,
    point_anno: Optional[Union[Sequence[str], str]] = None,
    textsize: Optional[Union[Sequence[int], int]] = None,
    textposition: Optional[Union[Sequence[str], str]] = None,
    hovertext: Optional[Union[Sequence[str], str]] = None,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    x_dtick: Optional[float] = None,
    y_dtick: Optional[float] = None,
) -> EZPlotlyPlot:
    """
    Makes a scatter plot using open-gl backend.

    :param x: The x-data to plot as `Sequence[Any]`
    :param y: The y-data to plot as `Sequence[Any]`
    :param error_y: The y-error data to plot as `Optional[Union[Sequence[float], np.array]]`
        If `error_y` is a sequence, it must be the same length as `y`.
        If `error_y` is a numpy array of shape (2, N), where N is the length of `y`,
         where the first row is the lower error and the second row is the upper error.
    :param title: Plot title as `Optional[str]`
    :param xlabel: X-axis label as `Optional[str]`
    :param ylabel: Y-axis label as `Optional[str]`
    :param xlim: The x-axis limits [x_left_lim, x_right_lim] as `Optional[List[float]]`
    :param ylim: The y-axis limits [y_left_lim, y_right_lim] as `Optional[List[float]]`
    :param marker_type The marker symbol type as `str`
    :param marker_size: The size of a marker as `Optional[float]`
    :param marker_color: The color of a marker as `Optional[str]`
    :param name: The name of the scatter plot as `Optional[str]` (useful for plotting series)
    :param point_anno: Text annotations for each point as `Optional[Union[Sequence[str], str]]`
    :param textsize: Text size for each bar as `Optional[Union[Sequence[int], int]]]`
    :param textposition: The text position as `Optional[Union[Sequence[str], str]]`
    :param hovertext: The hover text associated with each point as `Optional[Union[Sequence[str], str]]`
    :param xscale: The scale of the x-axis ('log', 'linear') as `Optional[str]`
    :param yscale: The scale of the y-axis ('log', 'linear') as `Optional[str]`
    :param x_dtick: The plotting delta tick (i.e. tick length) of the x-axis as `Optional[float]`
    :param y_dtick: The plotting delta tick (i.e. tick length) of the y-axis as `Optional[float]`
    :return:
        `EZPlotlyPlot` object representing scatter plot
    """

    # plot type
    plot_type = "scattergl"

    # assemble marker information
    marker = dict()
    marker["symbol"] = marker_type
    if marker_size is not None:
        marker["size"] = marker_size
    if marker_color is not None:
        marker["color"] = marker_color

    # annotation mode
    if point_anno is not None:
        mode = "markers+text"
    else:
        mode = "markers"

    # legend properties
    showlegend = name is not None

    # make scatter gl object
    scatter_obj = go.Scattergl(
        name=name,
        x=x,
        y=y,
        error_y=_prepare_error_bars(error_y=error_y, y=y),
        mode=mode,
        visible=True,
        marker=marker,
        text=point_anno,
        textfont=dict(size=textsize),
        textposition=textposition,
        hovertext=hovertext,
        showlegend=showlegend,
    )

    # return
    return EZPlotlyPlot(
        plot_type=plot_type,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        plot_obj=scatter_obj,
        xlim=xlim,
        ylim=ylim,
        xscale=xscale,
        yscale=yscale,
        x_dtick=x_dtick,
        y_dtick=y_dtick,
    )


def line(
    x: Sequence[Any],
    y: Sequence[Any],
    error_y: Optional[Union[Sequence[float], np.array]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlim: Optional[List[float]] = None,
    ylim: Optional[List[float]] = None,
    width: Optional[float] = None,
    color: Optional[str] = None,
    name: Optional[str] = None,
    point_anno: Optional[Union[Sequence[str], str]] = None,
    textsize: Optional[Union[Sequence[int], int]] = None,
    textposition: Optional[Union[Sequence[str], str]] = None,
    hovertext: Optional[Union[Sequence[str], str]] = None,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    x_dtick: Optional[float] = None,
    y_dtick: Optional[float] = None,
    dash: Optional[str] = None,
) -> EZPlotlyPlot:
    """
    Make line plot. Allows addition of confidence intervals specifying either uniform (via error_y)
    or non-uniform error bars (using lcl, ucl parameters).

    :param x: x-data as `Sequence[Any]`
    :param y: y-data as `Sequence[Any]`
    :param error_y: The y-error data to plot as `Optional[Union[Sequence[float], np.ndarray]]`
        If `error_y` is a sequence, it must be the same length as `y`.
        If `error_y` is a numpy array of shape (2, N), where N is the length of `y`,
         where the first row is the lower error and the second row is the upper error.
    :param title: Plot title as `Optional[str]`
    :param xlabel: The x-axis label as `Optional[str]`
    :param ylabel: The y-axis label as `Optional[str]`
    :param xlim: The x-axis limits [x_left_lim, x_right_lim] as `Optional[List[float]]`
    :param ylim: The y-axis limits [y_left_lim, y_right_lim] as `Optional[List[float]]`
    :param width: The width of the line as 'Optional[float]`
    :param color: The color of the line as `Optional[str]`
    :param name: The name of the line plot as `Optional[str]` (useful for plotting as series)
    :param point_anno: Text annotations for each point as `Optional[Union[Sequence[str], str]]`
    :param textsize: Text size for each bar as `Optional[Union[Sequence[int], int]]`
    :param textposition: The text position as `Optional[Union[Sequence[str], str]]`
    :param hovertext: The hover text associated with each point as `Optional[Union[Sequence[str], str]]`
    :param xscale: The scale of the x-axis ('log', 'linear') as `Optional[str]`
    :param yscale: The scale of the y-axis ('log', 'linear') as `Optional[str]`
    :param x_dtick: The plotting delta tick (i.e. tick length) of the x-axis as `Optional[float]`
    :param y_dtick: The plotting delta tick (i.e. tick length) of the y-axis as `Optional[float]`
    :param dash: The dash type as `Optional[str]`
    :return:
        `EZPlotlyPlot` object representing line plot
    """

    # plot type
    plot_type = "line"

    # assemble line information
    line_dict = dict()
    if width is not None:
        line_dict["width"] = width
    if color is not None:
        line_dict["color"] = color
    if dash is not None:
        line_dict["dash"] = dash

    # annotation mode
    if point_anno is not None:
        mode = "lines+text"
    else:
        mode = "lines"

    # legend properties
    showlegend = name is not None

    # make scatter object
    scatter_obj = go.Scattergl(
        name=name,
        x=x,
        y=y,
        error_y=_prepare_error_bars(error_y=error_y, y=y),
        line=line_dict,
        mode=mode,
        text=point_anno,
        textfont=dict(size=textsize),
        textposition=textposition,
        hovertext=hovertext,
        showlegend=showlegend,
    )

    # return
    return EZPlotlyPlot(
        plot_type=plot_type,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        plot_obj=scatter_obj,
        xlim=xlim,
        ylim=ylim,
        xscale=xscale,
        yscale=yscale,
        x_dtick=x_dtick,
        y_dtick=y_dtick,
    )


def violin(
    y: Sequence[float],
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    name: Optional[str] = None,
    xlim: Optional[List[float]] = None,
    ylim: Optional[List[float]] = None,
    xscale: Optional[List[float]] = None,
    yscale: Optional[List[float]] = None,
    x_dtick: Optional[float] = None,
    y_dtick: Optional[float] = None,
) -> EZPlotlyPlot:
    """
    Make a single violin plot.

    :param y: The data for the violin plot as `Sequence[float]`
    :param title:  The title of the plot as `Optional[str]`
    :param xlabel: The x-axis label as `Optional[str]`
    :param ylabel: The y-axis label as `Optional[str]`
    :param name: The name of the violin plot as `Optional[str]` (useful for plotting series)
    :param xlim: The x-axis limits [x_left_lim, x_right_lim] as `Optional[List[float]]`
    :param ylim: The y-axis limits [y_left_lim, y_right_lim] as `Optional[List[float]]`
    :param xscale: The scale of the x-axis ('log', 'linear') as `Optional[str]`
    :param yscale: The scale of the y-axis ('log', 'linear') as `Optional[str]`
    :param x_dtick: The plotting delta tick (i.e. tick length) of the x-axis as `Optional[float]`
    :param y_dtick: The plotting delta tick (i.e. tick length) of the y-axis as `Optional[float]`
    :return:
        EZPlotlyPlot object representing violin plot
    """

    # plot type
    plot_type = "violin"

    # legend properties
    showlegend = name is not None

    # make violin object
    violin_obj = go.Violin(y=y, name=name, showlegend=showlegend)

    # return
    return EZPlotlyPlot(
        plot_type=plot_type,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        plot_obj=violin_obj,
        xlim=xlim,
        ylim=ylim,
        xscale=xscale,
        yscale=yscale,
        x_dtick=x_dtick,
        y_dtick=y_dtick,
    )


def scatterheat(
    x: Sequence[float],
    y: Sequence[float],
    xbins: Sequence[float],
    ybins: Sequence[float],
    zscale: str = "linear",
    error_y: Optional[Union[Sequence[float], np.array]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    name: Optional[str] = None,
    xlim: Optional[List[float]] = None,
    ylim: Optional[List[float]] = None,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    x_dtick: Optional[float] = None,
    y_dtick: Optional[float] = None,
    cscale: Optional[List[float]] = None,
    show_text_heatmap: bool = False,
    texttemplate_heatmap: str = "%{text}",
    text_size_heatmap: Optional[int] = None,
    outfile: Optional[str] = None,
    plot: bool = True,
) -> Optional[List[EZPlotlyPlot]]:
    """
    Plots a scatterplot and density heatmap side-by-side.

    :param x: The x-axis data as `Sequence[float]`
    :param y: The y-axis data as `Sequence[float]`
    :param xbins: x-axis bin edges for density heatmap as `Sequence[float]`
    :param ybins: y-axis bin edges for density heatmap as `Sequence[float]`
    :param zscale: The scale of the frequency dimension ('log', 'linear') as `str`
    :param error_y: The y-error data to plot as `Optional[Union[Sequence[float], np.ndarray]]`
        If `error_y` is a sequence, it must be the same length as `y`.
        If `error_y` is a numpy array of shape (2, N), where N is the length of `y`,
         where the first row is the lower error and the second row is the upper error.
    :param title: Plot title as `Optional[str]`
    :param xlabel: The x-axis label as `Optional[str]`
    :param ylabel: The y-axis label as `Optional[str]`
    :param name: Name of plot as `Optional[str]`
    :param xlim: The x-axis limits [x_left_lim, x_right_lim] as `Optional[List[float]]`
    :param ylim: The y-axis limits [y_left_lim, y_right_lim] as `Optional[List[float]]`
    :param xscale: The scale of the x-axis ('log', 'linear') as `Optional[str]`
    :param yscale: The scale of the y-axis ('log', 'linear') as `Optional[str]`
    :param x_dtick: The plotting delta tick (i.e. tick length) of the x-axis as `Optional[float]`
    :param y_dtick: The plotting delta tick (i.e. tick length) of the y-axis as `Optional[float]`
    :param cscale: The color scale for heatmap [c_lower_lim, c_upper_lim] as `Optional[List[float]]`
    :param show_text_heatmap: Whether to plot text labels on heatmap or not as `bool`
    :param texttemplate_heatmap: The text template to use on the heatmap as `str`
    :param text_size_heatmap: Text size on heatmap labels as `int`
    :param outfile: If specified, save to this file as `Optional[str]`
    :param plot: Whether to plot right now or suppress plot and return the plot object
        for downstream plotting by the user as `bool`
    :return:
        List[EZPlotlyPlot] representing figure set (if plot=False)
    """
    # convert to numpy arrays if not already
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    if not isinstance(xbins, np.ndarray):
        xbins = np.array(xbins)
    if not isinstance(ybins, np.ndarray):
        ybins = np.array(ybins)
    bins = np.array([ybins, xbins])

    # transform data using np histogram
    z = np.histogram2d(x=y, y=x, bins=bins)[0]
    xlabels = xbins
    ylabels = ybins

    # transform data
    if zscale == "log":
        z = np.log10(z)

    # prepare heatmap text labels
    text: Optional[List[List[str]]] = None
    if show_text_heatmap:
        text = [[str(z[i, j]) for i in range(z.shape[0])] for j in range(z.shape[1])]

    # return plots
    figs: List[EZPlotlyPlot] = list()
    figs.append(
        scattergl(
            x=x,
            y=y,
            error_y=error_y,
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=xlim,
            ylim=ylim,
            xscale=xscale,
            yscale=yscale,
            x_dtick=x_dtick,
            y_dtick=y_dtick,
            name=name,
            marker_size=5,
        )
    )
    figs.append(
        heatmap(
            z=z,
            xlabels=xlabels,
            ylabels=ylabels,
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=xlim,
            ylim=ylim,
            xscale=xscale,
            yscale=yscale,
            x_dtick=x_dtick,
            y_dtick=y_dtick,
            cscale=cscale,
            text=text,
            texttemplate=texttemplate_heatmap,
            text_size=text_size_heatmap,
        )
    )
    if plot:
        plot_all(figs, title=title, outfile=outfile)
    else:
        return figs


def heatmap(
    z: Sequence[Sequence[float]],
    xlabels: Sequence[Any] = None,
    ylabels: Sequence[Any] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlim: Optional[List[float]] = None,
    ylim: Optional[List[float]] = None,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    x_dtick: Optional[float] = None,
    y_dtick: Optional[float] = None,
    cscale: Optional[List[float]] = None,
    showcscale: bool = True,
    text: Optional[List[List[str]]] = None,
    texttemplate: str = "%{text}",
    text_size: Optional[int] = None,
) -> EZPlotlyPlot:
    """
    Plot heatmap.

    :param z: 2-D heatmap as `Sequence[Sequence[float]]`
    :param xlabels: The xlabels as `Sequence[Any]`
    :param ylabels: The ylabels as `Sequence[Any]`
    :param title: The title of the heatmap as `Optional[str]`
    :param xlabel: The x-axis label as `Optional[str]`
    :param ylabel: The y-axis label as `Optional[str]`
    :param xlim: The x-axis limits [x_left_lim, x_right_lim] as `Optional[List[float]]`
    :param ylim: The y-axis limits [y_left_lim, y_right_lim] as `Optional[List[float]]`
    :param xscale: The scale of the x-axis ('log', 'linear') as `Optional[str]`
    :param yscale: The scale of the y-axis ('log', 'linear') as `Optional[str]`
    :param x_dtick: The plotting delta tick (i.e. tick length) of the x-axis as `Optional[float]`
    :param y_dtick: The plotting delta tick (i.e. tick length) of the y-axis as `Optional[float]`
    :param cscale: The color scale for heatmap [c_lower_lim, c_upper_lim] as `Optional[List[float]]`
    :param showcscale: Whether to show the color scale or not as `bool`
    :param text: Text per heatmap box as `Optional[List[List[str]]]`
    :param texttemplate: Text template as Optional[str]`
    :param text_size: Size of text as `Optional[int]`
    :return:
        EZPlotlyPlot object representing heatmap
    """

    # plot type
    plot_type = "heatmap"

    # color scale
    if cscale is None:
        zmin = None
        zmax = None
    else:
        zmin = cscale[0]
        zmax = cscale[1]

    # size
    if text_size is not None:
        textfont = {"size": text_size}
    else:
        textfont = {"size": 10}

    # make heatmap object
    heatmap_obj = go.Heatmap(
        z=z,
        x=xlabels,
        y=ylabels,
        zmin=zmin,
        zmax=zmax,
        showscale=showcscale,
        text=text,
        texttemplate=texttemplate,
        textfont=textfont,
    )

    # return
    return EZPlotlyPlot(
        plot_type=plot_type,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        plot_obj=heatmap_obj,
        xlim=xlim,
        ylim=ylim,
        xscale=xscale,
        yscale=yscale,
        x_dtick=x_dtick,
        y_dtick=y_dtick,
    )


def show(fig: Any) -> None:
    """
    Plots a single figure.

    :param fig: The figure to plot.
    """
    plotly.offline.iplot(fig, filename="Subplot")


def extract_panel_title_positions(fig: Any) -> Dict[Any, Any]:
    """
    Extracts panel title position.

    :param fig: The figure
    :return:
        Title positions in dict
    """
    title_annotations = list(fig["layout"]["annotations"])
    return {t["text"]: (t["x"], t["y"]) for t in title_annotations}


def plot_all(
    plots: Union[List[EZPlotlyPlot], EZPlotlyPlot],
    panels: Optional[List[int]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    withhold: Optional[bool] = False,
    numcols: int = 1,
    title: Optional[str] = None,
    showlegend: bool = False,
    paging: Optional[Dict[str, int]] = None,
    chrpacked: bool = False,
    outfile: Optional[str] = None,
    suppress_output: bool = False,
) -> Optional[Any]:
    """
    Global plotting function.

    :param plots: The plots to plot as `Union[List[EZPlotlyPlot], EZPlotlyPlot]`
    :param panels: A list of ints indicating which panel each plot goes into (if multiple panels) as
        `Optional[List[int]]`
    :param height: The height of the plotting area as `Optional[int]`
    :param width: The width of the plotting area as `Optional[int]`
    :param withhold: Whether to withhold the plot or plot right away as `bool`
    :param numcols: The number of panel columns per plotting row as `int`
    :param title: The title of the overall plot as `Optional[str]`
    :param showlegend: Whether to show the plot legends as `bool`
    :param chrpacked: Whether the plots are chrpacked (useful for bio chromosome plots) as `bool`
    :param paging: The paging settings as `Optional[Dict[str, int]].` Paging allows automatic splitting
                   of the plots into multiple subplots and subplots into different page *.png figures.
                   Dict contains the following parameters:
                        num_plots_per_subplot: The number of ezplotly plots per subplot as `int`
                        num_subplots_per_page: The number of subplots per page as `int`
    :param outfile: The file to write the plot to as `Optional[str]`
    :param suppress_output: Whether to suppress output or not as `bool.` Superseded by global plotly
        plot_settings.SUPPRESS_PLOT.
    :return:
        Base Plotly figure representing plot as `EZPlotlyPlot`
    """

    # if single EZPlotlyPlot, make list for consistency
    if isinstance(plots, EZPlotlyPlot):
        plots = [plots]

    # determine if paging is used and make recursive calls as necessary
    if paging is not None:
        # check paging parameters
        if "num_plots_per_subplot" not in paging:
            raise ValueError("num_plots_per_subplot must be specified in paging dict")
        if "num_subplots_per_page" not in paging:
            raise ValueError("num_subplots_per_page must be specified in paging dict")

        # make recursive calls
        num_plots_per_subplot = paging["num_plots_per_subplot"]
        num_subplots_per_page = paging["num_subplots_per_page"]
        num_subplots = int(np.ceil(len(plots) / num_plots_per_subplot))
        num_pages = int(np.ceil(num_subplots / num_subplots_per_page))
        for page_index in range(0, num_pages):
            start_subplot_index = page_index * num_subplots_per_page
            end_subplot_index = start_subplot_index + num_subplots_per_page
            start_plot_index = start_subplot_index * num_plots_per_subplot
            end_plot_index = min(end_subplot_index * num_plots_per_subplot, len(plots))
            if outfile is not None:
                outfile_ext = "." + outfile.split(".")[-1]
                new_outfile = outfile.replace(
                    outfile_ext, f"_{page_index}" + outfile_ext
                )
            else:
                new_outfile = None
            new_panels = list()
            panel_index = 0
            for plot_index in range(start_plot_index, end_plot_index):
                if plot_index % num_plots_per_subplot == 0:
                    panel_index += 1
                new_panels.append(panel_index)
            new_height = height
            if panel_index < num_subplots_per_page:
                new_height = height * panel_index / num_subplots_per_page
            plot_all(
                plots=plots[start_plot_index:end_plot_index],
                panels=new_panels,
                height=new_height,
                width=width,
                withhold=False,
                numcols=numcols,
                title=title,
                showlegend=showlegend,
                paging=None,
                chrpacked=chrpacked,
                outfile=new_outfile,
                suppress_output=suppress_output,
            )
        return None

    # compute num panels needed to display everything
    if panels is None:
        num_panels = len(plots)
        panels = range(1, len(plots) + 1)
    else:
        num_panels = np.max(panels)

    # determine layout of Nx2 grid and adjust sizing
    num_rows = int(np.ceil(num_panels / numcols))
    if height is None:
        if chrpacked:
            height = num_rows * 200
        else:
            height = num_rows * 300
    if width is None:
        if chrpacked:
            width = numcols * 200
        else:
            width = 1000

    # extract panel titles
    panel_titles = ["" for _ in range(0, num_panels)]
    for plot_index in range(0, len(plots)):
        p = plots[plot_index]
        plot_title = p[1]
        panel_index = panels[plot_index]
        if plot_title is not None:
            panel_titles[panel_index - 1] = plot_title

    # make overall figure
    fig = make_subplots(
        rows=num_rows, cols=numcols, subplot_titles=panel_titles, print_grid=False
    )
    title_positions = extract_panel_title_positions(fig)

    # loop over plot generation
    for plot_index in range(0, len(plots)):
        # property extraction
        panel_index = panels[plot_index]
        p = plots[plot_index]
        plot_title = p.title
        xlabel = p.xlabel
        ylabel = p.ylabel
        plot = p.plot_obj
        xlim = p.xlim
        ylim = p.ylim
        xscale = p.xscale
        yscale = p.yscale
        x_dtick = p.x_dtick
        y_dtick = p.y_dtick

        # row/col
        row_index = int((panel_index - 1) / numcols + 1)
        col_index = int((panel_index - 1) % numcols + 1)

        # set up axis for figure
        fig.add_trace(plot, row_index, col_index)
        fig["layout"]["xaxis" + str(panel_index)].update(showgrid=True)
        fig["layout"]["yaxis" + str(panel_index)].update(showgrid=True)

        # redo figure axis labels for chrPacked=True
        if chrpacked:
            if col_index != 1:
                ylabel = None
            if xlabel is not None:
                new_anno = dict(
                    text=xlabel,
                    x=title_positions[plot_title][0],
                    xanchor="center",
                    xref="paper",
                    y=-0.043 + (num_rows - row_index) * 0.22,
                    yanchor="bottom",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=14),
                )
                fig["layout"]["annotations"] += (new_anno,)

        # set figure labels
        if xlabel is not None and not chrpacked:
            fig["layout"]["xaxis" + str(panel_index)].update(title=xlabel)
        if ylabel is not None:
            fig["layout"]["yaxis" + str(panel_index)].update(title=ylabel)
        if xscale is not None:
            fig["layout"]["xaxis" + str(panel_index)].update(type=xscale)
        if yscale is not None:
            fig["layout"]["yaxis" + str(panel_index)].update(type=yscale)
        if x_dtick is not None:
            fig["layout"]["xaxis" + str(panel_index)].update(dtick=x_dtick)
        if y_dtick is not None:
            fig["layout"]["yaxis" + str(panel_index)].update(dtick=y_dtick)
        if xlim is not None:
            fig["layout"]["xaxis" + str(panel_index)].update(
                range=xlim, autorange=False, tick0=xlim[0]
            )
        if ylim is not None:
            fig["layout"]["yaxis" + str(panel_index)].update(
                range=ylim, autorange=False, tick0=ylim[0]
            )

    # set overall layout and either withold plot or display it
    fig["layout"].update(height=height, width=width, showlegend=showlegend, title=title)
    if withhold:  # return fig (if additional custom changes need to be made)
        return fig
    else:
        if not plot_settings.SUPPRESS_PLOTS and not suppress_output:
            plotly.offline.iplot(fig)
        if outfile is not None:
            plotly.io.write_image(fig, file=outfile)
