import plotly
import plotly.express as px
#plotly.offline.init_notebook_mode()  # run at the start of every notebook
import plotly.graph_objs as go
import numpy as np
import copy


def hist(data, min_bin=None, max_bin=None, bin_size=None, histnorm=None, title=None, name=None, xlabel=None,
         ylabel=None, color=None, xscale=None, yscale=None, x_dtick=None, y_dtick=None, xlim=None, ylim=None):
    # plot type
    plot_type = 'histogram'
    if ylabel is None:
        if histnorm == 'probability':
            ylabel = 'Probability'
        else:
            ylabel = 'Frequency'
        if yscale == 'log':
            ylabel = 'log ' + ylabel

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
        marker['color'] = color

    # assemble hist object
    hist_obj = go.Histogram(
        x=data,
        name=name,
        xbins=xbins,
        marker=marker,
        histnorm=histnorm
    )

    # return
    return plot_type, title, xlabel, ylabel, hist_obj, xlim, ylim, xscale, yscale, x_dtick, y_dtick


def bar(y, x=(), error_y=None, title=None, xlabel=None, ylabel=None, xlim=None, ylim=None, name=None, xscale=None,
        yscale=None, x_dtick=None, y_dtick=None, text=None, textposition=None, color=None):
    # plot type
    plot_type = 'bar'

    # define x
    if len(x) == 0:
        x = [a for a in range(0, len(y))]

    # assemble bar object
    bar_obj = go.Bar(
        name=name,
        x=x,
        y=y,
        error_y=dict(
            type='data',
            array=error_y,
            visible=True
        ),
        text=text,
        textposition=textposition,
        marker_color=color
    )

    # return
    return plot_type, title, xlabel, ylabel, bar_obj, xlim, ylim, xscale, yscale, x_dtick, y_dtick


def scatter(x, y, error_y=None, title=None, xlabel=None, ylabel=None, xlim=None, ylim=None, marker_size=None,
            marker_color=None, name=None, point_anno=None, xscale=None, yscale=None, x_dtick=None, y_dtick=None):
    # plot type
    plot_type = 'scatter'

    # assemble marker information
    marker = dict()
    if marker_size is not None:
        marker['size'] = marker_size
    if marker_color is not None:
        marker['color'] = marker_color

    # make scatter gl object
    scatter_obj = go.Scatter(
        name=name,
        x=x,
        y=y,
        mode='markers',
        visible=True,
        marker=marker,
        text=point_anno,
        error_y=dict(
            type='data',
            array=error_y,
            visible=True
        )
    )

    # return
    return plot_type, title, xlabel, ylabel, scatter_obj, xlim, ylim, xscale, yscale, x_dtick, y_dtick


def scattergl(x, y, error_y=None, title=None, xlabel=None, ylabel=None, xlim=None, ylim=None, marker_size=None,
              marker_color=None, name=None, point_anno=None, xscale=None, yscale=None, x_dtick=None, y_dtick=None):
    # plot type
    plot_type = 'scattergl'

    # assemble marker information
    marker = dict()
    if marker_size is not None:
        marker['size'] = marker_size
    if marker_color is not None:
        marker['color'] = marker_color

    # make scatter gl object
    scatter_obj = go.Scattergl(
        name=name,
        x=x,
        y=y,
        mode='markers',
        visible=True,
        marker=marker,
        text=point_anno,
        error_y=dict(
            type='data',
            array=error_y,
            visible=True
        )
    )

    # return
    return plot_type, title, xlabel, ylabel, scatter_obj, xlim, ylim, xscale, yscale, x_dtick, y_dtick


def line(x, y, error_y=None, lcl=None, ucl=None, title=None, xlabel=None, ylabel=None, xlim=None, ylim=None, width=None,
         color=None, name=None, xscale=None, yscale=None, x_dtick=None, y_dtick=None, dash=None):
    # plot type
    plot_type = 'line'

    # assemble line information
    line = dict()
    if width is not None:
        line['width'] = width
    if color is not None:
        line['color'] = color
    if dash is not None:
        line['dash'] = dash
    if lcl is not None and ucl is not None:
        error_y_dict = dict(
            type='data',
            symmetric=False,
            array=ucl,
            arrayminus=lcl,
            visible=True
        )
    elif error_y is not None:
        error_y_dict = dict(
            type='data',
            array=error_y,
            visible=True
        )
    else:
        error_y_dict = None

    # make scatter object
    scatter_obj = go.Scattergl(
        name=name,
        x=x,
        y=y,
        error_y=error_y_dict,
        line=line
    )

    # return
    return plot_type, title, xlabel, ylabel, scatter_obj, xlim, ylim, xscale, yscale, x_dtick, y_dtick


def violin(y, title=None, xlabel=None, ylabel=None, name=None, xlim=None, ylim=None, xscale=None, yscale=None,
           x_dtick=None, y_dtick=None):
    # plot type
    plot_type = 'violin'

    # make violin object
    violin_obj = go.Violin(
        y=y,
        name=name
    )

    # return
    return plot_type, title, xlabel, ylabel, violin_obj, xlim, ylim, xscale, yscale, x_dtick, y_dtick


def scatterheat(x, y, xbins, ybins, zscale='linear', title=None, xlabel=None, ylabel=None, name=None, xlim=None,
                ylim=None, xscale=None, yscale=None, x_dtick=None, y_dtick=None, cscale=None, outfile=None, plot=True):
    # transform data using np histogram
    z = np.histogram2d(x=y, y=x, bins=[ybins, xbins])[0]
    xlabels = xbins
    ylabels = ybins

    # transform data
    if zscale == 'log':
        z = np.log10(z)

    # return plots
    h = [None] * 2
    h[0] = scattergl(x, y, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, xscale=xscale, yscale=yscale,
                     x_dtick=x_dtick, y_dtick=y_dtick, name=name, marker_size=5)
    h[1] = heatmap(z=z, xlabels=xlabels, ylabels=ylabels, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim,
                   xscale=xscale, yscale=yscale, x_dtick=x_dtick, y_dtick=y_dtick, cscale=cscale)
    if plot:
        plot_all(h, title=title, outfile=outfile)
    else:
        return h


def heatmap(z, xlabels=None, ylabels=None, title=None, xlabel=None, ylabel=None, xlim=None, ylim=None, xscale=None,
            yscale=None, x_dtick=None, y_dtick=None, cscale=None, showscale=True):
    # plot type
    plot_type = 'heatmap'

    # color scale
    if cscale is None:
        zmin = None
        zmax = None
    else:
        zmin = cscale[0]
        zmax = cscale[1]

    # make heatmap object
    heatmap_obj = go.Heatmap(z=z, x=xlabels, y=ylabels, zmin=zmin, zmax=zmax, showscale=showscale)

    # return
    return plot_type, title, xlabel, ylabel, heatmap_obj, xlim, ylim, xscale, yscale, x_dtick, y_dtick


def show(fig):
    plotly.offline.iplot(fig, filename='Subplot')


def extract_panel_title_positions(fig):
    title_annotations = list(fig['layout']['annotations'])
    return {t['text']: (t['x'], t['y']) for t in title_annotations}


def plot_all(plots, panels=None, height=None, width=None, withhold=False, numcols=1, title=None, showlegend=False,
             chrpacked=False, outfile=None, suppress_output=False):
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
    panel_titles = ['' for a in range(0, num_panels)]
    for plot_index in range(0, len(plots)):
        p = plots[plot_index]
        plot_title = p[1]
        panel_index = panels[plot_index]
        if plot_title is not None:
            panel_titles[panel_index - 1] = plot_title

    # make overall figure
    fig = plotly.subplots.make_subplots(rows=num_rows, cols=numcols, subplot_titles=panel_titles, print_grid=False)
    title_positions = extract_panel_title_positions(fig)

    # loop over plot generation
    for plot_index in range(0, len(plots)):

        # property extraction
        panel_index = panels[plot_index]
        p = plots[plot_index]
        plot_type = p[0]
        plot_title = p[1]
        xlabel = p[2]
        ylabel = p[3]
        plot = p[4]
        xlim = p[5]
        ylim = p[6]
        xscale = p[7]
        yscale = p[8]
        x_dtick = p[9]
        y_dtick = p[10]

        # row/col
        row_index = int((panel_index - 1) / numcols + 1)
        col_index = int((panel_index - 1) % numcols + 1)

        # set up axis for figure
        fig.append_trace(plot, row_index, col_index)
        fig['layout']['xaxis' + str(panel_index)].update(showgrid=True)
        fig['layout']['yaxis' + str(panel_index)].update(showgrid=True)

        # redo figure axis labels for chrPacked=True
        if chrpacked:
            if col_index != 1:
                ylabel = None
            if xlabel is not None:
                new_anno = dict(
                    text=xlabel,
                    x=title_positions[plot_title][0],
                    xanchor='center',
                    xref='paper',
                    y=-0.043 + (num_rows - row_index) * 0.22,
                    yanchor='bottom',
                    yref='paper',
                    showarrow=False,
                    font=dict(size=14)
                )
                fig['layout']['annotations'] += (new_anno,)

        # set figure labels
        if xlabel is not None and not chrpacked:
            fig['layout']['xaxis' + str(panel_index)].update(title=xlabel)
        if ylabel is not None:
            fig['layout']['yaxis' + str(panel_index)].update(title=ylabel)
        if xscale is not None:
            fig['layout']['xaxis' + str(panel_index)].update(type=xscale)
        if yscale is not None:
            fig['layout']['yaxis' + str(panel_index)].update(type=yscale)
        if x_dtick is not None:
            fig['layout']['xaxis' + str(panel_index)].update(dtick=x_dtick)
        if y_dtick is not None:
            fig['layout']['yaxis' + str(panel_index)].update(dtick=y_dtick)
        if xlim is not None:
            fig['layout']['xaxis' + str(panel_index)].update(range=xlim, autorange=False, tick0=xlim[0])
        if ylim is not None:
            fig['layout']['yaxis' + str(panel_index)].update(range=ylim, autorange=False, tick0=ylim[0])

    # set overall layout and either withold plot or display it
    fig['layout'].update(height=height, width=width, showlegend=showlegend, title=title)
    if withhold:  # return fig (if additional custom changes need to be made)
        return fig
    else:
        if not suppress_output:
            plotly.offline.iplot(fig, filename='Subplot')
        if outfile is not None:
            plotly.io.write_image(fig, file=outfile)
