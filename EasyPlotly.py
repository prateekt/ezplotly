import plotly
plotly.offline.init_notebook_mode() # run at the start of every notebook
import plotly.graph_objs as go
import numpy as np

def hist(data,minBin=0.0,maxBin=1.0,binSize=0.1,title='Histogram',xlabel='Data',ylabel='Frequency'):
	plotType='histogram'
	histObj = go.Histogram(
		x=data,
		xbins=dict(
			start=minBin,
			end=maxBin,
			size=binSize
		)
	)
	return (plotType,title,xlabel,ylabel,histObj)

def bar(x,y,error_y=None,title='Bar Chart',xlabel='Data',ylabel='Data'):
	plotType='bar'	
	barObj = go.Bar(
		x=x,
		y=y,
		error_y=dict(
			type='data',
			array=error_y,
			visible=True
		)
	)
	return (plotType,title,xlabel,ylabel,barObj)

def scattergl(x,y,title='Scatter',xlabel='x',ylabel='y'):
	plotType='scattergl'
	scatterObj = go.Scattergl(
		x=x,
		y=y,
		mode='markers',
		visible=True
	)
	return(plotType,title,xlabel,ylabel,scatterObj)

def line(x,y,title='Line',xlabel='x',ylabel='y'):
	plotType='line'
	scatterObj = go.Scatter(
		x=x,
		y=y,
		line= dict(
			width=4
		)
	)
	return(plotType,title,xlabel,ylabel,scatterObj)

def show(fig):
	plotly.offline.iplot(fig,filename='Subplot')

def plotAll(plots,height=None,width=None,withhold=False):

	#determine layout of Nx2 grid
	numPlots = len(plots)
	numRows = int(np.ceil(numPlots/2.0))
	if(height==None):
		height = numRows*300
	if(width==None):
		width = 1000

	#extract plot types and titles
	plotTypes = [p[0] for p in plots]
	titles = [p[1] for p in plots]

	#make overall figure
	fig = plotly.tools.make_subplots(rows=numRows,cols=2,subplot_titles=titles,print_grid=False)

	#loop over plot generation
	row=1
	col=1	
	for pinx in range(0,len(plots)):
		if(plotTypes[pinx] in ['histogram','bar','scattergl','line']):
			p=plots[pinx]
			xlabel = p[2]
			ylabel = p[3]
			plot = p[4]
			fig.append_trace(plot,row,col)
			col = col + 1
			if(col==3):
				col=1
				row = row + 1
			fig['layout']['xaxis'+str(pinx+1)].update(title=xlabel)
			fig['layout']['yaxis'+str(pinx+1)].update(title=ylabel)

	#set layout and make plot
	fig['layout'].update(height=height,width=width,showlegend=False)
	if(withhold): 	#return fig (if additional custom changes need to be made)
		return fig
	else:
		plotly.offline.iplot(fig,filename='Subplot')