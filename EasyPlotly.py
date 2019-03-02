import plotly
plotly.offline.init_notebook_mode() # run at the start of every notebook
import plotly.graph_objs as go
import numpy as np
import copy

def hist(data,minBin=None,maxBin=None,binSize=None,title=None,name='hist',xlabel='x',ylabel='Frequency',color=None):

	#plot type
	plotType='histogram'

	#xbin properties
	if(minBin==None or maxBin==None or binSize==None):
		xlim=None
		ylim=None
		xbins=None
	else:
		xlim=[minBin,maxBin]
		ylim=None
		xbins=dict(start=minBin,end=maxBin,size=binSize)

	#assemble marker properties
	marker = dict()
	if(color!=None):
		marker['color']=color

	#assemble hist object
	histObj = go.Histogram(
		x=data,
		name=name,
		xbins=xbins,
		marker=marker
	)

	#return
	return (plotType,title,xlabel,ylabel,histObj,xlim,ylim)

def bar(y,x=(),error_y=None,title=None,xlabel=None,ylabel=None,xlim=None,ylim=None,name=None):

	#plot type
	plotType='bar'

	#define x
	if(len(x)==0):
		x = range(0,len(y))	

	#assemble bar object
	barObj = go.Bar(
		name=name,
		x=x,
		y=y,
		error_y=dict(
			type='data',
			array=error_y,
			visible=True
		)
	)

	#return
	return (plotType,title,xlabel,ylabel,barObj,xlim,ylim)

def scattergl(x,y,title=None,xlabel=None,ylabel=None,xlim=None,ylim=None,markerSize=None,markerColor=None):

	#plot type
	plotType='scattergl'

	#assemble marker information
	marker = dict()
	if(markerSize!=None):
		marker['size'] = markerSize
	if(markerColor!=None):
		marker['color'] = markerColor

	#make scatter gl object
	scatterObj = go.Scattergl(
		x=x,
		y=y,
		mode='markers',
		visible=True,
		marker=marker
	)
	
	#return
	return(plotType,title,xlabel,ylabel,scatterObj,xlim,ylim)

def line(x,y,title=None,xlabel=None,ylabel=None,xlim=None,ylim=None,width=None,color=None):

	#plot type
	plotType='line'

	#assemble line information
	line = dict()
	if(width!=None):
		line['width']=width
	if(color!=None):
		line['color']=color

	#make scatter object
	scatterObj = go.Scattergl(
		x=x,
		y=y,
		line=line	
	)

	#return
	return(plotType,title,xlabel,ylabel,scatterObj,xlim,ylim)

def show(fig):
	plotly.offline.iplot(fig,filename='Subplot')

def extractPanelTitlePositions(fig):
	titleAnnotations = list(fig['layout']['annotations'])
	return {t['text']: (t['x'],t['y']) for t in titleAnnotations}

def plotAll(plots,panels=None,height=None,width=None,withhold=False,numCols=2,title=None,showlegend=False,chrPacked=False):

	#compute num panels needed to display everything
	if(panels==None):
		numPanels = len(plots)
		panels = range(1,len(plots)+1)
	else:
		numPanels= np.max(panels)

	#determine layout of Nx2 grid and adjust sizing
	numRows = int(np.ceil(numPanels/numCols))
	if(height==None):
		if(chrPacked):
			height = numRows*200
		else:
			height = numRows*300
	if(width==None):
		if(chrPacked):
			width = numCols*200
		else:
			width=1000

	#extract panel titles
	panelTitles = ['' for a in range(0,numPanels)]
	for plotIndex in range(0,len(plots)):
		p = plots[plotIndex]
		plotTitle = p[1]
		panelIndex = panels[plotIndex]
		if(plotTitle!=None):
			panelTitles[panelIndex-1] = plotTitle

	#make overall figure
	fig = plotly.tools.make_subplots(rows=numRows,cols=numCols,subplot_titles=panelTitles,print_grid=False)
	titlePositions = extractPanelTitlePositions(fig)

	#loop over plot generation
	for plotIndex in range(0,len(plots)):	

		#property extraction	
		panelIndex = panels[plotIndex]
		p=plots[plotIndex]
		plotType = p[0]
		plotTitle = p[1]
		xlabel = p[2]
		ylabel = p[3]
		plot = p[4]
		xlim = p[5]
		ylim = p[6]

		#row/col
		rowIndex = int((panelIndex-1) / numCols + 1)
		colIndex = int((panelIndex-1) % numCols + 1)

		#set up axis for figure
		fig.append_trace(plot,rowIndex,colIndex)
		fig['layout']['xaxis'+str(panelIndex)].update(showgrid=True)
		fig['layout']['yaxis'+str(panelIndex)].update(showgrid=True)

		#redo figure axis labels for chrPacked=True
		if(chrPacked):
			if(colIndex!=1):
				ylabel=None
			if(xlabel!=None):
				newAnno = dict(
					text=xlabel,
					x=titlePositions[plotTitle][0],
					xanchor='center',
					xref='paper',
					y=-0.043+(numRows-rowIndex)*0.22,
					yanchor= 'bottom',
					yref='paper',
					showarrow=False,
					font=dict(size=14)
				)
				fig['layout']['annotations'] += (newAnno,)
		
		#set figure labels
		if(xlabel!=None and not chrPacked):
			fig['layout']['xaxis'+str(panelIndex)].update(title=xlabel)
		if(ylabel!=None):
			fig['layout']['yaxis'+str(panelIndex)].update(title=ylabel)
		if(xlim!=None):
			fig['layout']['xaxis'+str(panelIndex)].update(range=xlim,autorange=False)
		if(ylim!=None):
			fig['layout']['yaxis'+str(panelIndex)].update(range=ylim,autorange=False)

	#set overall layout and either withold plot or display it
	fig['layout'].update(height=height,width=width,showlegend=showlegend,title=title)
	if(withhold): 	#return fig (if additional custom changes need to be made)
		return fig
	else:
		plotly.offline.iplot(fig,filename='Subplot')