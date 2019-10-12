import plotly
plotly.offline.init_notebook_mode() # run at the start of every notebook
import plotly.graph_objs as go
import numpy as np
import copy

def hist(data,minBin=None,maxBin=None,binSize=None,histnorm=None,title=None,name=None,xlabel=None,ylabel=None,color=None,xScale=None,yScale=None,x_dTick=None,y_dTick=None,xlim=None,ylim=None):

	#plot type
	plotType='histogram'
	if(ylabel is None):
		if(histnorm=='probability'):
			ylabel = 'Probability'
		else:
			ylabel = 'Frequency'
		if(yScale=='log'):
			ylabel = 'log ' + ylabel

	#xbin properties
	if(minBin is None or maxBin is None or binSize is None):
		xbins=None
	else:
		if(xlim is None):
			xlim=[minBin,maxBin]
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
		marker=marker,
		histnorm=histnorm
	)

	#return
	return (plotType,title,xlabel,ylabel,histObj,xlim,ylim,xScale,yScale,x_dTick,y_dTick)

def bar(y,x=(),error_y=None,title=None,xlabel=None,ylabel=None,xlim=None,ylim=None,name=None,xScale=None,yScale=None,x_dTick=None,y_dTick=None):

	#plot type
	plotType='bar'

	#define x
	if(len(x)==0):
		x = [a for a in range(0,len(y))]

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
	return (plotType,title,xlabel,ylabel,barObj,xlim,ylim,xScale,yScale,x_dTick,y_dTick)

def scattergl(x,y,title=None,xlabel=None,ylabel=None,xlim=None,ylim=None,markerSize=None,markerColor=None,name=None,pointAnno=None,xScale=None,yScale=None,x_dTick=None,y_dTick=None):

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
		name=name,
		x=x,
		y=y,
		mode='markers',
		visible=True,
		marker=marker,
		text=pointAnno
	)
	
	#return
	return(plotType,title,xlabel,ylabel,scatterObj,xlim,ylim,xScale,yScale,x_dTick,y_dTick)

def line(x,y,title=None,xlabel=None,ylabel=None,xlim=None,ylim=None,width=None,color=None,pointAnno=None,name=None,xScale=None,yScale=None,x_dTick=None,y_dTick=None):

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
		name=name,
		x=x,
		y=y,
		line=line
	)

	#return
	return(plotType,title,xlabel,ylabel,scatterObj,xlim,ylim,xScale,yScale,x_dTick,y_dTick)

def violin(y,title=None,xlabel=None,ylabel=None,name=None,xlim=None,ylim=None,xScale=None,yScale=None,x_dTick=None,y_dTick=None):

	#plot type
	plotType='violin'

	#make violin object
	violinObj = go.Violin(
		y=y,
		name=name
	)

	#return
	return(plotType,title,xlabel,ylabel,violinObj,xlim,ylim,xScale,yScale,x_dTick,y_dTick)

def heatmap(z,xlabels=None,ylabels=None,title=None,xlabel=None,ylabel=None,name=None,xlim=None,ylim=None,xScale=None,yScale=None,x_dTick=None,y_dTick=None,cScale=None):
	
	#plot type
	plotType = 'heatmap'

	#color scale
	if(cScale is None):
		zmin=None
		zmax=None
	else:
		zmin=cScale[0]
		zmax=cScale[1]

	#make heatmap object
	heatmapObj = go.Heatmap(z=z,x=xlabels,y=ylabels,zmin=zmin,zmax=zmax)

	#return
	return(plotType,title,xlabel,ylabel,heatmapObj,xlim,ylim,xScale,yScale,x_dTick,y_dTick)

def show(fig):
	plotly.offline.iplot(fig,filename='Subplot')

def extractPanelTitlePositions(fig):
	titleAnnotations = list(fig['layout']['annotations'])
	return {t['text']: (t['x'],t['y']) for t in titleAnnotations}

def plotAll(plots,panels=None,height=None,width=None,withhold=False,numCols=1,title=None,showLegend=False,chrPacked=False,outFile=None):

	#compute num panels needed to display everything
	if(panels is None):
		numPanels = len(plots)
		panels = range(1,len(plots)+1)
	else:
		numPanels= np.max(panels)

	#determine layout of Nx2 grid and adjust sizing
	numRows = int(np.ceil(numPanels/numCols))
	if(height is None):
		if(chrPacked):
			height = numRows*200
		else:
			height = numRows*300
	if(width is None):
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
		xScale = p[7]
		yScale = p[8]
		x_dTick = p[9]
		y_dTick = p[10]

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
		if(xScale!=None):
			fig['layout']['xaxis'+str(panelIndex)].update(type=xScale)
		if(yScale!=None):
			fig['layout']['yaxis'+str(panelIndex)].update(type=yScale)
		if(x_dTick!=None):
			fig['layout']['xaxis'+str(panelIndex)].update(dtick=x_dTick)
		if(y_dTick!=None):
			fig['layout']['yaxis'+str(panelIndex)].update(dtick=y_dTick)
		if(xlim!=None):
			fig['layout']['xaxis'+str(panelIndex)].update(range=xlim,autorange=False,tick0=xlim[0])
		if(ylim!=None):
			fig['layout']['yaxis'+str(panelIndex)].update(range=ylim,autorange=False,tick0=ylim[0])

	#set overall layout and either withold plot or display it
	fig['layout'].update(height=height,width=width,showlegend=showLegend,title=title)
	if(withhold): 	#return fig (if additional custom changes need to be made)
		return fig
	else:
		plotly.offline.iplot(fig,filename='Subplot')
		if(outFile!=None):
			plotly.io.write_image(fig,file=outFile)