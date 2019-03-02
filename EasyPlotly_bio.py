import EasyPlotly as EP
import pandas as pd
import scipy.stats as sc
import numpy as np

def chrRollingMedian(chrPosDF,chrValDF,rollwinsize,ylabel=None,title=None,withhold=False,ylim=None):

	#plot
	x=chrPosDF.values
	y=chrValDF.rolling(rollwinsize).median()
	linePlot = EP.line(x=x,y=y,title=title,xlabel='Chr Position',ylabel=ylabel,ylim=ylim)
	if(withhold):
		return linePlot
	else:
		EP.plotAll([linePlot])

def chrCount(boolVals,chrDF,title=None,withhold=False):

	#extract unique chr vals
	uniqVals = chrDF.unique()

	#make chromosome-level counts
	counts = np.zeros((len(uniqVals),))
	cnt=0
	for u in uniqVals:
		counts[cnt] = np.sum(boolVals[chrDF==u])
		cnt = cnt + 1
	
	#plot
	barPlot = EP.bar(counts,x=uniqVals,title=title,xlabel='Chromosome',ylabel='Count')
	if(withhold):
		return barPlot
	else:
		EP.plotAll([barPlot])

def chrHist(df,chrCol,colName,minBin=None,maxBin=None,binSize=None,title=None):
	
	#extract unique chr values
	uniqVals = df.iloc[:,chrCol].unique()

	#make chromosome level hists
	hists = list()
	for u in uniqVals:
		data = df[colName][df.iloc[:,chrCol]==u]
		EPHist = EP.hist(data,title=u,xlabel=colName,minBin=minBin,maxBin=maxBin,binSize=binSize,ylabel='Count',color='#1ad1ff')
		hists.append(EPHist)
	
	#make plot
	EP.plotAll(hists,numCols=5,title=title,chrPacked=True)

def chrQQ(df,chrCol,colName,sparams=(),dist='norm',title=None):

	#extract unique chr values
	uniqVals = df.iloc[:,chrCol].unique()

	#make chromosome level qq-plots
	plots = list()
	panels = list()
	panelIndex=1
	for u in uniqVals:
		data = df[colName][df.iloc[:,chrCol]==u].values
		qq = qqplot(data,sparams=sparams,dist=dist,title=u)
		plots.append(qq[0])
		plots.append(qq[1])
		panels.append(panelIndex)
		panels.append(panelIndex)
		panelIndex = panelIndex + 1	

	#make plot
	EP.plotAll(plots,panels=panels,numCols=5,height=1000,title=title,chrPacked=True)

def qqplot(data,sparams=(),dist='norm',title=None):
	qq = sc.probplot(data,dist=dist,sparams=sparams)
	x=np.array([qq[0][0][0],qq[0][0][-1]])
	ptsScatter = EP.scattergl(x=qq[0][0],y=qq[0][1],title=title,xlabel='Expected',ylabel='Observed',markerSize=5,markerColor='blue',xlim=[0.0,1.05])
	linePlot = EP.line(x=x,y=qq[1][1] + qq[1][0]*x,width=3,color='red',title=title)
	return (ptsScatter,linePlot)