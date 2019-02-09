# EasyPlotly
An easy wrapper for making plotly plots in python notebooks

import EasyPlotly as EP
exampleHist = EP.hist(data=a,minBin=0.0,maxBin=1.0,binSize=0.1,title='MyHistogram',xlabel='a',ylabel='Frequency)
exampleScatter = EP.scatter(x=a,y=b,title='Test',xlabel='x',ylabel='y')
EP.plotAll([exampleHist,exampleScatter])
