# EasyPlotly For Python Notebooks
An easy wrapper for making plotly plots in python notebooks

```python
import EasyPlotly as EP

exampleHist = EP.hist(data=a,minBin=0.0,maxBin=1.0,binSize=0.1,title='MyHistogram',xlabel='a')
exampleScatter = EP.scattergl(x=a,y=b,title='Test',xlabel='x',ylabel='y')
EP.plotAll([exampleHist,exampleScatter])
```
