# EasyPlotly For Python Notebooks
An easy wrapper for making plotly plots in python notebooks

```python
import EasyPlotly as EP

exampleHist = EP.hist(data=a,min_bin=0.0,max_bin=1.0,bin_size=0.1,title='MyHistogram',xlabel='a')
exampleScatter = EP.scattergl(x=a,y=b,title='Test',xlabel='x',ylabel='y')
EP.plotAll([exampleHist,exampleScatter])
```

There's also a bioinformatics extension (EasyPlotly_bio) in the works for making common bioinformatics plots such as qqplots, chromosome rolling medians, chromsome count bar charts, and chromosome histograms.
