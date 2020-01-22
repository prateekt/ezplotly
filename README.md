# EasyPlotly and EasyPlotly_bio For Python Notebooks
An easy wrapper for making plotly plots in python jupyter notebooks

Plotly offers interactive plots, as opposed to the static plots that most other python visualization tools provide. However, Plotly syntax can be challenging to write, whereas the other libraries are a lot easier to plot with. EasyPlotly helps bridge the gap.

To make plotling with Plotly in python simpler (and more matplotlib or matlab-like), we introduce EasyPlotly -- a wrapper on plotly that makes plotting WAY easier. Works extremely well in a python Jupyter notebook environment!

Example syntax:

```python
import EasyPlotly as EP

exampleHist = EP.hist(data=a,min_bin=0.0,max_bin=1.0,bin_size=0.1,title='MyHistogram',xlabel='a')
exampleScatter = EP.scattergl(x=a,y=b,title='Test',xlabel='x',ylabel='y')
EP.plotAll([exampleHist,exampleScatter])
```

In the bioinformatics domain? Currently in the works is a bioinformatics extension (EasyPlotly_bio) for making common bioinformatics plots such as qqplots, chromosome rolling medians, chromsome count bar charts, and chromosome histograms.
