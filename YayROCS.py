import numpy as np

#input = predictor vector, ground truth vector
def roc(pred,gt):

	#make sure pred /gt are correct shape
	if (pred.ndim == 2) and (pred.shape[1] > pred.shape[0]):
		pred = pred.T
	elif pred.ndim==1:
		pred = pred.reshape(-1,1)
	if (gt.ndim == 2) and (gt.shape[1] > gt.shape[0]):
		gt = gt.T
	elif gt.ndim==1:
		gt = gt.reshape(-1,1)

	#make results table
	results = np.concatenate([pred,gt],axis=1)

	#compute step sizes
	a = results.shape[0]
	num_pos = np.sum(results[:,1])
	num_neg = a - num_pos
	step_x = 1.0/num_neg
	step_y = 1.0/num_pos
	current_x = 0.0
	current_y = 0.0

	#sort results
	ind = np.lexsort((results[:,1],-1*results[:,0]))
	results = results[ind]

	#roc compute
	fpr = np.zeros((A+1,))
	tpr = np.zeros((A+1,))
	cnt=1
	for i in range(0,A):
		if results[i, 1]==1:
			current_y = current_y + step_y
			fpr[cnt] = current_x
			tpr[cnt] = current_y
		else:
			current_x = current_x + step_x
			fpr[cnt] = current_x
			tpr[cnt] = current_y
		cnt = cnt + 1

	#resize and return
	tpr = tpr[0:cnt]
	fpr = fpr[0:cnt]
	return fpr, tpr

#input 1 roc (X,Y)
def auc(x,y):

	#align to reference binning
	ref_binning = np.arange(0,1.1,0.01)
	bin_ind = 0
	auc_val = 0
	for i in range(0,len(ref_binning)):
		#find appropriate bin to draw
		while (bin_ind < (len(x) - 1)) and (ref_binning[i] > x[bin_ind]):
			bin_ind = bin_ind + 1
		auc_val = auc_val + y[bin_ind]
	auc_val = auc_val / len(ref_binning)
	return auc_val
	
#compute p-value based on distribution and observed point
def compute_p_value(distr,observed,tailed='two-tailed'):

	#combine data and sort
	combined_data = np.append(distr,observed)
	combined_data.sort()
	n = len(combined_data)

	#identify ranks
	rank = np.where(combined_data==observed)
	lrank = rank[0][0]
	rrank = N-rank[0][-1]-1

	#tails
	lp = lrank / n
	rp = rrank / n
	if tailed== 'left':
		p_val = lp
	elif tailed== 'right':
		p_val = rp
	elif tailed== 'two-tailed':
		p_val = np.min([lp,rp])
	return p_val