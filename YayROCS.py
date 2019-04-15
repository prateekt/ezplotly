import numpy as np

#input = predictor vector, ground truth vector
def roc(pred,gt):

	#make results table
	results = np.concatenate([pred,gt],axis=1)

	#compute step sizes
	A = results.shape[0]
	numPos = np.sum(results[:,1])
	numNeg = A - numPos
	stepX = 1.0/numNeg
	stepY = 1.0/numPos
	currentX= 0.0
	currentY= 0.0

	#sort results
	ind = np.lexsort((results[:,1],-1*results[:,0]))
	results = results[ind]

	#roc compute
	FPR = np.zeros((A+1,))
	TPR = np.zeros((A+1,))
	cnt=1
	for i in range(0,A):
		if(results[i,1]==1):
			currentY = currentY + stepY
			FPR[cnt] = currentX
			TPR[cnt] = currentY
		else:
			currentX = currentX + stepX
			FPR[cnt] = currentX
			TPR[cnt] = currentY
		cnt = cnt + 1

	#resize and return
	cnt = cnt - 1
	TPR = TPR[0:cnt].reshape(1,-1)
	FPR = FPR[0:cnt].reshape(1,-1)
	return(FPR,TPR)

#input (N rocs by M binning of ROC)
def auc(X,Y):

	#extract lengths
	N = X.shape[0]

	#compute AUCs for each roc
	AUCs = np.zeros((N,1))
	for j in range(0,N):

		#align each to reference binning
		ref_binning = np.arange(0,1.1,0.01)
		binInd = 0
		for i in range(0,len(ref_binning)):

			#find appropriate bin to draw
			while((binInd < (X.shape[1]-1)) and (ref_binning[i] > X[j,binInd])):
				binInd = binInd + 1
			AUCs[j] = AUCs[j] + Y[j,binInd]
		AUCs[j] = AUCs[j] / len(ref_binning)
	return AUCs