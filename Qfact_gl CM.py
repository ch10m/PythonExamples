import numpy as np
from scipy.signal import find_peaks

#---> for testing
#sign= [0.3, 0.2, 0.5, 0.1, 0.8, 0.2, 0.9, 0.1, 0.1]
sign=[895, 967, 179, 105, 223, 504, 984, 625, 872, 123, 997, 982, 44]
freq=np.array([  0, 3.47, 6.94, 10.41, 13.88, 17.36, 20.83, 24.30, 27.77, 31.25])
print freq.shape
#matr=np.random.randint(1000, size=(10,10)) #np.asarray(matr) is matr 

matr=np.array([[494, 0, 518, 152,  76,   0,  66, 631,   0, 258],
       [213,  0, 444, 252, 975, 119, 488, 713, 980, 971],
       [446, 0, 857, 647, 525, 757, 701, 731, 204, 380],
       [  0, 0, 512, 857, 157, 956, 478, 976, 193, 559],
       [703, 0, 461, 684, 402, 176, 402, 543, 143, 728],
       [272, 0, 505, 567, 899,  84, 986, 940, 583, 241],
       [218, 0, 257, 969, 107, 987,   0, 895, 807, 575],
       [216, 0, 954, 899, 873, 667, 672, 635, 183, 313],
       [ 71, 0, 458, 416, 193, 860, 720, 801, 991, 722],
       [833, 0, 582, 353, 805, 104, 613, 838, 577,  99]])
print matr.shape

matr1 = matr
coeff_thr = 0.05
thr = coeff_thr*np.max(matr)
matr1[matr1<thr] = 0

ampl_thr = 100


#-------------------------------------------
# Finding new I function when I>len(sign)/2
#-------------------------------------------
def find_newI(sign):
    signal=np.array(sign)
    peak_loc, _ = find_peaks(signal)
    peak_val = signal[peak_loc] 
    
    ##---> If there are no peaks (peak_loc=[]) then take the index of the max
    I = peak_loc[np.argmax(peak_val)] #if len(peak_loc) > 0 else np.argmax(signal)

    return I

# 0 position in the array is ignored, does Matlab does the same?
#signal=np.array([734, 438, 168, 496, 455, 507, 385])

#-------------------------------------------
# Qfact_gl function - notes
#-------------------------------------------
    
##---> Left flexion and right flexion
# check data dimension just in case? if len(I)=len(locs_right) else print ("wrong dimension, check data")
# minimum_posL=min(i for i in eps1 if i >= 0) # ---> need to ask to client about min being = 0 and what if there are all <0?
# flesso_sx=eps1.index(minimum_posL)          # ---> in matlab they seem to be taking the row, is it a row vector?

#-------------------------------------------
# Qfact_gl function - matrix form
#-------------------------------------------
def Qfact_gl(freq,matr,rows):
    ##---> If the maximum is more than half the spectrum, the maximum is taken 
    I = np.array([find_newI(i[0: np.argmax(i)]) if np.argmax(i)>rows/2 else np.argmax(i) for i in matr.T])
    
    ##---> Left flexion
    locs_left = np.array([find_peaks(np.diff(col))[0] for col in matr.T])
    eps1=[(I[i]-locs_left[i]).tolist() for i in xrange(0, len(locs_left))] 
    minimum_posL=[min(eps1[i]) for i in xrange(0, len(eps1))] 
    flesso_sx=[eps1[i].index(minimum_posL[i]) for i in xrange(0, len(eps1))]
    ind_sx = [locs_left[index][flesso_sx[index]] for index in xrange(0, len(flesso_sx))]
    
    ##---> Right flexion
    locs_right = np.array([find_peaks(-np.diff(col))[0] for col in matr.T])
    eps2=[(-(I[i]-locs_left[i])).tolist() for i in xrange(0, len(locs_right))] 
    minimum_posR=[min(eps2[i]) for i in xrange(0, len(eps2))]
    flesso_dx=[eps2[i].index(minimum_posR[i]) for i in xrange(0, len(eps2))]
    ind_dx = [locs_right[index][flesso_dx[index]] for index in xrange(0, len(flesso_dx))]
    
    print locs_left.shape
    print locs_right.shape
    print ind_sx
    print ind_dx
    
    ##---> Q calculation
    divisor = np.array([freq[index] for index in ind_sx]) -np.array([freq[index] for index in ind_dx])
    dividend = np.array([freq[index] for index in I])
    Q = dividend/divisor
    
    return Q

#-------------------------------------------
# calling Qfact_gl function - matrix form
#-------------------------------------------

en=np.power(matr1.sum(axis=0),2).tolist()
col=[en.index(i) for i in en if i> 0]
matr_col=np.array([matr[:,i] for i in col]).T
(rows, cols) = matr_col.shape

ampl_max = matr_col.max(axis=0)
index = np.argmax(matr_col,axis=0)
fmax = [freq[i] for i in index]

E = np.array(np.zeros((len(col))))
#E[ampl_max>=ampl_thr] = matr_col.sum(axis=0)[ampl_max>=ampl_thr]
E = matr_col.sum(axis=0)[ampl_max>=ampl_thr]
Q = np.array(np.zeros((len(col))))
Q[ampl_max>=ampl_thr] =Qfact_gl(freq,matr_col,rows)[ampl_max>=ampl_thr]


sum_test=matr_col.sum(axis=0)
E = matr_col.sum(axis=0)[ampl_max>=ampl_thr]
print matr_col[:,0]


# =============================================================================
# test = matr_col[0]
# =============================================================================
test=np.array([ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 5.37048772e+08, 5.37890352e+08, 5.38694429e+08, 5.39461083e+08,
 5.40190404e+08, 5.40882493e+08, 5.41537464e+08, 5.42155439e+08,
 5.42736550e+08, 5.43280943e+08, 5.43788770e+08, 5.44260196e+08,
 5.44695392e+08, 5.45094543e+08, 5.45457841e+08, 5.45785487e+08,
 5.46077692e+08, 5.46334675e+08, 5.46556665e+08, 5.46743897e+08,
 5.46896617e+08, 5.47015078e+08, 5.47099539e+08, 5.47150269e+08])
    
peak_loc, _ = find_peaks(test)
peak_val = test[peak_loc] 
I = peak_loc[np.argmax(peak_val)] if len(peak_loc) > 0 else np.argmax(test)
test.shape

# =============================================================================
# E test
# =============================================================================
ampl_thr = 900
rows, cols = matr.shape
# could apply MD5 hash to matrices to check they are the same in both python and matlab

en = np.power(np.sum(matr1, axis=0), 2)
(col,) = np.where(en>0)
sp = matr[:,col]

# ---> ampl_max. Vectors shape: (148L,), (72L,), (61L,)
ampl_max = matr.max(axis=0)

sp_with_non_threshold_zeroed = np.where(
            (ampl_max >= ampl_thr) & (en>0), 
            matr, 
            np.zeros((rows, 1))
    )

E = np.sum(sp_with_non_threshold_zeroed, axis=0)

# =============================================================================
# Q test
# =============================================================================
en = np.power(np.sum(matr1, axis=0), 2)
(col,) = np.where(en>0)
#sp = matr[:,col]
#sp_ampl_max = sp.max(axis=0)


(subset_cols, ) = np.where((ampl_max >= ampl_thr) & (en>0))
sp_subset = matr[:, subset_cols]
Q_subset = Qfact_gl(freq, sp_subset, rows)
Q = np.zeros(cols)
Q[subset_cols] = Q_subset
Q.shape
    
Q = np.where(
    (ampl_max >= ampl_thr),
    Q, 
    "NA")

Q_nan = (col.shape)[0] - (subset_cols.shape)[0]
Q = np.concatenate([Qfact_gl(freq, sp_subset, rows), np.zeros(cols - (subset_cols.shape[0] + Q_nan))])