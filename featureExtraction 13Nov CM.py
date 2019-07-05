import struct
import datetime
import time
import numpy as np
import pandas as pd
import math
import allantools
from scipy.signal import find_peaks
from numpy import linalg as LA
from pywt import cwt
import os
from itertools import chain
from os.path import expanduser


def gatherInputFiles(directory):
    """
    Returns all files with 'gse2.SAC' in the provided directory 
    Can handle files nested in subdirectories
    Returns as the relative path from the cwd with the filename
    e.g. returns:
        /subdir1/file1.gse2
        /subdir1/file2.gse2
        /subdir2/file1.gse2
    """
    all_files = []

    for (dirpath, dirnames, filenames) in os.walk(directory):
        if len(filenames) == 0:
            continue
        rel_filepaths = [os.path.join(os.path.abspath(dirpath), name) for name in filenames]
        all_files.append(rel_filepaths)

    all_files = list(chain.from_iterable(all_files))
    sac_files = []
    other_files = []

    if len(all_files) == 0:
        print("no files found in directory provided")
        raise SystemExit()

    for f in all_files:
        if f[-17:-9] == 'gse2.SAC':
            sac_files.append(f)
        else:
            other_files.append(f)

    if len(sac_files) == 0:
        print("no SAC files found in directory provided")
        raise SystemExit()

    return sac_files


def groupStations(directory):
    """
    Search a directory and arrange all found SAC files into station groups
    This is a 'dumb' method - replace with something better.
    """
    sac_files = gatherInputFiles(directory)
    sac_files.sort()
    
    if len(sac_files) % 3 != 0:
        print ("one station does not have 3 orientation files")
        
    stations = [sac_files[i:i+3] for i in range(0, len(sac_files), 3)]

    return stations




def readStationHeader(filename):
    headerBytes = 632
    L = 4
    
    with open(filename, "rb") as f:
        header = f.read(headerBytes)

        delta = int(struct.unpack("<f", header[0:0+L])[0]*1000)
        
        nzyear = str(struct.unpack("<i", header[280:280+L])[0]).zfill(4)
        nzjday = str(struct.unpack("<i", header[284:284+L])[0]).zfill(3)
        nzhour = str(struct.unpack("<i", header[288:288+L])[0]).zfill(2)
        nzmin = str(struct.unpack("<i", header[292:292+L])[0]).zfill(2)
        nzsec = str(struct.unpack("<i", header[296:296+L])[0]).zfill(2)
        nzmsec = str(struct.unpack("<i", header[300:300+L])[0]).zfill(3) + '000'
        
        npts = struct.unpack("<i", header[316:316+L])[0]
        
        
        start_timestamp = nzyear + '-' + nzjday + ' ' + nzhour + ':' + nzmin + ':' + nzsec + '.' + nzmsec
        start_time = time.mktime(datetime.datetime.strptime(start_timestamp, "%Y-%j %H:%M:%S.%f").timetuple())
        
        delta_series = map(lambda x: x*delta, range(0,npts))
        time_series = map(lambda x: x+start_time, delta_series)
    
    return delta_series, time_series
    

def readData(filename):
    data = []
    with open(filename, "rb") as f:
        f.seek(632)
        byte = f.read(4)
        
        while byte != "":
            datum = struct.unpack("<f", byte)[0]
            data.append(datum)
            byte = f.read(4)

    return data


def packageAsDataFrame(delta_series, time_series, data_z, data_n, data_e):
    data = np.transpose(np.stack([
        np.asarray(delta_series),
        np.asarray(time_series),
        np.asarray(data_z),
        np.asarray(data_n),
        np.asarray(data_e)]
    ))
    dataframe = pd.DataFrame(data, columns = ['delta', 'timestamps', 'z', 'n', 'e'])
    return dataframe


def createStationDataFrame(z_file, n_file, e_file):
    delta_series, time_series = readStationHeader(z_file)
    data_z = readData(z_file)
    data_n = readData(n_file)
    data_e = readData(e_file)
    return packageAsDataFrame(delta_series, time_series, data_z, data_n, data_e)




def calculateResiduals(x, y, n):    
    coeff_scaled = np.polyfit(x, y, n)
    f_y = np.polyval(coeff_scaled, x)
    residuals = y - f_y
    return residuals


def computeAllanDeviation(rate, data): # make data the waveform
    alpha = np.logspace(-2, 2, 200)
    
    ##########################################################
    
    # Matlab selects taus from alpha by the equation below:    
    # tau = tau(find(tau >= tmstep & tau <= halftime));
    # It selected 164 tau values in our test
    
    # AllanTools selects 110 tau values in the same test
    
    # the adev returned seems to have the same values, to begin with,
    # but there is randomized repetition of the values at the start of the
    # Matlab version.
    # In the first test, the end result of the find_peaks was the same,
    # but further investigation is necessary
    
    ##########################################################
    
    (taus_used, adev, adeverror, adev_n) = allantools.adev(
        data=data, 
        data_type='freq', # 'phase' or 'freq' ?
        rate=rate, 
        taus=alpha)
    
    return adev


def calculateAllanDev(rate, data):
    retval = computeAllanDeviation(rate, data)
    peak_indices = find_peaks(retval)[0]
    
    maxAllanIndex = peak_indices[0]
    maxAllanValue = retval[maxAllanIndex]
    
    return maxAllanIndex, maxAllanValue


def stockwellTransformation(residuals):
    N = residuals.size
    N2 = int(math.floor(N/2.0))
    
    if float(N) % 2 == 0:
        j = 0
    else:
        j = 1

    f = np.true_divide(np.concatenate((np.arange(0, N2+1), np.arange(-N2+1-j, 0))), N)
    
    f_mapped = map(lambda x: x*2*math.pi, f)
    
    alpha = np.linspace(1, 1, N2)
    counts = np.arange(2, N2+1)

    residuals_fft = np.fft.fft(residuals, N, axis=0)

    residuals_rotated = np.array(map(lambda x: np.roll(residuals_fft, -(x-1)), counts))
    
    W = np.array(map(lambda x: np.multiply((1.0/(f[x-1]**alpha[x-1])), f_mapped), counts))

    # G = exp((-W.^2)/2);
    # this looks like the -W is squared before dividing by 2, but in Matlab
    # the minus sign is applied to the entire exponent
    #G = np.array(map(lambda x: np.exp(np.true_divide(np.square(-x), 2)), W))
    G = np.array(map(lambda x: np.exp(-np.true_divide(np.square(x), 2)), W))

    modst_input = map(lambda x: np.multiply(residuals_rotated[x], G[x]), range(0, counts.size))
    
    modst = np.array(map(lambda x: np.fft.ifft(x), modst_input))
    
    zeros = np.zeros(modst.shape[1]).reshape(1, modst.shape[1])
    modst_padded = np.concatenate((zeros, modst, zeros))
    
    
    return modst_padded


def prepareQfactorVariables(residuals, thresholding, coeff_thr, rate):
    modst = stockwellTransformation(residuals)
    
    matr = np.square(np.absolute(modst))

    if thresholding:
        matr1 = matr
        thr = coeff_thr*np.max(matr)
        matr1[matr1<thr] = 0

    freq = np.linspace(0, rate/2, modst.shape[0])
    
    return matr, matr1, freq


#-------------------------------------------
# Finding new I function when I>len(sign)/2
#-------------------------------------------
def find_newI(sign):
    signal=np.array(sign)
    peak_loc, _ = find_peaks(signal)
    peak_val = signal[peak_loc]
    
    # 1.3 is a magic number to catch the cases where
    # there is no peak in the signal, i.e. it is entirely
    # flat or globally increasing
    # we catch this later on to give Q a value of 0
    I = peak_loc[np.argmax(peak_val)] if len(peak_loc) > 0 else -1 

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
def Qfact_gl(freq, matr, rows):
    ##---> If the maximum is more than half the spectrum, the maximum is taken 
    I = np.array([find_newI(i[0: np.argmax(i)]) if np.argmax(i)>rows/2 else np.argmax(i) for i in matr.T])
    
    ##---> Left flexion
    locs_left = np.array([find_peaks(np.diff(col))[0] for col in matr.T])
    #if len(I)=len(locs_left) - stop the function
    eps1=[(I[i]-locs_left[i]).tolist() for i in xrange(0, len(locs_left))] 
    minimum_posL=[min(eps1[i]) for i in xrange(0, len(eps1))] # add: condition of i>0
    flesso_sx=[eps1[i].index(minimum_posL[i]) for i in xrange(0, len(eps1))]
    ind_sx = [locs_left[index][flesso_sx[index]] for index in xrange(0, len(flesso_sx))]
    
    ##---> Right flexion
    locs_right = np.array([find_peaks(-np.diff(col))[0] for col in matr.T])
    eps2=[(-(I[i]-locs_left[i])).tolist() for i in xrange(0, len(locs_right))] 
    minimum_posR=[min(eps2[i]) for i in xrange(0, len(eps2))]
    flesso_dx=[eps2[i].index(minimum_posR[i]) for i in xrange(0, len(eps2))]
    ind_dx = [locs_right[index][flesso_dx[index]] for index in xrange(0, len(flesso_dx))]
    
    print (locs_left)
    
#    print locs_right
#    print locs_right.shape
    
    ##---> Q calculation
    divisor = np.array([freq[index] for index in ind_sx]) -np.array([freq[index] for index in ind_dx])
    dividend = np.array([freq[index] if index != -1 else 0 for index in I])
    Q = dividend/divisor
    
    return Q


def calculateQfactor(residuals, thresholding, coeff_thr, rate):
    matr, matr1, freq = prepareQfactorVariables(residuals, thresholding, coeff_thr, rate)
    ampl_thr = 100
    rows, cols = matr.shape
    
    # could apply MD5 hash to matrices to check they are the same in both python and matlab
    
    en = np.power(np.sum(matr1, axis=0), 2)
    (col,) = np.where(en>0)
    sp = matr[:,col]
    
    # ---> ampl_max. Vector's actual data shape: (148L,), (72L,), (61L,)
    ampl_max = matr.max(axis=0)
    
    # ---> fmax. Vector's shape: (4000 rows, 1 column)  
    index = np.argmax(sp, axis=0)
    fmax = np.concatenate([freq[index], np.zeros(cols-index.shape[0])])
    
    """
    # fmax is a vector that gets populated from (1:number of nonzero columns). 
    In matlab it is initialised as a vector with 4000 zeros (num of columns in matr)
    And then a for is used:
          for i = 1:numel(col)%nc
  
            sp = matr(:,col(i));
            [ampl_max(i), index] = max(sp);
            fmax(i) = freq(index);
            
    Meaning that for the first file fmax(1)=freq(513), 
    i.e. using the row where the max is in the matr column 1858 (row 513 is where the max is)
    This means, fmax is actually a vector like this: [8, 8.2,..., 3.43, 0, 0, 0, ..., 0], ie it is filled with zeros in the end. 
    Not sure if this is correct or the values should match the original column where the max was found. ie 
    fmax[0, 0, ..., 8, ... ] 
    fmax_columns[0, 1, ..., 1858, ... ] 
    the concatanate used is just a fix just to match the matlab code
    """
    # ---> E. Vector's shape: (4000 rows, 1 column)     
    sp_with_non_threshold_zeroed = np.where(
            (ampl_max >= ampl_thr) & (en>0), 
            matr, 
            np.zeros((rows, 1))
    )
    E = np.sum(sp_with_non_threshold_zeroed, axis=0)
    
    # ---> Q. Vector's shape: (4000 rows, 1 column)  
    # Use Q_nan to replace some of the padding zeros with pseudo-nan/None values
    (subset_cols, ) = np.where((ampl_max >= ampl_thr) & (en>0))
    sp_subset = matr[:, subset_cols]
    Q_subset = Qfact_gl(freq, sp_subset, rows)
    Q = np.zeros(cols)
    Q[subset_cols] = Q_subset
        
    # Use Q_nan to replace some of the padding zeros with pseudo-nan/None values
#    Q_nan = col.shape - subset_cols.shape
#    Q = np.concatenate([Qfact_gl(freq, sp_subset, rows), np.zeros(cols - (subset_cols.shape[0] + Q_nan))])
    
    return Q, E, fmax


def polarAnalysis(X_res, Y_res, Z_res, tau):
        
    #Output
    N = int(math.floor(len(X_res)/tau))
    Rec = []
    P_azimuth = [] 
    phi = []
    Dip = []
    
    for ii in range(1,N+1):
    
        #It takes data in groups of length "tau" e.g. if our data has length 200 and tau=20, N=10 and we take batches of data with length 20 each time
        ind_start = tau*(ii-1) 
        ind_end = tau*ii
        
        #matrix, numer of rows equals v length (tau), and 3 columns (X,Y,Z) - it basically creates a matrix with the batch of data we're looking at (length tau) and each column has the X,Y and Z values
        temp=np.matrix([X_res[ind_start:ind_end],Y_res[ind_start:ind_end],Z_res[ind_start:ind_end]])#.getT()
        
        #covariance matrix
        s=np.cov(temp)
        
        #Eigen values and vectors
        D, V = LA.eig(s)
        
        L1 = D[0]
        L2 = D[1]
        L3 = D[2]
        U1 = V[:,0]
        U2 = V[:,1]
        U3 = V[:,2]
        Rec.append(1-(L2+L1)/L3)
        
        #direction cosines
        r1 = math.sqrt(pow(U1[0],2)+pow(U1[1],2)+pow(U1[2],2))
        u31 = U1[2]/r1
            
        r2 = math.sqrt(pow(U2[0],2)+pow(U2[1],2)+pow(U2[2],2))
        u32 = U2[2]/r2
        
        r3 = math.sqrt(pow(U3[0],2)+pow(U3[1],2)+pow(U3[2],2))
        u13 = U3[0]/r3
        u23 = U3[1]/r3
        u33 = U3[2]/r3
        
        #Azimuth of propagation
        P_azimuth.append(math.atan(u13*np.sign(u33)/u23*np.sign(u33)))
        #Apparent vertical incidence anglev
        phi.append(math.acos(abs(u33)))
        #Dip of the direction of maximum polarization
        Dip.append(math.atan(u33/math.sqrt(pow(u32,2)+pow(u31,2))))

    return Rec, P_azimuth, phi, Dip


def microseismicAnalysis(residuals, thresholding, rate, coeff_thr):
    Q, E, fmax = calculateQfactor(residuals, thresholding, coeff_thr, rate)

    maxAllanIndex, maxAllanValue = calculateAllanDev(rate, residuals)
    
    sig = np.transpose(residuals)
    Wavelet = cwt(sig, np.arange(1, 21), 'morl')
    scale_max = np.max(Wavelet[0])
    
    return [np.mean(Q), np.std(Q), np.mean(E), np.std(E), np.mean(fmax), np.std(fmax), np.max(sig), scale_max, maxAllanValue, maxAllanIndex]


def stationAnalysis(station):
    polynomial_degree = 6
    thresholding = True
    coeff_thr = 0.05
    
    date = station[0][-42:-32]
    event = station[0][-31:-27]
    stationNumber = station[0][-23:-20]
    stationName = station[0].split(".")[-2]
    
    data = createStationDataFrame(station[0], station[1], station[2])
    
    delta = float(data['delta'].iloc[[2]]) - float(data['delta'].iloc[[1]])
    rate = 1/(delta/1000)
    
    data['z_residuals'] = calculateResiduals(data['delta'].values, data['z'].values, polynomial_degree)
#    data['n_residuals'] = calculateResiduals(data['delta'].values, data['n'].values, polynomial_degree)
#    data['e_residuals'] = calculateResiduals(data['delta'].values, data['e'].values, polynomial_degree)
    
    microseismicAttributes_z = microseismicAnalysis(data['z_residuals'].values, thresholding, rate, coeff_thr)
#    microseismicAttributes_n = microseismicAnalysis(data['n_residuals'].values, thresholding, rate, coeff_thr)
#    microseismicAttributes_e = microseismicAnalysis(data['e_residuals'].values, thresholding, rate, coeff_thr)
    
    tau = 50
    Rec, P_azimuth, phi, Dip = polarAnalysis(data['z_residuals'].values, data['n_residuals'].values, data['e_residuals'].values, tau)
    polarAttributes = [np.mean(Rec), np.mean(P_azimuth), np.mean(phi), np.mean(Dip)]
    
    # Combine microseismic and polar attributes into output dataframe
    
    attributes_z = [date, event, stationNumber, stationName, 'Z'] + microseismicAttributes_z + polarAttributes  
#    attributes_n = [date, event, stationNumber, stationName, 'N'] + microseismicAttributes_n + polarAttributes
#    attributes_e = [date, event, stationNumber, stationName, 'E'] + microseismicAttributes_e + polarAttributes
    
    attributes_data = [attributes_z, attributes_n, attributes_e]
    attributes_cols = ['date', 'event', 'stationNumber', 'stationName', 'orientation', 'meanQ', 'stdQ', 'meanE', 'stdE', 'meanFmax', 'stdFmax', 'maxSig', 'scaleMax', 'maxAllanValue', 'maxAllanIndex', 'meanRec', 'meanPazimuth', 'meanPhi', 'meanDip']
    station_df = pd.DataFrame(attributes_data, columns=attributes_cols)
    
#    picklePath = "./pickle/" + date + "_" + event + "_" + stationName + ".pkl"
#    station_df.to_pickle(picklePath)

    return station_df
    

#########################################################################
# Call Code
#########################################################################


HOME =  expanduser("~").replace("\\", "/")
DIRECTORY = "{home}/Documents/E/1/SAC/1030/".format(home=HOME)
stations = groupStations(DIRECTORY)


#from multiprocessing import Pool
#
#pool = Pool(4)
#stationDataframes = pool.map(stationAnalysis, stations)
#attributesDataframe = pd.concat(stationDataframes)


#stationDataframes = map(lambda x: stationAnalysis(x), stations)  
#attributesDataframe = pd.concat(stationDataframes)

stationDataframes = stationAnalysis(stations[0])

print (stationDataframes)
print(stationDataframes.to_string())
