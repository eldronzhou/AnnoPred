import numpy as np
import h5py
import logging
import pdb
import pandas as pd
#from memory_profiler import profile
from concurrent.futures import ProcessPoolExecutor, wait
from multiprocessing import cpu_count

def inHelp(left,right,rightOrd,repRange,core):
    leftOrd=np.argsort(left).astype(int)
    left=left[leftOrd]
    
    ss=np.searchsorted(right,left)
    loc=np.where(left==right[np.minimum(ss,len(right)-1)])[0].astype(int)

    return(repRange[leftOrd[loc]],rightOrd[np.unique(ss[loc])],core)
    
#@profile
def get_1000G_snps(sumstats, out_file):
    sf=pd.read_csv(sumstats,header=0,dtype=str,sep=' ').values
    h5f = h5py.File('ref/AnnotMatrix/1000G_SNP_info.h5','r')
    rf = h5f['snp_chr'][:]
    h5f.close()

    rfOrd=np.argsort(rf[:,2])
    rfSorted=rf[rfOrd,2].astype(str)
    numCores=cpu_count()
    
    futures=[]
    ind1=[None]*numCores
    ind2=[None]*numCores
    with ProcessPoolExecutor(numCores) as executor:
        for core in range(numCores):
            blockLen=int(np.ceil(sf.shape[0]/numCores))
            repRange=np.array(range(core*blockLen,min(sf.shape[0],(core+1)*blockLen))).astype(int)
            #inHelp(sf[repRange,1].astype(str),rfSorted,rfOrd)
            futures+=[executor.submit(inHelp,sf[repRange,1].astype(str),rfSorted,rfOrd,repRange,core)]

        for f in wait(futures)[0]:
            core=f.result()[2]
            ind1[core]=f.result()[0].reshape(-1,1)
            ind2[core]=f.result()[1].reshape(-1,1)
        
        ind1=np.concatenate(ind1,axis=0).flatten()
        ind2=np.concatenate(ind2,axis=0).flatten()
            
    #ind1 = np.in1d(sf[:,1],rf[:,2])
    #ind2 = np.in1d(rf[:,2],sf[:,1])
    sf1 = sf[ind1]
    rf1 = rf[ind2]
    ### check order ###

    if sum(sf1[:,1]==rf1[:,2].astype(str))==len(rf1[:,2]):
        logging.debug('Good!')
    else:
        logging.debug('Shit happens, sorting sf1 to have the same order as rf1')
        O1 = np.argsort(sf1[:,1])
        O2 = np.argsort(rf1[:,2])
        O3 = np.argsort(O2)
        sf1 = sf1[O1][O3]
    out=pd.DataFrame(np.concatenate([sf1[:,0:4],rf1[:,1:2].astype(str),sf1[:,5:7]],axis=1),columns=['hg19chrc', 'snpid', 'a1', 'a2', 'bp', 'or', 'p'])

    out.to_csv(out_file,sep=' ',header=True,index=False)

