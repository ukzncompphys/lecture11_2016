import numpy as np
from matplotlib import pyplot as plt
class Data:
    def __init__(self,x,sigma,x0,amp):
        self.x=x.copy()
        self.y=amp*np.exp(-0.5*(x-x0)**2/sigma**2)
        
    def add_noise(self,noise_amp=1.0):
        self.noise=np.random.randn(self.y.size)*noise_amp
        self.noise=np.reshape(self.noise,self.y.shape)
        self.y+=self.noise
        self.noise_amp=noise_amp
    def chisq(self,params):
        x0=params[0]
        sigma=params[1]
        amp=params[2]
        
        pred=amp*np.exp(-0.5*(self.x-x0)**2/sigma**2)
        delta=self.y-pred
        chisq=delta**2/self.noise_amp**2
        return np.sum(chisq)

def mcmc_driver(p_start,data,p_sig,nstep=10000):
    p_cur=p_start.copy()
    chi_cur=data.chisq(p_cur)
    npar=len(p_start)

    samps=np.zeros([nstep,npar])
    samps[0,:]=p_start
    
    big_chisq=np.zeros(nstep)
    big_chisq[0]=chi_cur

    for i in range(1,nstep):
        p_trial=p_cur+p_sig*np.random.randn(npar)
        chi_trial=data.chisq(p_trial)
        delta_chi=chi_trial-chi_cur
        accept_prob=np.exp(-0.5*delta_chi)    
        accept=(np.random.rand()<accept_prob)
        
        if accept:
            p_cur=p_trial
            chi_cur=chi_trial
        samps[i,:]=p_cur
        big_chisq[i]=chi_cur
    return samps,big_chisq

if __name__=='__main__':
    x=np.linspace(-5,5,1000)
    sigma=2.0
    x0=0.0
    amp=1.0
    mydat=Data(x,sigma,x0,amp)
    mydat.add_noise()
    
    plt.ion()
    plt.clf()
    #plt.plot(mydat.x,mydat.y)
    #chi1=mydat.chisq([-3,0.5,5])
    #chi2=mydat.chisq([x0,sigma,amp])
    #print chi1,chi2,chi1-chi2
    
    p_start=np.asarray([x0,sigma,amp+0.5])
    p_sig=np.asarray([0.02,0.02,0.1])/5

    samps,chisq=mcmc_driver(p_start,mydat,p_sig,100000)
    p_sig_new=np.std(samps,axis=0)
    samps,chisq=mcmc_driver(p_start,mydat,p_sig_new,10000)
