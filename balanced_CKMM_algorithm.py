import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

##import package
import numpy as np
import multiprocessing
import itertools as it
from functools import partial
from scipy.integrate import quad_vec
from scipy.integrate import quad
from scipy.stats import norm
from scipy.special import logsumexp
from sklearn.cluster import KMeans
import time
import multiprocessing.pool

###functions used in CKMM
def upper_bound(d,g,data,posterior):
    N = data.shape[2]
    Time = data.shape[1]
    data_vector = data[d,0:,0:].flatten(order='F')
    DistT = np.zeros(shape=(int(N * Time), int(N * Time)))
    post_vector = np.array([np.repeat(posterior[0:,g], repeats=Time)])
    post_matrix = (np.repeat(post_vector, (N * Time), axis=0)).T
    for i in range(N * Time):
        DistT[0:,i] = (data_vector - data_vector[i])**2
    post_matrix[np.where(DistT == 0)] =0
    bandwidth = 0.5*np.sum(post_vector[0,0:]*np.sum(post_matrix*DistT,axis=0))/(np.sum(post_vector[0,0:]*np.sum(post_matrix,axis=0))+np.sum(post_vector**2))
    return np.sqrt(bandwidth)


def kernel_cdf(x,t,dataT,bandwidth,posterior,d,g):
    Time = dataT.shape[1]
    dist = x-dataT[d, t, 0:]
    prob = np.repeat(np.array([posterior[0:, g]]), Time, axis=0)
    kernel_cdf = np.sum(prob * norm.cdf(dist, loc=0, scale=bandwidth)) / np.sum(prob)
    return kernel_cdf

def quantile_functionsub(dataT, posterior, bandwidth,g,d):
    prob_matrix = np.array([kernel_cdf(x=dataT[d, j, m], dataT=dataT, bandwidth=bandwidth, posterior=posterior, d=d, g=g,t=j) for m in range(np.shape(dataT)[2]) for j in range(np.shape(dataT)[1])]).reshape((np.shape(dataT)[2], np.shape(dataT)[1]), order="C")
    quantile_matrix = norm.ppf(prob_matrix, loc=0, scale=1)
    quantile_matrix[np.where(quantile_matrix==np.inf)] =np.nan
    quantile_matrix[np.where(np.isnan(quantile_matrix)==True)] = np.nanmax(quantile_matrix)
    quantile_matrix[np.where(quantile_matrix==-np.inf)] =np.nan
    quantile_matrix[np.where(np.isnan(quantile_matrix)==True)] = np.nanmin(quantile_matrix)
    return quantile_matrix

def quantile_func(dataT,posterior,bandwidth):
    quantile_data = np.zeros(shape=(np.shape(dataT)[2],np.shape(dataT)[0]*np.shape(dataT)[1],np.shape(posterior)[1]))
    for i in range(np.shape(posterior)[1]):
        for j in range(np.shape(dataT)[0]):
            quantile_data[0:,(j*np.shape(dataT)[1]):(j+1)*(np.shape(dataT)[1]),i] = quantile_functionsub(dataT=dataT,posterior=posterior,bandwidth=bandwidth[j,i],g=i,d=j)
    return quantile_data

def permutation (feature, Time):
    P = np.matrix(np.zeros(shape=((feature*Time),(feature*Time))))
    for m in range(Time):
        for d in range(feature):
            P[feature*m+d, Time*d+m] = 1
    return P

def eigenmatrix (Time,feature):
    subeigen_matrix = np.zeros(shape=(Time,Time),dtype=np.complex_)
    for k in range(Time):
        for i in range(Time):
            subeigen_matrix[i,k] = (1/np.sqrt(Time))*np.exp(-(k)*((i)/Time)*np.pi*2j)
    eigen_matrix = np.matrix(np.zeros(shape=(feature*Time, feature*Time),dtype=np.complex_))
    for k in range(feature):
        eigen_matrix[Time*k:Time*(k+1),Time*k:Time*(k+1)] = subeigen_matrix
    return eigen_matrix

def R_estimate(dataT, posterior, dataquantil):
    feature = dataT.shape[0]
    Time = dataT.shape[1]
    N = dataT.shape[2]
    G = np.shape(posterior)[1]
    # dataquantil = quantile(data=dataT_n, label=label, G=G)
    # dataquantil = quantiledata
    perm = permutation(feature=feature, Time=Time)
    eigen_matrix = eigenmatrix(Time=Time, feature=feature)
    W = np.dot(perm, eigen_matrix.H)
    R_matrix = np.zeros(shape=((Time * feature), (Time * feature), G))
    for i in range(G):
        Ct = np.zeros(shape=((feature * Time), (feature * Time)), dtype=np.complex_)
        for k in range(Time):
            for m in range(N):
                Ct[(feature * k):(feature * (k + 1)), (feature * k):(feature * (k + 1))] = Ct[(feature * k):(feature * (k + 1)), (feature * k):(feature * (k + 1))] + (
                        posterior[m, i] / np.sum(posterior[0:, i])) * np.dot(np.dot(np.matrix(W[feature * k:feature * (k + 1), 0:]),np.matrix(dataquantil[m, 0:, i]).reshape((feature * Time), 1)),np.dot(np.matrix(W[feature * k:feature * (k + 1), 0:]),np.matrix(dataquantil[m, 0:, i]).reshape((feature * Time), 1)).H)
        #R_matrix[0:, 0:, i] = np.real(np.dot(np.dot(eigen_matrix, np.dot(np.dot(perm.T, Ct), perm)), eigen_matrix.H))
        R_matrix[0:,0:,i] = Ct
    return R_matrix

def marginal_loglik(x,g,d,dataT,bandwidth,posterior):
    data = dataT[d,0:,0:].flatten(order="F")
    weights = np.repeat(posterior[0:,g],np.shape(dataT)[1])
    #est_log = np.dot(weights,norm.pdf(data-x,loc=0,scale=bandwidth))/np.sum(weights)
    est_dens = np.dot(weights,norm.pdf(data-x,loc=0,scale=bandwidth))
    est_log= logsumexp(-(data-x)**2/(2*bandwidth**2)+np.log(weights))
    #numerator = xlogy(est_dens,est_log)
    numerator = est_dens*est_log
    return numerator



def expected_loglik(g,d,dataT,bandwidth,fixed_bandwidth,posterior,C_est,n_limit):
    N = dataT.shape[2]
    Time = dataT.shape[1]
    feature = dataT.shape[0]
    quantile_data = np.zeros(shape=(N, Time*feature))
    for i in range(feature):
        if (i==d):
            quantile_data[0:,Time*i:(i+1)*Time] = quantile_functionsub(dataT=dataT, posterior = posterior, bandwidth = bandwidth, g=g, d=d)
        else:
            quantile_data[0:,Time*i:(i+1)*Time] = quantile_functionsub(dataT=dataT, posterior = posterior, bandwidth = fixed_bandwidth[i,g], g=g, d=i)
    perm = permutation(feature=feature, Time=Time)
    eigen_matrix = eigenmatrix(Time=Time, feature=feature)
    W = np.dot(perm, eigen_matrix.H)
    expected_logliksub1 = 0
    for i in range(N):
        expected_logliksub1 = expected_logliksub1-0.5*posterior[i,g]*np.real(np.dot(np.dot(np.dot(np.matrix(W[0:, 0:]),np.matrix(quantile_data[i, 0:].reshape((feature *Time), 1))).H,np.linalg.inv(C_est[0:,0:,g])-np.identity(feature*Time)),np.dot(np.matrix(W[0:, 0:]),np.matrix(quantile_data[i, 0:]).reshape((feature * Time), 1))))
    expected_logliksub2 = quad(marginal_loglik,-100,100,args=(g,d,dataT,bandwidth,posterior),points=dataT[d,0:,0:].flatten(order="F"),limit=n_limit)[0]-0.5*np.log(2*np.pi*bandwidth**2)*Time*np.sum(posterior[0:,g])-Time*np.sum(posterior[0:,g])*np.log(Time*np.sum(posterior[0:,g]))
    expected_loglikesub = expected_logliksub2+expected_logliksub1[0,0]
    return expected_loglikesub, expected_logliksub1[0,0], expected_logliksub2



def bandwidth_func(g,d,bandwidth_value,dataT,posterior,C_est,maxiter,n_limit):
    h = [bandwidth_value[d,g]]
    h.append(h[0]-0.001)
    loglik = [expected_loglik(g=g,d=d,dataT=dataT,bandwidth=h[0],fixed_bandwidth=bandwidth_value,posterior=posterior,C_est=C_est,n_limit=n_limit)]
    loglik.append(expected_loglik(g=g,d=d,dataT=dataT,bandwidth=h[1],fixed_bandwidth=bandwidth_value,posterior=posterior,C_est=C_est,n_limit=n_limit))
    if (h[0] < 0.1) and (1-loglik[1][0]/loglik[0][0] <= 1e-03):
        h_dg = h[0]
    else:
        if (h[1]+0.001*(loglik[1][0]-loglik[0][0])/(h[1]-h[0])>0.1):
            h.append(h[1]+0.001*(loglik[1][0]-loglik[0][0])/(h[1]-h[0]))
        elif (h[1]+0.0001*(loglik[1][0]-loglik[0][0])/(h[1]-h[0])>0.1):
            h.append(h[1]+0.0001*(loglik[1][0]-loglik[0][0])/(h[1]-h[0]))
        else:
            h.append(h[1]-0.001)
        loglik.append(expected_loglik(g=g,d=d,dataT=dataT,bandwidth=h[2],fixed_bandwidth=bandwidth_value,posterior=posterior,C_est=C_est,n_limit=n_limit))
        i=2
        while (h[i] > 0) and (1-loglik[i][0]/loglik[i-1][0] >= 1e-03) and (i <= maxiter) :
            if (h[i]+0.001*(loglik[i][0]-loglik[i-1][0])/(h[i]-h[i-1])> 0.1):
                h.append(h[i]+0.001*(loglik[i][0]-loglik[i-1][0])/(h[i]-h[i-1]))
            elif (h[i]+0.0001*(loglik[i][0]-loglik[i-1][0])/(h[i]-h[i-1])>0.1):
                h.append(h[i]+0.0001*(loglik[i][0]-loglik[i-1][0])/(h[i]-h[i-1]))
            else:
                h.append(h[i]-0.001)
            i=i+1
            loglik.append(expected_loglik(g=g,d=d,dataT=dataT,bandwidth=h[i],fixed_bandwidth=bandwidth_value,posterior=posterior,C_est=C_est,n_limit=n_limit))
        if (h[np.size(h)-1]<0) or (float("{:.4f}".format(h[np.size(h)-1]))-float("{:.4f}".format(h[np.size(h)-2]))==0.001):
            h_dg = h[np.size(h)-2]
        else:
            h_dg = h[np.size(h)-1]
    return h_dg



def kernel_pdfsmooth(x,data_vector,dataT,bandwidth,posterior,feature,g):
    data = dataT[feature,0:,0:].flatten(order="F")
    weights = np.repeat(posterior[0:,g],np.shape(dataT)[1])
    est_log = logsumexp(-(data-x)**2/(2*bandwidth**2)+np.log(weights))
    a= norm.pdf(x-data_vector,scale=bandwidth,loc=0)
    smooth = a*est_log
    return smooth


def marginalloglik_value(dataT,bandwidth,posterior,n_limit):
    Time = np.shape(dataT)[1]
    feature = np.shape(dataT)[0]
    G= np.shape(posterior)[1]
    N=np.shape(dataT)[2]
    value = [np.array(quad_vec(partial(kernel_pdfsmooth,data_vector = dataT[d,0:,0:].flatten(order="F"),dataT=dataT,bandwidth=bandwidth[d,g],posterior=posterior,feature=d,g=g),-100,100,points=dataT[d,0:,0:].flatten(order="F"),limit=n_limit)[0]).reshape((Time,N),order="F") for d in range(feature) for g in range(G)]
    y=np.zeros(shape=(feature,G,N))
    for i in range(feature):
        for j in range(G):
            y[i,j,0:] = (np.sum(value[G*i+j],axis=0))-0.5*Time*np.log(2*np.pi*bandwidth[i,j]**2)-Time*np.log(Time*np.sum(posterior[0:,j]))
    return y




def E_step(dataT,marginallog,posterior,C_est,quantile,prior):
    N=np.shape(dataT)[2]
    G= np.shape(posterior)[1]
    feature = np.shape(dataT)[0]
    Time = np.shape(dataT)[1]
    label_newsub = np.zeros(shape=(N,G))
    jointlog_value = np.zeros(shape=(N,G))
    perm = permutation(feature=feature, Time=Time)
    eigen_matrix = eigenmatrix(Time=Time, feature=feature)
    W = np.dot(perm, eigen_matrix.H)
    for g in range(G):
        expected_logliksub1 = np.zeros(N)
        det_joint = 0
        for i in range(N):
            expected_logliksub1[i] = -0.5*np.real(np.dot(np.dot(np.dot(np.matrix(W[0:, 0:]),np.matrix(quantile[i, 0:,g].reshape((feature *Time), 1))).H,np.linalg.inv(C_est[0:,0:,g])-np.identity(feature*Time)),np.dot(np.matrix(W[0:, 0:]),np.matrix(quantile[i, 0:,g]).reshape((feature * Time), 1))))
        for i in range(Time):
            det_joint = det_joint - 0.5*np.log(np.real(np.linalg.det(C_est[feature * i:feature * (i + 1),feature * i:feature * (i + 1),g])))
        jointlog_value[0:,g] = expected_logliksub1+det_joint
    x=(np.log(np.repeat(prior,N).reshape((N,G),order="F"))+np.sum(marginallog,axis=0).T+(jointlog_value))
    posterior_new = np.zeros(shape=(N,G))
    for i in range(N):
        for j in range(G):
            posterior_new[i,j] = 1/np.sum(np.exp(x[i,0:]-x[i,j]))
    posterior_new[np.where(posterior_new==0)] = 1e-50
    for i in range(N):
        label_newsub[i,np.where(posterior_new[i,0:]==np.max(posterior_new[i,0:]))]=1
    label_new = np.dot(label_newsub, np.matrix(range(G)).reshape((G,1),order='F'))
    obs_loglik=np.sum(posterior_new*x-posterior_new*np.log(posterior_new))
    prior_new = np.mean(posterior_new,axis=0)
    return label_new, obs_loglik, posterior_new,prior_new


def M_step (dataT,posterior,bandwidth,C_est,n_limit):
    G= np.shape(posterior)[1]
    feature = np.shape(dataT)[0]
    #estimatie bandwidth
    pool = multiprocessing.Pool(processes=G*feature)
    h_estv = pool.starmap(partial(bandwidth_func, bandwidth_value=bandwidth,dataT=dataT, posterior=posterior,C_est = C_est, maxiter=100,n_limit = n_limit),it.product(range(G), range(feature)))
    pool.close()
    pool.join()
    h_est = np.array(h_estv).reshape((feature,G),order="F")
    #estimate quantile
    quantile_data=quantile_func(dataT=dataT,posterior=posterior,bandwidth=h_est)
    ##estimate R_matrix
    C_estnew = R_estimate(dataT=dataT, posterior=posterior,dataquantil=quantile_data)
    ###estimate prior
    prior = np.mean(posterior,axis=0)
    marginal = marginalloglik_value(dataT=dataT,bandwidth=h_est, posterior=posterior, n_limit=n_limit)
    return prior, C_estnew, h_est, quantile_data, marginal


def EM(dataT,marginal_int,C_int,h_int,posterior_int,prior_int,quantile_int,n_limit):
    start = time.perf_counter()
    E_stepdata= E_step(dataT=dataT, marginallog=marginal_int, posterior=posterior_int, C_est=C_int, quantile=quantile_int, prior=prior_int)
    M_stepdata = M_step (dataT=dataT,posterior=E_stepdata[3],bandwidth=h_int,C_est=C_int,n_limit=n_limit)
    log_like=E_stepdata[1]
    E_stepdata= E_step(dataT=dataT, marginallog=M_stepdata[4], posterior=E_stepdata[3], C_est=M_stepdata[1], quantile=M_stepdata[3], prior=M_stepdata[0])
    log_like = np.append(log_like, E_stepdata[1])
    while np.abs(log_like[int(len(log_like)-1)]-log_like[int(len(log_like)-2)])/ np.abs(log_like[int(len(log_like)-1)]) > 1e-5:
        M_stepdata = M_step(dataT=dataT,posterior=E_stepdata[3],bandwidth=M_stepdata[2],C_est=M_stepdata[1],n_limit=n_limit)
        E_stepdata= E_step(dataT=dataT, marginallog=M_stepdata[4], posterior=E_stepdata[3], C_est=M_stepdata[1], quantile=M_stepdata[3], prior=M_stepdata[0])
        log_like = np.append(log_like, E_stepdata[1])
    end = time.perf_counter()
    elapsed = end - start
    return log_like, E_stepdata[0],E_stepdata[3],M_stepdata[0],M_stepdata[1],M_stepdata[2],M_stepdata[4],elapsed

#####


def CKMM(data,num_cluster,num_feature,rand_seed):
    time_len = int(np.shape(data)[1]/num_feature)
    N = np.shape(data)[0]
    n_limit = time_len*N+50
    kmeans = KMeans(n_clusters=num_cluster, random_state=rand_seed).fit(data)
    label_est = kmeans.labels_
    dataT_n = np.zeros(shape=(num_feature, time_len, N))
    for i in range(np.shape(data)[0]):
        for f in range(num_feature):
            dataT_n[f, 0:, i] = data[i, int(f*time_len):int((f+1)*time_len)]
    feature = dataT_n.shape[0]
    G=num_cluster
    post_matrix = np.full((N, G),0.0001)
    for i in range(N):
        for g in range(G):
            post_matrix[np.where(label_est==g),g] = 1-0.0001*(G-1)
    prior_int = np.array([np.size(np.where(label_est==g))/N for g in range(G)])
    upper_est = np.array([upper_bound(d=i, g=j, data=dataT_n, posterior=post_matrix) for i in range(np.shape(dataT_n)[0]) for j in
                          range(np.shape(post_matrix)[1])]).reshape((np.shape(dataT_n)[0], np.shape(post_matrix)[1]), order="C")
    quantile_data = quantile_func(dataT=dataT_n, posterior=post_matrix, bandwidth=upper_est)
    C_est = R_estimate(dataT=dataT_n, posterior=post_matrix, dataquantil=quantile_data)
    pool = multiprocessing.Pool(processes=12)
    h_estv = pool.starmap(partial(bandwidth_func, bandwidth_value=upper_est, dataT=dataT_n, posterior=post_matrix, C_est=C_est, maxiter=100,n_limit=n_limit),
                          it.product(range(G), range(feature)))
    pool.close()
    pool.join()
    h_est = np.array(h_estv).reshape((feature, G), order="F")
    quantile_int = quantile_func(dataT=dataT_n, posterior=post_matrix, bandwidth=h_est)
    C_int = R_estimate(dataT=dataT_n, posterior=post_matrix, dataquantil=quantile_int)
    marginal = marginalloglik_value(dataT=dataT_n,bandwidth=h_est, posterior=post_matrix, n_limit=n_limit)
    em = EM(dataT= dataT_n, marginal_int = marginal, C_int = C_int, h_int=h_est,posterior_int=post_matrix,prior_int=prior_int, quantile_int = quantile_int,n_limit=n_limit)
    return em

