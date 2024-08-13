import numpy as np


#####four cluster three features
Time = 30
cor12 = 0
cor13 = 0
cor23 = 0
coef1_1 = (-1 + (1 - 4 * 0.25 ** 2) ** (1/2)) / (2 * 0.25) #autocorelation is -0.25
coef2_1 = (1 - (1 - 4 * 0.45 ** 2) ** (1/2)) / (2 * 0.45)  #autocorelation is 0.45
coef3_1 = (1 - (1 - 4 * 0.25 ** 2) ** (1/2)) / (2 * 0.25)  #autocorelation is 0.45
A = np.array([[coef1_1, 0.0, 0.0],
              [0.0, coef2_1,0.0],
              [0.0, 0.0,coef3_1] ])
errormatrix = np.array([[1, (cor12 * (1 + coef1_1 ** 2) ** (1/2) * (1 + coef2_1 ** 2) ** (1/2)) / (1 + coef1_1* coef2_1),
                         (cor13 * (1 + coef1_1 ** 2) ** (1/2) * (1 + coef3_1 ** 2) ** (1/2)) / (1 + coef1_1 * coef3_1)],
                        [(cor12 * (1 + coef1_1 ** 2) ** (1/2) * (1 + coef2_1 ** 2) ** (1/2)) / (1 + coef1_1 * coef2_1), 1, (cor23 * (1 + coef2_1 ** 2) ** (1/2) * (1 + coef3_1 ** 2) ** (1/2)) / (1 + coef3_1 * coef2_1)],
                        [(cor13 * (1 + coef1_1 ** 2) ** (1/2) * (1 + coef3_1 ** 2) ** (1/2)) / (1 + coef3_1 * coef1_1),
                         (cor23 * (1 + coef2_1 ** 2) ** (1/2) * (1 + coef3_1 ** 2) ** (1/2)) / (1 + coef3_1 * coef2_1),1]
                        ])
variance_1 = errormatrix + np.dot(np.dot(A, errormatrix), A.T)
variance_lag1 = np.dot(A, errormatrix)
autocovariance1_1 = np.array([[variance_1[0, 0], variance_lag1[0, 0]]])
autocovariance2_1 = np.array([[variance_1[1, 1], variance_lag1[1, 1]]])
autocovariance3_1 = np.array([[variance_1[2, 2], variance_lag1[2, 2]]])
autocovariance12_1 = np.array([[variance_lag1[0, 1], variance_1[0, 1], variance_lag1[1, 0]]])
autocovariance13_1 = np.array([[variance_lag1[0, 2], variance_1[0, 2], variance_lag1[2, 0]]])
autocovariance23_1 = np.array([[variance_lag1[1, 2], variance_1[1, 2], variance_lag1[2, 1]]])


automatrix1_1 = np.zeros(shape=(Time, Time))
for i in range(Time):
    automatrix1_1[i, i] = autocovariance1_1[0, 0]

for i in range(Time-1):
    automatrix1_1[i, i + 1] = autocovariance1_1[0, 1]
    automatrix1_1[i+1, i] = autocovariance1_1[0, 1]

automatrix2_1 = np.zeros(shape=(Time, Time))
for i in range(Time):
    automatrix2_1[i, i] = autocovariance2_1[0, 0]

for i in range(Time-1):
    automatrix2_1[i, i + 1] = autocovariance2_1[0, 1]
    automatrix2_1[i+1, i] = autocovariance2_1[0, 1]

automatrix3_1 = np.zeros(shape=(Time, Time))
for i in range(Time):
    automatrix3_1[i, i] = autocovariance3_1[0, 0]

for i in range(Time-1):
    automatrix3_1[i, i + 1] = autocovariance3_1[0, 1]
    automatrix3_1[i+1, i] = autocovariance3_1[0, 1]

crossmatrixup12_1 = np.zeros(shape=(Time, Time))
for i in range(Time):
    crossmatrixup12_1[i, i]=autocovariance12_1[0,1]

for i in range(Time-1):
    crossmatrixup12_1[i, i + 1]=autocovariance12_1[0,2]
    crossmatrixup12_1[i+1, i]=autocovariance12_1[0,0]


crossmatrixup13_1 = np.zeros(shape=(Time, Time))
for i in range(Time):
    crossmatrixup13_1[i, i]=autocovariance13_1[0,1]

for i in range(Time-1):
    crossmatrixup13_1[i, i + 1]=autocovariance13_1[0,2]
    crossmatrixup13_1[i+1, i]=autocovariance13_1[0,0]

crossmatrixup23_1 = np.zeros(shape=(Time, Time))
for i in range(Time):
    crossmatrixup23_1[i, i]=autocovariance23_1[0,1]

for i in range(Time-1):
    crossmatrixup23_1[i, i + 1]=autocovariance23_1[0,2]
    crossmatrixup23_1[i+1, i]=autocovariance23_1[0,0]


#variancematrixT_1 = np.zeros(shape=(2*Time, 2*Time))
#variancematrixT_1[0:Time, 0:Time] = automatrix1_1[0:, 0:]
#variancematrixT_1[Time:, Time:] = automatrix2_1[0:, 0:]
#variancematrixT_1[0:Time, Time:] = crossmatrixup_1[0:, 0:]
#variancematrixT_1[Time:, 0:Time] = crossmatrixup_1[0:, 0:].T

variancematrixT_1 = np.zeros(shape=(3*Time, 3*Time))
variancematrixT_1[0:Time, 0:Time] = automatrix1_1[0:, 0:]
variancematrixT_1[Time:(2*Time), Time:(2*Time)] = automatrix2_1[0:, 0:]
variancematrixT_1[(2*Time):, (2*Time):] = automatrix3_1[0:, 0:]


corrlationmatrixT_1 = np.zeros(shape=(3*Time, 3*Time))
corrlationmatrixT_1[0:Time, 0:Time] = variancematrixT_1[0:Time, 0:Time] / variancematrixT_1[0, 0]
corrlationmatrixT_1[Time:(2*Time), Time:(2*Time)] = automatrix2_1[0:, 0:] / variancematrixT_1[Time, Time]
corrlationmatrixT_1[(2*Time):, (2*Time):] = automatrix3_1[0:, 0:] / variancematrixT_1[(2*Time), (2*Time)]

corrlationmatrixT_1[0:Time, Time:(2*Time)] = crossmatrixup12_1[0:, 0:] / ((variancematrixT_1[0, 0] * variancematrixT_1[Time,Time])**(1/2))
corrlationmatrixT_1[0:Time, (2*Time):(3*Time)] = crossmatrixup13_1[0:, 0:] / ((variancematrixT_1[0, 0] * variancematrixT_1[(2*Time),(2*Time)])**(1/2))
corrlationmatrixT_1[Time:(2*Time), (2*Time):(3*Time)] = crossmatrixup23_1[0:, 0:] / ((variancematrixT_1[Time, Time] * variancematrixT_1[(2*Time),(2*Time)])**(1/2))

corrlationmatrixT_1[Time:(2*Time), 0:Time] = corrlationmatrixT_1[0:Time, Time:(2*Time)].T
corrlationmatrixT_1[(2*Time):(3*Time), 0:Time] = corrlationmatrixT_1[0:Time, (2*Time):(3*Time)].T
corrlationmatrixT_1[(2*Time):(3*Time), Time:(2*Time)] = corrlationmatrixT_1[Time:(2*Time), (2*Time):(3*Time)].T


#####cluster2

cor12 = 0.5
cor13 = 0.5
cor23 = 0.5
coef1_2 = (-1 + (1 - 4 * 0.25 ** 2) ** (1/2)) / (2 * 0.25) #autocorelation is -0.25
coef2_2 = (1 - (1 - 4 * 0.45 ** 2) ** (1/2)) / (2 * 0.45)  #autocorelation is 0.45
coef3_2 = (1 - (1 - 4 * 0.25 ** 2) ** (1/2)) / (2 * 0.25)  #autocorelation is 0.45
A = np.array([[coef1_2, 0.0, 0.0],
              [0.0, coef2_2,0.0],
              [0.0, 0.0,coef3_2] ])
errormatrix = np.array([[1, (cor12 * (1 + coef1_2 ** 2) ** (1/2) * (1 + coef2_2 ** 2) ** (1/2)) / (1 + coef1_2* coef2_2),
                         (cor13 * (1 + coef1_2 ** 2) ** (1/2) * (1 + coef3_2 ** 2) ** (1/2)) / (1 + coef1_2 * coef3_2)],
                        [(cor12 * (1 + coef1_2 ** 2) ** (1/2) * (1 + coef2_2 ** 2) ** (1/2)) / (1 + coef1_2 * coef2_2), 1, (cor23 * (1 + coef2_2 ** 2) ** (1/2) * (1 + coef3_2 ** 2) ** (1/2)) / (1 + coef3_2 * coef2_2)],
                        [(cor13 * (1 + coef1_2 ** 2) ** (1/2) * (1 + coef3_2 ** 2) ** (1/2)) / (1 + coef3_2 * coef1_2),
                         (cor23 * (1 + coef2_2 ** 2) ** (1/2) * (1 + coef3_2 ** 2) ** (1/2)) / (1 + coef3_2 * coef2_2),1]
                        ])
variance_2 = errormatrix + np.dot(np.dot(A, errormatrix), A.T)
variance2_lag1 = np.dot(A, errormatrix)
autocovariance1_2 = np.array([[variance_2[0, 0], variance2_lag1[0, 0]]])
autocovariance2_2 = np.array([[variance_2[1, 1], variance2_lag1[1, 1]]])
autocovariance3_2 = np.array([[variance_2[2, 2], variance2_lag1[2, 2]]])
autocovariance12_2 = np.array([[variance2_lag1[0, 1], variance_2[0, 1], variance2_lag1[1, 0]]])
autocovariance13_2 = np.array([[variance2_lag1[0, 2], variance_2[0, 2], variance2_lag1[2, 0]]])
autocovariance23_2 = np.array([[variance2_lag1[1, 2], variance_2[1, 2], variance2_lag1[2, 1]]])


automatrix1_2 = np.zeros(shape=(Time, Time))
for i in range(Time):
    automatrix1_2[i, i] = autocovariance1_2[0, 0]

for i in range(Time-1):
    automatrix1_2[i, i + 1] = autocovariance1_2[0, 1]
    automatrix1_2[i+1, i] = autocovariance1_2[0, 1]

automatrix2_2 = np.zeros(shape=(Time, Time))
for i in range(Time):
    automatrix2_2[i, i] = autocovariance2_2[0, 0]

for i in range(Time-1):
    automatrix2_2[i, i + 1] = autocovariance2_2[0, 1]
    automatrix2_2[i+1, i] = autocovariance2_2[0, 1]

automatrix3_2 = np.zeros(shape=(Time, Time))
for i in range(Time):
    automatrix3_2[i, i] = autocovariance3_2[0, 0]

for i in range(Time-1):
    automatrix3_2[i, i + 1] = autocovariance3_2[0, 1]
    automatrix3_2[i+1, i] = autocovariance3_2[0, 1]

crossmatrixup12_2 = np.zeros(shape=(Time, Time))
for i in range(Time):
    crossmatrixup12_2[i, i]=autocovariance12_2[0,1]

for i in range(Time-1):
    crossmatrixup12_2[i, i + 1]=autocovariance12_2[0,2]
    crossmatrixup12_2[i+1, i]=autocovariance12_2[0,0]


crossmatrixup13_2 = np.zeros(shape=(Time, Time))
for i in range(Time):
    crossmatrixup13_2[i, i]=autocovariance13_2[0,1]

for i in range(Time-1):
    crossmatrixup13_2[i, i + 1]=autocovariance13_2[0,2]
    crossmatrixup13_2[i+1, i]=autocovariance13_2[0,0]

crossmatrixup23_2 = np.zeros(shape=(Time, Time))
for i in range(Time):
    crossmatrixup23_2[i, i]=autocovariance23_2[0,1]

for i in range(Time-1):
    crossmatrixup23_2[i, i + 1]=autocovariance23_2[0,2]
    crossmatrixup23_2[i+1, i]=autocovariance23_2[0,0]



variancematrixT_2 = np.zeros(shape=(3*Time, 3*Time))
variancematrixT_2[0:Time, 0:Time] = automatrix1_2[0:, 0:]
variancematrixT_2[Time:(2*Time), Time:(2*Time)] = automatrix2_2[0:, 0:]
variancematrixT_2[(2*Time):, (2*Time):] = automatrix3_2[0:, 0:]


corrlationmatrixT_2 = np.zeros(shape=(3*Time, 3*Time))
corrlationmatrixT_2[0:Time, 0:Time] = variancematrixT_2[0:Time, 0:Time] / variancematrixT_2[0, 0]
corrlationmatrixT_2[Time:(2*Time), Time:(2*Time)] = automatrix2_2[0:, 0:] / variancematrixT_2[Time, Time]
corrlationmatrixT_2[(2*Time):, (2*Time):] = automatrix3_2[0:, 0:] / variancematrixT_2[(2*Time), (2*Time)]

corrlationmatrixT_2[0:Time, Time:(2*Time)] = crossmatrixup12_2[0:, 0:] / ((variancematrixT_2[0, 0] * variancematrixT_2[Time,Time])**(1/2))
corrlationmatrixT_2[0:Time, (2*Time):(3*Time)] = crossmatrixup13_2[0:, 0:] / ((variancematrixT_2[0, 0] * variancematrixT_2[(2*Time),(2*Time)])**(1/2))
corrlationmatrixT_2[Time:(2*Time), (2*Time):(3*Time)] = crossmatrixup23_2[0:, 0:] / ((variancematrixT_2[Time, Time] * variancematrixT_2[(2*Time),(2*Time)])**(1/2))

corrlationmatrixT_2[Time:(2*Time), 0:Time] = corrlationmatrixT_2[0:Time, Time:(2*Time)].T
corrlationmatrixT_2[(2*Time):(3*Time), 0:Time] = corrlationmatrixT_2[0:Time, (2*Time):(3*Time)].T
corrlationmatrixT_2[(2*Time):(3*Time), Time:(2*Time)] = corrlationmatrixT_2[Time:(2*Time), (2*Time):(3*Time)].T

###cluster 1 independent, cluster2 ,3 and 4 have the same covariance structure.
corrlationmatrixT_3 = corrlationmatrixT_2
corrlationmatrixT_4 = corrlationmatrixT_2

#####generate margins

from scipy.stats import norm
from scipy.stats import t
from scipy.stats import gamma
from scipy.stats import chi2
from scipy.stats import lognorm


N=300
label_sample = np.zeros(shape=(105,4))
np.random.seed(3456)
for i in range(105):
    label_sample[i,0:] = np.random.multinomial(N, [1.0 / 4, 1.0 / 4, 1.0/4,1.0/4])

label_index = np.zeros(shape=(105,300))
for i in range(105):
    x = np.array([1,2,3,4])
    label_index[i,0:] = np.repeat(x, label_sample[i,0:].astype(int))


mean = np.zeros(shape=(1, (3*Time)))

data1 = np.zeros(shape=(N,3*Time,105))
np.random.seed(456)
for i in range(105):
    data1[0:int(label_sample[i,0]),0:,i] = np.random.multivariate_normal(mean=mean[0,0: ], cov=corrlationmatrixT_1, size=int(label_sample[i,0]), tol=1e-8)

print(data1[0,0:,0])
data2 = np.zeros(shape=(N,3*Time,105))

np.random.seed(789)
for i in range(105):
    data2[0:int(label_sample[i,1]),0:,i] = np.random.multivariate_normal(mean=mean[0,0: ], cov=corrlationmatrixT_2, size=int(label_sample[i,1]), tol=1e-8)


data3 = np.zeros(shape=(N,3*Time,105))

np.random.seed(123)
for i in range(105):
    data3[0:int(label_sample[i,2]),0:,i] = np.random.multivariate_normal(mean=mean[0,0: ], cov=corrlationmatrixT_3, size=int(label_sample[i,2]), tol=1e-8)

data4 = np.zeros(shape=(N,3*Time,105))

np.random.seed(456)
for i in range(105):
    data4[0:int(label_sample[i,3]),0:,i] = np.random.multivariate_normal(mean=mean[0,0: ], cov=corrlationmatrixT_4, size=int(label_sample[i,3]), tol=1e-8)



df_1 = 2*variancematrixT_1[Time,Time]/(variancematrixT_1[Time,Time]-1)


timeseries_1=np.zeros(shape=(N,3*Time,105))
for i in range(105):
    for j in range(int(label_sample[i,0])):
        timeseries_1[j,0:Time,i]=(norm.ppf(norm.cdf(data1[j,0: Time,i], loc=0, scale=1),loc=1,scale=variancematrixT_1[0,0]**(1/2)))
        timeseries_1[j,Time:(2*Time),i] = norm.ppf(norm.cdf(data1[j,Time: (2*Time),i], loc=0, scale=1),loc=3, scale=np.sqrt(6))
        timeseries_1[j,(2*Time):(3*Time),i] = norm.ppf(norm.cdf(data1[j,(2*Time): (3*Time),i], loc=0, scale=1),loc=4,
                                                       scale=variancematrixT_1[(2*Time),(2*Time)]**(1/2))


def mean_function(t):
    a=-8*(t-0.5)**2+4
    return a

time_index = np.array(range(30))/30
mean_c2_f2 = mean_function(time_index)


timeseries_2=np.zeros(shape=(N,3*Time,105))
for i in range(105):
    for j in range(int(label_sample[i,1])):
        timeseries_2[j,0:Time,i]=(gamma.ppf(norm.cdf(data2[j,0: Time,i], loc=0, scale=1),a=2,scale=1/2))
        timeseries_2[j,Time:(2*Time),i] = norm.ppf(norm.cdf(data2[j,Time: (2*Time),i], loc=0, scale=1),loc=3, scale=np.sqrt(6))
        timeseries_2[j,(2*Time):(3*Time),i] = norm.ppf(norm.cdf(data2[j,(2*Time): (3*Time),i], loc=0, scale=1),loc=mean_c2_f2,
                                                       scale=np.sqrt(2))


def mean_function2(t):
    a=8*(t-0.5)**2+4
    return a


mean_c3_f3 = mean_function2(time_index)


timeseries_3=np.zeros(shape=(N,3*Time,105))
for i in range(105):
    for j in range(int(label_sample[i,2])):
        timeseries_3[j,0:Time,i]=(gamma.ppf(norm.cdf(data3[j,0: Time,i], loc=0, scale=1),a=5,scale=1/2))
        timeseries_3[j,Time:(2*Time),i] = chi2.ppf(norm.cdf(data3[j,Time: (2*Time),i], loc=0, scale=1),df=3)
        timeseries_3[j,(2*Time):(3*Time),i] = lognorm.ppf(norm.cdf(data3[j,(2*Time): (3*Time),i], loc=0, scale=1),loc=mean_c3_f3,
                                                          s=np.sqrt(0.25))


timeseries_4=np.zeros(shape=(N,3*Time,105))
for i in range(105):
    for j in range(int(label_sample[i,3])):
        timeseries_4[j,0:Time,i]=(norm.ppf(norm.cdf(data4[j,0: Time,i], loc=0, scale=1),loc=1,scale=variancematrixT_2[0,0]**(1/2)))
        timeseries_4[j,Time:(2*Time),i] = norm.ppf(norm.cdf(data4[j,Time: (2*Time),i], loc=0, scale=1), loc=2, scale=np.sqrt(6))
        timeseries_4[j,(2*Time):(3*Time),i] = norm.ppf(norm.cdf(data4[j,(2*Time): (3*Time),i], loc=0, scale=1),loc=4,
                                                       scale=variancematrixT_2[(2*Time),(2*Time)]**(1/2))


dataTT = np.zeros(shape=(N,(3*Time),105))
for i in range(105):
    dataTT[0:int(label_sample[i,0]),0: ,i] = timeseries_1[0:int(label_sample[i,0]),0: ,i]
    dataTT[int(label_sample[i,0]):(int(label_sample[i,0])+int(label_sample[i,1])),0: ,i] = timeseries_2[0:int(label_sample[i,1]),0:,i]
    dataTT[(int(label_sample[i,0])+int(label_sample[i,1])):(int(label_sample[i,0])+int(label_sample[i,1])+int(label_sample[i,2])),0: ,i] = timeseries_3[0:int(label_sample[i,2]),0:,i]
    dataTT[(int(label_sample[i,0])+int(label_sample[i,1])+int(label_sample[i,2])):,0: ,i] = timeseries_4[0:int(label_sample[i,3]),0:,i]

