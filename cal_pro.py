#!/usr/bin/env python
# coding: utf-8

# In[12]:


#readfile
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

point=8 #number of datapoints of one compound
descriptor = "_351_" #input descriptors

data_file = np.zeros((18860*point,13,100))

for j in range(41,51):
    for k in range(1,11):
        strtemp = "TbTcPc_new_" + str(j) + descriptor + str(k) + ".txt"
        data = np.loadtxt(strtemp,skiprows=1)
        for i in range(len(data)):
            for l in range(len(data[0])):
                data_file[i][l][10*(j-41)+k-1] = data[i][l]

#calculate average, std, maximum and minimum
mean_Tb = np.zeros(18860*point)
std_Tb = np.zeros(18860*point)
mean_Tc = np.zeros(18860*point)
std_Tc = np.zeros(18860*point)
mean_Pc = np.zeros(18860*point)
std_Pc = np.zeros(18860*point)
mean_pvap = np.zeros(18860*point)
std_pvap = np.zeros(18860*point)

max_Tb[:]  = np.max(data_file,axis=2)[0:,1]
min_Tb[:]  = np.min(data_file,axis=2)[0:,1]
mean_Tb[:] = np.mean(data_file,axis=2)[0:,1]
std_Tb[:]  = np.std(data_file,axis=2,ddof=1)[0:,1]
max_Tc[:]  = np.max(data_file,axis=2)[0:,4]
min_Tc[:]  = np.min(data_file,axis=2)[0:,4]
mean_Tc[:] = np.mean(data_file,axis=2)[0:,4]
std_Tc[:]  = np.std(data_file,axis=2,ddof=1)[0:,4]
max_Pc[:]  = np.max(data_file,axis=2)[0:,7]
min_Pc[:]  = np.min(data_file,axis=2)[0:,7]
mean_Pc[:] = np.mean(data_file,axis=2)[0:,7]
std_Pc[:]  = np.std(data_file,axis=2,ddof=1)[0:,7]
max_pvap[:]  = np.max(data_file,axis=2)[0:,10]
min_pvap[:]  = np.min(data_file,axis=2)[0:,10]
mean_pvap[:] = np.mean(data_file,axis=2)[0:,10]
std_pvap[:]  = np.std(data_file,axis=2,ddof=1)[0:,10]

Tb_average = np.zeros(18860)
Tc_average = np.zeros(18860)
Pc_average = np.zeros(18860)
id_average = np.zeros(18860)
Tb_exp = np.zeros(18860)
Tc_exp = np.zeros(18860)
Pc_exp = np.zeros(18860)
Tb_PRSAC = np.zeros(18860)
Tc_PRSAC = np.zeros(18860)
Pc_PRSAC = np.zeros(18860)
data_deviation_Tb_PRSAC = np.zeros(18860)
data_deviation_Tb_ML = np.zeros(18860)
data_deviation_Tc_PRSAC = np.zeros(18860)
data_deviation_Tc_ML = np.zeros(18860)
data_deviation_Pc_PRSAC = np.zeros(18860)
data_deviation_Pc_ML = np.zeros(18860)

m = 0
AAD_Tb_ML = 0
AAD_Tc_ML = 0
AAD_Pc_ML = 0
AARD_Tb_ML = 0
AARD_Tc_ML = 0
AAD_Tb_PRSAC = 0
AAD_Tc_PRSAC = 0
AAD_Pc_PRSAC = 0
AARD_Tb_PRSAC = 0
AARD_Tc_PRSAC = 0

for i in range(4,len(data_file),point):
    Tb_average[m] = mean_Tb[i]
    Tc_average[m] = mean_Tc[i]
    Pc_average[m] = mean_Pc[i]
    id_average[m] = int(data_file[i][0][0])
    Tb_exp[m] = data_file[i][2][0]
    Tc_exp[m] = data_file[i][5][0]
    Pc_exp[m] = data_file[i][8][0]
    Tb_PRSAC[m] = data_file[i][3][0]
    Tc_PRSAC[m] = data_file[i][6][0]
    Pc_PRSAC[m] = data_file[i][9][0]
    
    AAD_Tb_ML = AAD_Tb_ML + abs(Tb_average[m] - Tb_exp[m])
    AAD_Tc_ML = AAD_Tc_ML + abs(Tc_average[m] - Tc_exp[m])
    AAD_Pc_ML = AAD_Pc_ML + abs(Pc_average[m] - Pc_exp[m])
    AARD_Tb_ML = AARD_Tb_ML + abs((Tb_average[m] - Tb_exp[m]) / Tb_exp[m])
    AARD_Tc_ML = AARD_Tc_ML + abs((Tc_average[m] - Tc_exp[m]) / Tc_exp[m])
    
    AAD_Tb_PRSAC = AAD_Tb_PRSAC + abs(Tb_PRSAC[m] - Tb_exp[m])
    AAD_Tc_PRSAC = AAD_Tc_PRSAC + abs(Tc_PRSAC[m] - Tc_exp[m])
    AAD_Pc_PRSAC = AAD_Pc_PRSAC + abs(Pc_PRSAC[m] - Pc_exp[m])
    AARD_Tb_PRSAC = AARD_Tb_PRSAC + abs((Tb_PRSAC[m] - Tb_exp[m]) / Tb_exp[m])
    AARD_Tc_PRSAC = AARD_Tc_PRSAC + abs((Tc_PRSAC[m] - Tc_exp[m]) / Tc_exp[m])
    
    data_deviation_Tb_PRSAC[m] = abs(Tb_PRSAC[m] - Tb_exp[m])
    data_deviation_Tb_ML[m] = abs(Tb_average[m] - Tb_exp[m])
    data_deviation_Tc_PRSAC[m] = abs(Tc_PRSAC[m] - Tc_exp[m])
    data_deviation_Tc_ML[m] = abs(Tc_average[m] - Tc_exp[m])
    data_deviation_Pc_PRSAC[m] = abs(Pc_PRSAC[m] - Pc_exp[m])
    data_deviation_Pc_ML[m] = abs(Pc_average[m] - Pc_exp[m])

    m = m + 1

pvap_cal = np.zeros((18860,point))
pvap_exp = np.zeros((18860,point))
pvap_PRSAC = np.zeros((18860,point))
ALD_ML = [0.0] * point
ALD_PRSAC = [0.0] * point
number_pvap_ML = [0] * point
number_pvap_PRSAC = [0] * point
m = 0
n = 0

for i in range(0,len(data_file)):
    pvap_cal[m][n] = mean_pvap[i]
    pvap_exp[m][n] = data_file[i][11][0]
    pvap_PRSAC[m][n] = data_file[i][12][0]
    if data_file[i][11][0] != -500 and data_file[i][12][0] != -500:
        ALD_ML[n] = ALD_ML[n] + abs(mean_pvap[i] - data_file[i][11][0])
        number_pvap_ML[n] = number_pvap_ML[n] + 1
    if data_file[i][11][0] != -500 and data_file[i][12][0] != -500:
        ALD_PRSAC[n] = ALD_PRSAC[n] + abs(data_file[i][12][0] - data_file[i][11][0])
        number_pvap_PRSAC[n] = number_pvap_PRSAC[n] + 1
    n = n + 1
    if n == 8:
        m = m + 1
        n = 0

print(m)

print("AAD_Tb_ML: ",AAD_Tb_ML/18860," AAD_Tb_PRSAC: ",AAD_Tb_PRSAC/18860)
print("AARD_Tb_ML: ",AARD_Tb_ML/18860," AARD_Tb_PRSAC: ",AARD_Tb_PRSAC/18860)
print("AAD_Tc_ML: ",AAD_Tc_ML/18860," AAD_Tc_PRSAC: ",AAD_Tc_PRSAC/18860)
print("AARD_Tc_ML: ",AARD_Tc_ML/18860," AARD_Tc_PRSAC: ",AARD_Tc_PRSAC/18860)
print("ALD_Pc_ML: ",AAD_Pc_ML/18860," ALD_Pc_PRSAC: ",AAD_Pc_PRSAC/18860)
print("AARD_Pc_ML: ",math.exp(AAD_Pc_ML/18860) - 1," ALD_Pc_PRSAC: ",math.exp(AAD_Pc_PRSAC/18860) - 1)
for i in range(point):
    print("ALD_Pr_",str(round(0.3+0.1*i, 1)),"Tc_ML: ",ALD_ML[i]/number_pvap_ML[i]," ALD_Pr_",str(round(0.3+0.1*i, 1)),"Tc_PRSAC: ",ALD_PRSAC[i]/number_pvap_PRSAC[i])
    print("AARD_Pr_",str(round(0.3+0.1*i, 1)),"Tc_ML: ",math.exp(ALD_ML[i]/number_pvap_ML[i])-1," AARD_Pr_",str(round(0.3+0.1*i, 1)),"Tc_PRSAC: ",math.exp(ALD_PRSAC[i]/number_pvap_PRSAC[i])-1)


# In[13]:


print("chemical Tb(K)(1~100) Tc(K)(1~100) lnPc(1~100) lnPvap(1~100)")
print(int(id_average[0]),end=" ")
for i in range(0,100)    :
    print("{:.2f}".format(data_file[0][4][i]),end=" ") #data_file[compound][property(1=Tb, 4=Tc, 7=Pc, 10=Pr)][number_file]
for i in range(0,100)    :
    print("{:.2f}".format(data_file[0][7][i]),end=" ")
for i in range(0,100)    :
    print("{:.2f}".format(data_file[0][10][i]+data_file[0][7][0]),end=" ")
print("")

print("chemical Tb(K)(avg+std) Tc(K)(avg+std) lnPc(avg+std) T(K) lnPr(T)(avg+std)")
print(id_average[0],end=" ")
#print(mean_Tb[0],"(",std_Tb[0],") ",mean_Tc[0],"(",std_Tc[0],") ",mean_Pc[0],"(",std_Pc[0],") ",mean_pvap[0],"(",std_pvap[0],") ")
print("{:.2f}({:.2f}) {:.2f}({:.2f}) {:.2f}({:.2f}) {:.2f}({:.2f})".format(mean_Tb[0],std_Tb[0],mean_Tc[0],std_Tc[0],mean_Pc[0],std_Pc[0],mean_pvap[0],std_pvap[0]))


# In[ ]:




