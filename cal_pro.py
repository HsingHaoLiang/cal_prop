#!/usr/bin/env python
# coding: utf-8

# In[16]:


#real model and calculate properties
import math
import numpy as np
from keras.models import  load_model, Sequential
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Activation, LSTM, concatenate, Dropout, LSTM, Bidirectional, Flatten
from keras_self_attention import SeqSelfAttention
from keras.optimizers import Adam
from keras.preprocessing import sequence

#normalize and renormalize
def normalize_test(original, ave, std):
    normalized = (original-ave)/std
    return normalized

def renormalize(normalized, ave, std):
    back = (normalized*std) + ave
    return back

df = pd.read_csv('ave_std.csv')
#read ave and std
for m in range(41,42):
    for k in range(1,2):
        n = 32*(m-41)
        ave = df.iloc[n:n+27,0]
        std = df.iloc[n:n+27,1]
        ave_T = df.iloc[n+27,0]
        std_T = df.iloc[n+27,1]
        ave_Tb = df.iloc[n+28,0]
        std_Tb = df.iloc[n+28,1]
        ave_Tc = df.iloc[n+29,0]
        std_Tc = df.iloc[n+29,1]
        ave_Pc = df.iloc[n+30,0]
        std_Pc = df.iloc[n+30,1]
        ave_pvap = df.iloc[n+31,0]
        std_pvap = df.iloc[n+31,1]

        strtemp = "./testdata" + str(m) + "_Trcri_PRSAC.txt"
        data2 = np.loadtxt(strtemp) #read test file
        id_test = data2[:,0:1]
        x_test = data2[:,1:len(data2[0])-31]
        x_temp1 = data2[:,len(data2[0])-31:len(data2[0])-4]
        pvap_test = data2[:,len(data2[0])-4:len(data2[0])-3]
        Tb_test = data2[:,len(data2[0])-3:len(data2[0])-2]
        Tc_test = data2[:,len(data2[0])-2:len(data2[0])-1]
        Pc_test = data2[:,len(data2[0])-1:]
        pvap_PRSAC = data2[:,len(data2[0])-8:len(data2[0])-7]
        Tb_PRSAC = data2[:,len(data2[0])-7:len(data2[0])-6]
        Tc_PRSAC = data2[:,len(data2[0])-6:len(data2[0])-5]
        Pc_PRSAC = data2[:,len(data2[0])-5:len(data2[0])-4]
        T_temp1 = data2[:,len(data2[0])-9:len(data2[0])-8]

        #normalized
        x_test_normalized= normalize_test(x_temp1, ave.values, std.values)
    
        T_test_normalized= normalize_test(T_temp1, ave_T, std_T)
    
        x_test  = np.hstack([x_test ,x_test_normalized])

        # load model
        strtemp = "./model_save/PRSAC_TbTcPcPvap" + str(m) + "_351_1.h5"
        model = load_model(strtemp, custom_objects={'SeqSelfAttention': SeqSelfAttention})

        #print(model.summary())

        z3 = model.predict(x_test)

        z3_renormalized = renormalize(z3[2], ave_Tc, std_Tc)

        z6_renormalized = renormalize(z3[3], ave_Pc, std_Pc)
        z9_renormalized = renormalize(z3[1], ave_Tb, std_Tb)
        z12_renormalized = renormalize(z3[0], ave_pvap, std_pvap)

        pvap_test_renormalized = renormalize(pvap_test, ave_pvap, std_pvap)
        Tb_test_renormalized = renormalize(Tb_test, ave_Tb, std_Tb)
        Tc_test_renormalized = renormalize(Tc_test, ave_Tc, std_Tc)
        Pc_test_renormalized = renormalize(Pc_test, ave_Pc, std_Pc)
    

        strtemp = "TbTcPc_new_test" + str(m) + "_351_" + str(k) + ".txt"
        f  = open(strtemp,"w")

        print("id T Tbcal(K) Tccal(K) lnPccal(Pa) lnPr(cal)",file=f)

        for i in range(0,len(id_test)):
            print(int(id_test[i])," ",float(T_temp1[i])*float(z3_renormalized[i])," ",float(z9_renormalized[i])," ",float(z3_renormalized[i])," ",float(z6_renormalized[i])," ",float(z12_renormalized[i]),file=f)

        f.close()


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

max_Tb[:]  = np.max(data_file,axis=2)[0:,2]
min_Tb[:]  = np.min(data_file,axis=2)[0:,2]
mean_Tb[:] = np.mean(data_file,axis=2)[0:,2]
std_Tb[:]  = np.std(data_file,axis=2,ddof=1)[0:,2]
max_Tc[:]  = np.max(data_file,axis=2)[0:,3]
min_Tc[:]  = np.min(data_file,axis=2)[0:,3]
mean_Tc[:] = np.mean(data_file,axis=2)[0:,3]
std_Tc[:]  = np.std(data_file,axis=2,ddof=1)[0:,3]
max_Pc[:]  = np.max(data_file,axis=2)[0:,4]
min_Pc[:]  = np.min(data_file,axis=2)[0:,4]
mean_Pc[:] = np.mean(data_file,axis=2)[0:,4]
std_Pc[:]  = np.std(data_file,axis=2,ddof=1)[0:,4]
max_pvap[:]  = np.max(data_file,axis=2)[0:,5]
min_pvap[:]  = np.min(data_file,axis=2)[0:,5]
mean_pvap[:] = np.mean(data_file,axis=2)[0:,5]
std_pvap[:]  = np.std(data_file,axis=2,ddof=1)[0:,5]

Tb_average = np.zeros(18860)
Tc_average = np.zeros(18860)
Pc_average = np.zeros(18860)
id_average = np.zeros(18860)
T = np.zeros(18860)

m = 0

for i in range(4,len(data_file),point):
    Tb_average[m] = mean_Tb[i]
    Tc_average[m] = mean_Tc[i]
    Pc_average[m] = mean_Pc[i]
    id_average[m] = int(data_file[i][0][0])
    T = float(data_file[i][1][0])
    m = m + 1

pvap_cal = np.zeros((18860,point))
m = 0
n = 0

for i in range(0,len(data_file)):
    pvap_cal[m][n] = mean_pvap[i] 
    n = n + 1
    if n == 8:
        m = m + 1
        n = 0


# In[13]:


print("chemical T(K) Tb(K)(1~100) Tc(K)(1~100) lnPc(1~100) lnPvap(1~100)")
print(int(id_average[0])," ",T[0],end=" ")
for i in range(0,100)    :
    print("{:.2f}".format(data_file[0][2][i]),end=" ") #data_file[compound][property(2=Tb, 3=Tc, 4=Pc, 5=Pr)][number_file]
for i in range(0,100)    :
    print("{:.2f}".format(data_file[0][3][i]),end=" ")
for i in range(0,100)    :
    print("{:.2f}".format(data_file[0][4][i]),end=" ")
for i in range(0,100)    :
    print("{:.2f}".format(data_file[0][5][i]+data_file[0][4][0]),end=" ")
print("")

print("chemical T Tb(K)(avg+std) Tc(K)(avg+std) lnPc(avg+std) T(K) lnPr(T)(avg+std)")
print(id_average[0]," ",T[0],end=" ")
print("{:.2f}({:.2f}) {:.2f}({:.2f}) {:.2f}({:.2f}) {:.2f}({:.2f})".format(mean_Tb[0],std_Tb[0],mean_Tc[0],std_Tc[0],mean_Pc[0],std_Pc[0],mean_pvap[0],std_pvap[0]))


# In[ ]:




