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
import os

#normalize and renormalize
def normalize_test(original, ave, std):
    normalized = (original-ave)/std
    return normalized

def renormalize(normalized, ave, std):
    back = (normalized*std) + ave
    return back

#read ave and std
df = pd.read_csv('./ave_std.csv')

for m in range(41,51):
    for k in range(1,11):
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

        strtemp = "./test.txt"
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
        strtemp = "./model_save/PRSAC_TbTcPcPvap" + str(m) + "_351_" + str(k) + ".h5"
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

#readfile
datapoint=2 #number of datapoints
descriptor = "_351_" #input descriptors

data_file = np.zeros((datapoint,5,100))

for j in range(41,51):
    for k in range(1,11):
        strtemp = "TbTcPc_new_test" + str(j) + descriptor + str(k) + ".txt"
        data = np.loadtxt(strtemp,skiprows=1)
        for i in range(len(data)):
            for l in range(len(data[0])):
                data_file[i][l][10*(j-41)+k-1] = data[i][l]
        
        file_path = strtemp
        if os.path.exists(file_path):
            # remove file
            os.remove(file_path)

#calculate average, std, maximum and minimum
mean_Tb = np.zeros(datapoint)
std_Tb = np.zeros(datapoint)
mean_Tc = np.zeros(datapoint)
std_Tc = np.zeros(datapoint)
mean_Pc = np.zeros(datapoint)
std_Pc = np.zeros(datapoint)
mean_pvap = np.zeros(datapoint)
std_pvap = np.zeros(datapoint)

mean_Tb[:] = np.mean(data_file,axis=2)[0:,2]
std_Tb[:]  = np.std(data_file,axis=2,ddof=1)[0:,2]
mean_Tc[:] = np.mean(data_file,axis=2)[0:,3]
std_Tc[:]  = np.std(data_file,axis=2,ddof=1)[0:,3]
mean_Pc[:] = np.mean(data_file,axis=2)[0:,4]
std_Pc[:]  = np.std(data_file,axis=2,ddof=1)[0:,4]
mean_pvap[:] = np.mean(data_file,axis=2)[0:,5]
std_pvap[:]  = np.std(data_file,axis=2,ddof=1)[0:,5]

id_average = np.zeros(datapoint)
T = np.zeros(datapoint)

for i in range(0,datapoint):
    id_average[i] = int(data_file[i][0][0])
    T[i] = data_file[i][1][0]

for j in range(0,len(id_average)):
    print("chemical T(K) Tb(K)(1~100) Tc(K)(1~100) lnPc(1~100) lnPvap(1~100)")
    
    print(id_average[j],end=" ")
    print("{:.2f}".format(T[j]))
    for i in range(0,100):
        print("{:.2f}".format(data_file[j][2][i]),end=" ") #data_file[datapoint][property(2=Tb, 3=Tc, 4=Pc, 5=Pr)][number of models]
    for i in range(0,100):
        print("{:.2f}".format(data_file[j][3][i]),end=" ")
    for i in range(0,100):
        print("{:.2f}".format(data_file[j][4][i]),end=" ")
    for i in range(0,100):
        print("{:.2f}".format(data_file[j][5][i]+data_file[0][4][0]),end=" ")
    print("")
for j in range(0,len(id_average)):
    print("chemical T Tb(K)(avg+std) Tc(K)(avg+std) lnPc(avg+std) T(K) lnPr(T)(avg+std)")
    print(id_average[j],end=" ")
    print("{:.2f} {:.2f}({:.2f}) {:.2f}({:.2f}) {:.2f}({:.2f}) {:.2f}({:.2f})".format(T[j],mean_Tb[j],std_Tb[j],mean_Tc[j],std_Tc[j],mean_Pc[j],std_Pc[j],mean_pvap[j],std_pvap[j]))