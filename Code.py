import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import date

# %matplotlib inline
# from sympy import *                      #For other Approaches I imported them.
# import tensorflow as tf
# from tensorflow import keras


#################################################################################################################################
############################## Defining Global Parameters and importing the Data ################################################
#################################################################################################################################
alpha = 1 / 5.8
gammas = 1 / 5
eps = 0.66
Pop = 70 * 10 ** 6
R0_bounds = (15.6 * Pop / 100, 36 * Pop / 100)
CIR0_bound = (12, 30)


# Data Importing and Precessing
def import_data(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df['Date']).unique()

    conf = df['Confirmed']
    tested = df['Tested']
    dosed = df['First Dose Administered']

    df['Daily_Conf'] = df['Confirmed'].shift()
    df['Daily_Conf'] = df['Confirmed'] - df['Daily_Conf']
    df['DailyConf_RunningAvg'] = df['Daily_Conf'].rolling(7).mean()

    df['Daily_Test'] = df['Tested'].shift()
    df['Daily_Tested'] = df['Tested'] - df['Daily_Test']
    df['DailyTest_RunningAvg'] = df['Daily_Tested'].rolling(7).mean()

    df['Daily_Vac'] = df['First Dose Administered'].shift()
    df['Daily_Vac'] = df['First Dose Administered'] - df['Daily_Vac']
    dfx = df
    df = df[(df.Date >= datetime(2021, 3, 16)) &
            (df.Date <= datetime(2021, 4, 26))]
    df.reset_index(drop=True, inplace=True)

    df = df[['Confirmed', 'Tested', 'First Dose Administered', 'Daily_Vac', 'Daily_Tested',
             'Daily_Conf', 'DailyConf_RunningAvg', 'DailyTest_RunningAvg']]
    return df, dfx


data, datax = import_data('../COVID19_data.csv')  # Importing the Data


#################################################################################################################################
################################## Problem 1: Implementing the SEIRV model ######################################################
#################################################################################################################################


def SEIRV_model(betas, S0, E0, I0, R0, eps, i):
    if i <= 29:
        W0 = R0 / 30
    elif i > 173:
        W0 = SEIRV_model(betas, S0, E0, I0, R0, eps, i - 180)[3] + eps * \
             SEIRV_model(betas, S0, E0, I0, R0, eps, i - 180)[3]
    else:
        W0 = 0
    if i < 0:
        i = i + 7
    if i >= len(datax):
        del_V = 200000
    else:
        del_V = datax['Daily_Vac'][i]

    del_S = -betas * S0 * I0 / Pop - eps * del_V + W0
    del_E = betas * S0 * I0 / Pop - alpha * E0
    del_I = alpha * E0 - gammas * I0
    del_R = gammas * I0 + eps * del_V - W0

    #     del_S = tf.cast(del_S,tf.float32)          # Changed the type for the GradientTape() approach
    #     del_E = tf.cast(del_E,tf.float32)
    #     del_I = tf.cast(del_I,tf.float32)
    #     del_R = tf.cast(del_R,tf.float32)

    return del_S, del_E, del_I, del_R


#################################################################################################################################
############################################# Defining the Loss Function ########################################################
#################################################################################################################################


def loss_func(betas, S0, E0, I0, R0, CIR0, draw=False):
    #     betas, S0, E0, I0, R0 , CIR0 = Pn

    n = len(data)
    S = [0] * n
    E = [0] * n
    I = [0] * n
    R = [0] * n
    CIR = [0] * n
    betas, S[0], E[0], I[0], R[0], CIR[0] = betas, S0, E0, I0, R0, CIR0
    S[0] = Pop - E[0] - I[0] - R[0]
    for i in range(1, len(data)):
        S[i], E[i], I[i], R[i] = np.array([S[i - 1], E[i - 1], I[i - 1], R[i - 1]]) + SEIRV_model(betas, S[i - 1],
                                                                                                  E[i - 1], I[i - 1],
                                                                                                  R[i - 1], eps, i)
        CIR[i] = CIR[0] * data['DailyTest_RunningAvg'][0] / (data['DailyTest_RunningAvg'][i])
        S[i] = Pop - E[i] - I[i] - R[i]

    S = np.array(S)
    E = np.array(E)
    I = np.array(I)
    R = np.array(R)
    CIR = np.array(CIR)

    #     S = tf.convert_to_tensor(S, dtype=tf.float32)             # For purposes of using the GradientTape()
    #     E = tf.convert_to_tensor(E, dtype=tf.float32)
    #     I = tf.convert_to_tensor(I, dtype=tf.float32)
    #     R = tf.convert_to_tensor(R, dtype=tf.float32)
    #     CIR = tf.convert_to_tensor(CIR, dtype=tf.float32)

    del_I = alpha * E / CIR

    bar_del_I = [0] * n

    for i in range(7, len(del_I)):
        bar_del_I[i] = np.sum(del_I[i - 6:i + 1]) / 7
    for i in range(7):
        bar_del_I[i] = np.sum(del_I[:7]) / 7

    C = data['DailyConf_RunningAvg']
    bar_del_I = np.array(bar_del_I)
    residual = np.log10(C) - np.log10(bar_del_I)                #Logarithmic Base of 10
    loss = np.mean((residual ** 2))

    #     loss = tf.convert_to_tensor(loss , dtype=tf.float32)


    if draw:
        plt.figure(figsize=(8, 8))
        plt.plot(bar_del_I, label='Predicted daily average cases')
        plt.plot(C, label='Given average daily cases')
        plt.xlabel('Days')
        plt.ylabel('No. of Cases')
        plt.title("Given and Predicted average Daily Cases")
        plt.legend()
        plt.show()
    return loss


#################################################################################################################################
##################################### The sympy and GradientTape() approach #####################################################
#################################################################################################################################


######################## THE SYMPY APPROACH FOR FINDING THE PARTIAL DERIVATIVES#################


# def der_loss_func(Pn):
#     betas,S0,E0,I0,R0,CIR0 = symbols('betas S0 E0 I0 R0 CIR0')
#     Pnx = (betas,S0,E0,I0,R0,CIR0)
#     expr = loss_func(Pnx)
#     print(expr)
#     der_expr = [Derivative(expr,betas).doit(),Derivative(expr,S0).doit(),Derivative(expr,E0).doit(),Derivative(expr,I0).doit(),Derivative(expr,R0).doit(),Derivative(expr,CIR0).doit()]
# #     print(der_expr)
#     der_value = np.float32(np.array([der_expr[0].subs([(betas,Pn[0]) , (S0,Pn[1]) , (E0,Pn[2]) , (I0,Pn[3]) , (R0,Pn[4]) , (CIR0,Pn[5])]) ,
#                                      der_expr[1].subs([(betas,Pn[0]) , (S0,Pn[1]) , (E0,Pn[2]) , (I0,Pn[3]),(R0,Pn[4]),(CIR0,Pn[5])]),
#                                      der_expr[2].subs([(betas,Pn[0]) , (S0,Pn[1]) , (E0,Pn[2]) , (I0,Pn[3]),(R0,Pn[4]),(CIR0,Pn[5])]),
#                                      der_expr[3].subs([(betas,Pn[0]) , (S0,Pn[1]) , (E0,Pn[2]) , (I0,Pn[3]),(R0,Pn[4]),(CIR0,Pn[5])]),
#                                      der_expr[4].subs([(betas,Pn[0]) , (S0,Pn[1]) , (E0,Pn[2]) , (I0,Pn[3]),(R0,Pn[4]),(CIR0,Pn[5])]),
#                                      der_expr[5].subs([(betas,Pn[0]) , (S0,Pn[1]) , (E0,Pn[2]) , (I0,Pn[3]),(R0,Pn[4]),(CIR0,Pn[5])])]))
#     return der_value

# i=0
# while loss_func(Pn)>0.01:
#     print(Pn)
#     ders = der_loss_func(Pn)
#     betas = betas - (1/i+1)*ders[0]
#     S0 = S0 - (1/i+1)*ders[1]
#     E0 = E0 - (1/i+1)*ders[2]
#     R0 = R0 - (1/i+1)*ders[3]
#     I0 = I0 - (1/i+1)*ders[4]
#     CIR0 = CIR0 - (1/i+1)*ders[5]
#     Pn =  (betas,S0,E0,I0,R0,CIR0)

#     i += 1
#     if i==2:
#         break


#########################################################################################################################


################ TENSORFLOW APPROACH FOR FINDING THE GRADIENTS USING GradientTape() #############


# optimizer = keras.optimizers.Adam(learning_rate=1e-1)
# betas = tf.Variable(initial_value=tf.cast(betas,tf.float32),dtype = tf.float32, name='betas')
# S0 = tf.Variable(initial_value=tf.cast(S0,tf.float32), dtype = tf.float32, name='S0')
# E0 = tf.Variable(initial_value=tf.cast(E0,tf.float32),dtype = tf.float32,name='E0')
# I0 = tf.Variable(initial_value=tf.cast(I0,tf.float32),dtype = tf.float32, name='I0')
# R0 = tf.Variable(initial_value=tf.cast(R0,tf.float32),dtype = tf.float32, name='R0')
# CIR0 = tf.Variable(initial_value=tf.cast(CIR0,tf.float32),dtype = tf.float32, name='CIR0')

# iterations = 5
# lambda_ = 1
# for iter in range(iterations):
#     with tf.GradientTape() as tape:

#         cost_value = loss_func(betas,S0,E0,I0,R0,CIR0)
#     print(cost_value)
#     grads = tape.gradient( cost_value, [betas,S0,E0,I0,R0,CIR0] )

#     optimizer.apply_gradients( zip(grads, [betas,S0,E0,I0,R0,CIR0]) )


#################################################################################################################################
####################################### Problem 2: Calibrating the model parameters #############################################
#################################################################################################################################

betas = 2
E0 = 7.7e+04
I0 = 7.7e+04
R0 = 2 * 10 ** 7
S0 = Pop - E0 - R0 - I0  # For conservation
CIR0 = 13  # chose this as initial value after infinite trial and errors

val = loss_func(betas, S0, E0, I0, R0, CIR0)
i = 0
max_iter = 100

while val > 0.01:
    beta2 = betas - (1 / (i + 1)) * (
            loss_func(betas + 0.01, S0, E0, I0, R0, CIR0) - loss_func(betas - 0.01, S0, E0, I0, R0, CIR0)) / 0.02

    S02 = S0 - (1 / (i + 1)) * (
            loss_func(betas, S0 + 1, E0, I0, R0, CIR0) - loss_func(betas, S0 - 1, E0, I0, R0, CIR0)) / 2

    E02 = E0 - (1 / (i + 1)) * (
            loss_func(betas, S0, E0 + 1, I0, R0, CIR0) - loss_func(betas, S0, E0 - 1, I0, R0, CIR0)) / 2

    I02 = I0 - (1 / (i + 1)) * (
            loss_func(betas, S0, E0, I0 + 1, R0, CIR0) - loss_func(betas, S0, E0, I0 - 1, R0, CIR0)) / 2

    R02 = R0 - (1 / (i + 1)) * (
            loss_func(betas, S0, E0, I0, R0 + 1, CIR0) - loss_func(betas, S0, E0, I0, R0 - 1, CIR0)) / 2

    if R02 > 36 * Pop / 100:
        R02 = 36 * Pop / 100
    elif R02 < 15.6 * Pop / 100:
        R02 = 15.6 * Pop / 100

    CIR02 = CIR0 - (1 / (i + 1)) * (
            loss_func(betas, S0, E0, I0, R0, CIR0 + 0.1) - loss_func(betas, S0, E0, I0, R0, CIR0 - 0.1)) / 0.2

    if CIR02 > 30:
        CIR02 = 30
    elif CIR02 < 12:
        CIR02 = 12

    betas, S0, E0, I0, R0, CIR0 = beta2, S02, E02, I02, R02, CIR02
    val = loss_func(betas, S0, E0, I0, R0, CIR0)
    i += 1
    print('iter {}  : val {}'.format(i, val))
    print(betas, S0, E0, I0, R0, )
    if i == max_iter:
        break
print(f'Optimal Parameters: beta:{round(betas, 4)}, S0:{round(S0, 4)}, E0:{round(E0, 4)}, I0:{round(I0, 4)}, R0:{round(R0, 4)}, CIR_0:{round(CIR0, 4)}')
print(f'Final Loss Value = {val}')
loss_func(betas, S0, E0, I0, R0, CIR0, draw=True)


#################################################################################################################################
########################### Problem 3 and Problem 4: Open loop and Closed loop Controls and PLOTTING #############################
#################################################################################################################################


def numOfDays(date1, date2):
    return (date2 - date1).days


date1 = date(2021, 12, 31)
date2 = date(2021, 3, 16)
n = numOfDays(date2, date1)                 #The number of dats between the specified dates

dates = np.arange(date2, date1)             #List of the required dates

beta_list = [betas, 2 * betas / 3, betas / 2, betas / 3]
datax = datax[(datax.Date >= datetime(2021, 3, 16)) &
              (datax.Date <= datetime(2021, 12, 31))]
datax.reset_index(drop=True, inplace=True)
datax.loc[datax.Date >= datetime(2021, 4, 27), ['Daily_Vac']] = 200000

plt.figure(figsize=(10, 10))
plt.title('Open Loop and Closed Loop Control for Average Number of New Cases')
plt.xlabel("Days")
plt.ylabel("Average Number of New Cases")
# plt.xticks(dates,rotation = 90)

beta_list_name = ['Beta, Open Loop', '2/3 Beta, Open Loop', '1/2 Beta, Open Loop', '1/3 Beta, Open Loop']
for x, beta in enumerate(beta_list):

    S = [0] * n
    E = [0] * n
    I = [0] * n
    R = [0] * n
    CIR = [0] * n

    #Using the optimal parameters obtained in Problem 2
    betas, S[0], E[0], I[0], R[0], CIR[0] = (0.4263, 49846000.0, 77000.0, 77000.0, 20000000.0, 13.1659)

    S[0] = Pop - E[0] - I[0] - R[0]
    for i in range(1, n):
        S[i], E[i], I[i], R[i] = np.array([S[i - 1], E[i - 1], I[i - 1], R[i - 1]]) + SEIRV_model(beta, S[i - 1],
                                                                                                  E[i - 1], I[i - 1],
                                                                                                  R[i - 1], eps, i)
        if i < len(datax):
            CIR[i] = CIR[0] * datax['DailyTest_RunningAvg'][0] / (datax['DailyTest_RunningAvg'][i])
        else:
            CIR[i] = CIR[CIR.index(0) - 1]
        S[i] = Pop - E[i] - I[i] - R[i]

    S = np.array(S)
    E = np.array(E)
    I = np.array(I)
    R = np.array(R)
    CIR = np.array(CIR)

    del_I = alpha * E / CIR

    bar_del_I = [0] * n
    for i in range(7, len(del_I)):
        bar_del_I[i] = np.sum(del_I[i - 6:i + 1]) / 7
    for i in range(7):
        bar_del_I[i] = np.sum(del_I[:7]) / 7
    C = data['DailyConf_RunningAvg']
    bar_del_I = np.array(bar_del_I)
    #     print(beta_list[x])
    plt.plot(bar_del_I, label=beta_list_name[x])
    plt.legend()

S = [0] * n
E = [0] * n
I = [0] * n
R = [0] * n
CIR = [0] * n
bar_del_I = [0] * n

# Using the optimal parameters obtained in Problem 2
betas, S[0], E[0], I[0], R[0], CIR[0] = (0.4263, 49846000.0, 77000.0, 77000.0, 20000000.0, 13.1659)
S[0] = Pop - E[0] - I[0] - R[0]
for i in range(1, 7):
    #     day = dates[n].astype(datetime.datetime).isoweekday()
    S[i], E[i], I[i], R[i] = np.array([S[i - 1], E[i - 1], I[i - 1], R[i - 1]]) + SEIRV_model(beta, S[i - 1], E[i - 1],
                                                                                              I[i - 1], R[i - 1], eps,
                                                                                              i)
    CIR[i] = CIR[0] * datax['DailyTest_RunningAvg'][0] / (datax['DailyTest_RunningAvg'][i])

S = np.array(S)
E = np.array(E)
I = np.array(I)
R = np.array(R)
CIR = np.array(CIR)

del_I[:7] = alpha * E[:7] / CIR[:7]
for i in range(7):
    bar_del_I[i] = np.sum(del_I[:7]) / 7
CIR = CIR.tolist()
for i in range(7, n):
    day = dates[i].astype(datetime).isoweekday()
    if day == 2:
        if bar_del_I[i] <= 10000:
            beta = beta_list[0]
        elif 25000 >= bar_del_I[i] > 10001:
            beta = beta_list[1]
        elif 100000 >= bar_del_I[i] > 25001:
            beta = beta_list[2]
        else:
            beta = beta_list[3]

    S[i], E[i], I[i], R[i] = np.array([S[i - 1], E[i - 1], I[i - 1], R[i - 1]]) + SEIRV_model(beta, S[i - 1], E[i - 1],
                                                                                              I[i - 1], R[i - 1], eps,
                                                                                              i)
    if i < len(datax):
        CIR[i] = CIR[0] * datax['DailyTest_RunningAvg'][0] / (datax['DailyTest_RunningAvg'][i])
    else:
        CIR[i] = CIR[CIR.index(0) - 1]
    S[i] = Pop - E[i] - I[i] - R[i]
    del_I[i] = alpha * E[i] / CIR[i]
    bar_del_I[i] = np.sum(del_I[i - 6:i + 1]) / 7
plt.plot(bar_del_I, label='Closed Loop')

plt.plot(datax['Daily_Conf'], label="Ground Truth(Reported Cases)")
plt.legend()
#
plt.show()
#################################################################################################################################
########################### Problem 4: Plotting the evolution of the fraction of the susceptible population #####################
#################################################################################################################################

plt.figure(figsize=(10, 10))
plt.title('Open Loop and Closed Loop Control for Fraction of Susceptible People')
plt.xlabel("Days")
plt.ylabel("Fraction of Susceptible People")
beta_list_name = ['Beta, Open Loop', '2/3 Beta, Open Loop', '1/2 Beta, Open Loop', '1/3 Beta, Open Loop']
for x, beta in enumerate(beta_list):

    n = 290

    S = [0] * n
    E = [0] * n
    I = [0] * n
    R = [0] * n
    CIR = [0] * n
    betas, S[0], E[0], I[0], R[0], CIR[0] = (0.4263, 49846000.0, 77000.0, 77000.0, 20000000.0, 13.1659)
    S[0] = Pop - E[0] - I[0] - R[0]
    for i in range(1, n):
        S[i], E[i], I[i], R[i] = np.array([S[i - 1], E[i - 1], I[i - 1], R[i - 1]]) + SEIRV_model(beta, S[i - 1],
                                                                                                  E[i - 1], I[i - 1],
                                                                                                  R[i - 1], eps, i)
        if i < len(datax):
            CIR[i] = CIR[0] * datax['DailyTest_RunningAvg'][0] / (datax['DailyTest_RunningAvg'][i])
        else:
            CIR[i] = CIR[CIR.index(0) - 1]
        S[i] = Pop - E[i] - I[i] - R[i]

    S = np.array(S)
    E = np.array(E)
    I = np.array(I)
    R = np.array(R)
    CIR = np.array(CIR)

    del_I = alpha * E / CIR

    bar_del_I = [0] * n
    for i in range(7, len(del_I)):
        bar_del_I[i] = np.sum(del_I[i - 6:i + 1]) / 7
    for i in range(7):
        bar_del_I[i] = np.sum(del_I[:7]) / 7
    C = data['DailyConf_RunningAvg']
    bar_del_I = np.array(bar_del_I)
    #     print(beta_list[x])
    plt.plot(S / Pop, label=beta_list_name[x])
    plt.legend()

S = [0] * n
E = [0] * n
I = [0] * n
R = [0] * n
CIR = [0] * n
bar_del_I = [0] * n
betas, S[0], E[0], I[0], R[0], CIR[0] = (0.4263, 49846000.0, 77000.0, 77000.0, 20000000.0, 13.1659)
S[0] = Pop - E[0] - I[0] - R[0]
for i in range(1, 7):
    #     day = dates[n].astype(datetime.datetime).isoweekday()
    S[i], E[i], I[i], R[i] = np.array([S[i - 1], E[i - 1], I[i - 1], R[i - 1]]) + SEIRV_model(beta, S[i - 1], E[i - 1],
                                                                                              I[i - 1], R[i - 1], eps,
                                                                                              i)
    CIR[i] = CIR[0] * datax['DailyTest_RunningAvg'][0] / (datax['DailyTest_RunningAvg'][i])

S = np.array(S)
E = np.array(E)
I = np.array(I)
R = np.array(R)
CIR = np.array(CIR)

del_I[:7] = alpha * E[:7] / CIR[:7]
for i in range(7):
    bar_del_I[i] = np.sum(del_I[:7]) / 7
CIR = CIR.tolist()
for i in range(7, n):
    day = dates[i].astype(datetime).isoweekday()
    if day == 2:
        if bar_del_I[i] <= 10000:
            beta = beta_list[0]
        elif 25000 >= bar_del_I[i] > 10001:
            beta = beta_list[1]
        elif 100000 >= bar_del_I[i] > 25001:
            beta = beta_list[2]
        else:
            beta = beta_list[3]

    S[i], E[i], I[i], R[i] = np.array([S[i - 1], E[i - 1], I[i - 1], R[i - 1]]) + SEIRV_model(beta, S[i - 1], E[i - 1],
                                                                                              I[i - 1], R[i - 1], eps,
                                                                                              i)
    if i < len(datax):
        CIR[i] = CIR[0] * datax['DailyTest_RunningAvg'][0] / (datax['DailyTest_RunningAvg'][i])
    else:
        CIR[i] = CIR[CIR.index(0) - 1]
    S[i] = Pop - E[i] - I[i] - R[i]
    del_I[i] = alpha * E[i] / CIR[i]
    bar_del_I[i] = np.sum(del_I[i - 6:i + 1]) / 7
plt.plot(S / Pop, label='Closed Loop')

plt.legend()
plt.show()
