#!/usr/bin/python3
import math
from numpy.core.overrides import verify_matching_signatures
# import tensorflow as tf
import numpy as np
from math import exp
from numpy.random import binomial
# from random import shuffle
# from random import seed
import pandas as pd
import matplotlib.pyplot as plt

##########################
def ID_to_ind(nx,ny,nz,ID):
    x = int(ID/(ny*nz))
    y = int( (ID-(ny*nz)*x) / nz)
    z = int(ID%nz)
    return [x, y, z]


##########################
def LIF(V_neuron_prev,I_input_prev,I_input_next,N,h,index_next,index_prev_spike, params):
    C, g_L, E_L, V_T, R_p = params.values()
    R_p_ind = np.math.ceil(R_p/h)
    
    vmax, vmin = 32, -32
    
    V_neuron_next = E_L*np.ones((N,), dtype=np.float64)
    Spike_next = np.zeros((N,), dtype=np.int64)
    
    k1 = (1/C)*(-g_L*(V_neuron_prev-E_L)+I_input_prev)
    V_temp = V_neuron_prev + k1*h/2
    I_temp = I_input_prev/2 + I_input_next/2
    k2 = (1/C)*(-g_L*(V_temp-E_L)+I_temp)
    V_temp = V_neuron_prev + k2*h
    
    for i in range(N):
        if index_next-index_prev_spike[i] < R_p_ind:
            V_neuron_next[i] = E_L
#         elif V_temp[i] > vmax:
#             V_neuron_next[i] = vmax
        elif V_temp[i] < vmin:
            V_neuron_next[i] = vmin
        elif V_temp[i] < V_T:
            V_neuron_next[i] = V_temp[i] 
        else:
            Spike_next[i]  = np.int64(1)
            V_neuron_next[i] = V_temp[i]
    
    return V_neuron_next, Spike_next

# def LIF(V_neuron_prev,I_input_prev,I_input_next,N,h,index_next,index_prev_spike, params):
#     C, g_L, E_L, V_T, R_p = params.values()
#     R_p_ind = np.math.ceil(R_p/h)
    
#     V_neuron_next = E_L*np.ones((N,), dtype=np.float64)
#     Spike_next = np.zeros((N,), dtype=np.int64)
#     V_temp = V_neuron_prev - V_neuron_prev*g_L + I_input_next + I_input_prev
#     for i in range(N):
#             if index_next-index_prev_spike[i] < R_p_ind:
#                 V_neuron_next[i] = E_L
#             elif V_temp[i] < V_T:
#                 V_neuron_next[i] = V_temp[i] 
#             else:
#                 Spike_next[i]  = np.int64(1)
#                 V_neuron_next[i] = V_temp[i]
    
#     return V_neuron_next, Spike_next    

##########################
def syn_res(syn_string,type_syn,t,time,i,j,w_ij,del_i,h,M):  
    # spike in neuron i, produces a synaptic current in neuron j, weight = w_ij

    syn_curr = np.zeros((M),dtype=np.float64)
    # ts_ds = np.float64(time[t]) + del_i
    ind = np.int64(del_i/h) + t
    if ind > len(time) - 1:
#         print(ind)
        return syn_curr
    
    ts_ds = np.float64(time[ind]) 
    
    if syn_string == "static":
        syn_curr[ind] = w_ij/h
       
    elif syn_string == "first-order":
        tau_s = 4 * h 
        temp = w_ij * (1/tau_s) * np.exp(-(1/tau_s)*(time -ts_ds))
        syn_curr[ind:M] = temp[ind:M]


    elif syn_string == "second-order":
        if type_syn == 1:
            tau_s1, tau_s2 = 4, 8
        elif type_syn == 0:
            tau_s1, tau_s2 = 4, 2
        temp = (w_ij/(tau_s1-tau_s2)) * (np.exp(-(1/tau_s1)*(time -ts_ds)) -np.exp(-(1/tau_s2)*(time -ts_ds)))
        syn_curr[ind:M] = temp[ind:M]
               
    return syn_curr


##########################
def reservoir_solver(N, Delay, synapes, M, h, I_app, params_potential, Weights, syn_string):

    C, g_L, vrest, V_T, R_p = params_potential.values()
    
    I_syn = np.zeros((N,M),dtype=np.float64)
    I_total = np.zeros((N,M),dtype=np.float64)
    V_neurons = vrest*np.ones((N,M),dtype=np.float64) # potential of each neuron at every time instant
    Spikes = np.zeros((N,M),dtype=np.int64)         # 1 if ith neuron spikes at jth time step

#     syn_string = "static"
    
    index_prev_spike = -1*(M)*np.ones((N,),dtype=np.int64)

    time = np.array([j*h for j in range(M)],dtype=np.float64)

    for t in range(1,M):
        I_total = I_app + I_syn
#         print("current" , I_total[:,t-1])
        V_neuron, Spike = LIF(V_neurons[:,t-1],I_total[:,t-1],I_total[:,t],N,h,t,index_prev_spike, params_potential)  # solve for neuron potential and check if spike is produced
#         print("Potential", V_neuron)
        V_neurons[:,t] = V_neuron
        Spikes[:,t] = Spike

        I_syn_additional = np.zeros((N,M),dtype=np.float64)

        for i in range(N):
            if int(Spike[i]) == 1:
                index_prev_spike[i] = t
                
                neurons = synapes[i]["connections"]
                neuron_tp = synapes[i]["Neuron_type"]

                for j in range(len(neurons)): # iteration over the synapic connection from i to neurons[j]
                    updates = syn_res(syn_string,neuron_tp,t,time,i,neurons[j],np.float64(Weights[i,j]),Delay,h,M)
                    I_syn_additional[neurons[j],:] = updates
      
        I_syn = I_syn + I_syn_additional
    
    return V_neurons, Spikes


########################
def conc_update(prev_conc, Spike, tau_c, h):
#     print("\n", prev_conc)
    return prev_conc*(1 - h/tau_c) + Spike * h


#########################
def Weight_learner(last_conc, weight_prev,
                   C_theta=10,  del_c=2, nbit=3, type_syn = None):
    """
        Set type_syn as 1 for E --> E/I and 0 for I --> E/I, basically fanout from I or E.
    """
    
    p_plus = 0.1; p_minus = 0.1;
    
    
    # if type_syn not in (1, 0): raise ValueError("Invalid type")
    
    # Wmax = 8 if type_syn==0 else 8*(1 - 2**(nbit - 1))
    # Wmin = -8 if type_syn==1 else -8*(1 - 2**(nbit - 1))
    # del_w = 0.0002 * (2**(nbit - 4))

    Wmax = 8 
    Wmin = -8 
    del_w = 0.05

#     print("\n" + "new")
    
    if (C_theta < last_conc < C_theta + del_c) and (weight_prev < Wmax):
        Wnew = weight_prev + del_w if binomial(1, p_plus) == 1 else weight_prev
    elif (C_theta - del_c < last_conc < C_theta ) and (weight_prev > Wmin):
        Wnew = weight_prev - del_w if binomial(1, p_minus) == 1 else weight_prev
    else:
        Wnew = weight_prev
        
        
    return Wnew


#######################################
def teacher_current(neuron_ids, desired_neuron_ids, N_read, Calcium_conc, params_conc):
    C_theta, del_c, tau_c, nbits, delta_c = params_conc.values()
    I_teach = np.zeros((N_read,))
    I_infi = 20 * 10**(-3 + 12)
    for a_neuron_id in neuron_ids:
        if a_neuron_id in desired_neuron_ids:
            I_teach[a_neuron_id] = I_infi * np.heaviside(C_theta +  delta_c - Calcium_conc[a_neuron_id], 0)
        else:
            I_teach[a_neuron_id] = - 0.75 * I_infi * np.heaviside(Calcium_conc[a_neuron_id] - (C_theta -  delta_c), 0)
    
    return I_teach


######################################
def readOut_response(N_read,N, Delay, synapses_res, M, h, spikes_res, 
                     params_potential, params_conc, Weights_readOut_in,syn_string,training=False, train_ids=None):
                     
    C_theta, del_c, tau_c, nbit, delta_c = params_conc.values()
    C, g_L, vrest, V_T, R_p = params_potential.values()
    

    I_syn = np.zeros((N_read,M))
    I_total = np.zeros((N_read,M))
    V_neurons = vrest*np.ones((N_read,M)) # potential of each neuron at every time instant
    Spikes = np.zeros((N_read,M))         # 1 if ith neuron spikes at jth time step
    Calcium_conc = np.ones((N_read,M)) * C_theta
    I_teach = np.zeros((N_read,))

    Weights_readOut = Weights_readOut_in

#     syn_string = "static"
    
    index_prev_spike = -1*(M)*np.ones((N_read,))

    time = np.array([j*h for j in range(M)],dtype=np.float64)
    
    for t in range(1,M):
        I_total = I_syn 
        I_total[:,t-1] = I_total[:,t-1] + I_teach
        V_neuron, Spike = LIF(V_neurons[:,t-1],I_total[:,t-1],I_total[:,t],N_read,h,t,index_prev_spike, params_potential)  # solve for neuron potential and check if spike is produced


        V_neurons[:,t] = V_neuron
        Spikes[:,t] = Spike
        
        conc = conc_update(Calcium_conc[:,t-1], Spike, tau_c, h)
        Calcium_conc[:,t] = conc
        
        if training:
            neuron_ids = [i for i in range(N_read)]
            desired_neuron_ids = train_ids
            I_teach = teacher_current(neuron_ids, desired_neuron_ids,N_read, Calcium_conc[:,t], params_conc)
        
        for i in range(N_read):
            if Spike[i] == 1:
                index_prev_spike[i] = t
                
        I_syn_additional = np.zeros((N_read,M))
        for i in range(N):
            if spikes_res[i,t] == 1:
#                 print("\n Spike from:", (i,t))
                
                neuron_tp = synapses_res[i]["Neuron_type"]
                for j in range(N_read):
                    updates = syn_res(syn_string,neuron_tp,t,time,i,j,np.float64(Weights_readOut[j,i]),Delay,h,M)
                    I_syn_additional[j,:] += updates
                    if training:
                        W_new = Weight_learner(Calcium_conc[j,t-1], Weights_readOut[j,i], C_theta,  del_c, nbit, neuron_tp)
                        Weights_readOut[j,i] = W_new

        I_syn = I_syn + I_syn_additional
#         print(Weights_readOut[train_ids])

    return V_neurons, Spikes, Weights_readOut


##############################
def classifier(Spikes_readout,synapes_read):
    No_of_spikes = np.sum(Spikes_readout,1)
    class_out = np.argmax(No_of_spikes)
    return synapes_read[class_out], class_out, No_of_spikes


####################
def plot_spikes(Spike_train,N,M):
    plt.plot(0, 0)

    for i in range(N):
        for j in range(M):
            if(Spike_train[i,j] == 1):
                x1 = [i-0.25 , i+0.25]
                x2 = [j,j]
                plt.plot(x2,x1,color = 'black')

    plt.xlim([0, M])
    plt.ylim([0, N])
    plt.title("Spikes")  
    plt.xlabel("Time index")
    plt.ylabel("Neuron ID")
    plt.show()


##############################
def class_of_sample(label):
    if label == '00':
        return 0
    elif label == '01':
        return 1
    elif label == '02':
        return 2
    elif label == '03':
        return 3
    elif label == '04':
        return 4
    elif label == '05':
        return 5
    elif label == '06':
        return 6
    elif label == '07':
        return 7
    elif label == '08':
        return 8
    elif label == '09':
        return 9


###########################3
def Input_current_gen(file_name_List, syn_string, N, time_params, Input_CXNs, sign_win_matrix, training=False, train_Labels=None, seedvalue=4):
    input_num = 0
    h, Delay = time_params.values()
    for idx in range(len(file_name_List)):
        data = pd.read_csv(file_name_List[idx], sep=",", header=None)
        data_as_numpy = data.to_numpy()
        input = data_as_numpy.transpose()   # Single Sample input
        # (L,M) = input.shape

        (L,M1) = input.shape

        T = 500
        ## Input scaling to T = 1000ms, h = 1ms 
        M = math.ceil(T/h)

        h1 = T/M1
        input_temp = np.zeros((L,M))

        ind = (np.where(input == 1))
        t1 = np.array(ind[0])
        t2 = np.array(np.array(ind[1])*h1/h,dtype=np.int)

        input_temp[t1,t2] = 1

        input = input_temp

        ## Connection from input neurons to reservoir
        W_in_res = np.zeros((L,N)) # (i,j) entry is the weight of synapse from ith input to jth neuron in reservoir 
        W_in = 8
        Fin = 4 # no. of neurons a single input neuron is connected to

        connection_in_res = np.zeros((L,Fin),dtype=np.int64) # stores the id of reservoir neurons

#         reservoir_ID = [i for i in range(N)]
        
        for i in range(L):
            for j in range(Fin):
                sign_W_in = sign_win_matrix[i, j]
                W_in_res[i,Input_CXNs[i,j]] = sign_W_in*W_in
                connection_in_res[i,j] = Input_CXNs[i,j]
#         print("\n" , connection_in_res)

        ## Current input to the reservoir from the input neurons
        In_neurons = input   # spike train of L input neurons, over M timesteps, 1 if spike, 0 if no spike
        # print(In_neurons)
        In_app = np.zeros((N,M),dtype=np.float64)    # input current to the reservoir.
#         plot_spikes(input, L, M)

        time = np.array([j*h for j in range(M)],dtype=np.float64)

        for t in range(M):
            for i in range(L):
                if int(In_neurons[i,t]) == 1:
                    for j in range(Fin):
                        n_ID = connection_in_res[i,j]
                        w_ij = W_in_res[i,n_ID]
                        updates = syn_res(syn_string,1,t,time,i,n_ID,w_ij,Delay,h,M)
                        indices = [[n_ID,k] for k in range(M)]

                        In_app[n_ID,:] += updates
                        
        train_Label = class_of_sample(train_Labels[idx]) if training else "Null"
        input_num += 1
        yield In_app, L, M, train_Label, input_num