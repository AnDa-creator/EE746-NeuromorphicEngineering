#!/usr/bin/python3
from numpy.core.overrides import verify_matching_signatures
# import tensorflow as tf
import numpy as np
from math import exp
from numpy.random import binomial
from random import shuffle
from random import seed
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
        elif V_temp[i] < V_T:
            V_neuron_next[i] = V_temp[i] 
        else:
            Spike_next[i]  = np.int64(1)
            V_neuron_next[i] = V_temp[i]
    
    return V_neuron_next, Spike_next


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
    return prev_conc*(1 - h/tau_c) + Spike


#########################
def Weight_learner(last_conc, weight_prev,
                   C_theta=5,  del_c=3, nbit=3, type_syn = None):
    """
        Set type_syn as 1 for E --> E/I and 0 for I --> E/I, basically fanout from I or E.
    """
    
    p_plus = 0.1; p_minus = 0.1;
    
    
    # if type_syn not in (1, 0): raise ValueError("Invalid type")
    
    Wmax = 8 if type_syn==0 else 8*(1 - 2**(nbit - 1))
    Wmin = -8 if type_syn==1 else -8*(1 - 2**(nbit - 1))
    del_w = 0.0002 * 2**(nbit - 4)
    
    
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
    I_infi = 1000
    for a_neuron_id in neuron_ids:
        if a_neuron_id in desired_neuron_ids:
            I_teach[a_neuron_id] = I_infi * np.heaviside(C_theta +  delta_c - Calcium_conc[a_neuron_id], 0)
        else:
            I_teach[a_neuron_id] = - I_infi * np.heaviside(Calcium_conc[a_neuron_id] - (C_theta -  delta_c), 0)
    
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
    Calcium_conc = np.zeros((N_read,M))
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
        
        for i in range(N):
            if spikes_res[i,t] == 1:
                I_syn_additional = np.zeros((N_read,M))
                neuron_tp = synapses_res[i]["Neuron_type"]
                for j in range(N_read):
                    updates = syn_res(syn_string,neuron_tp,t,time,i,j,np.float64(Weights_readOut[j,i]),Delay,h,M)
                    I_syn_additional[j,:] = updates

                    W_new = Weight_learner(Calcium_conc[j,t-1], Weights_readOut[j,i], C_theta,  del_c, nbit, neuron_tp)
                    Weights_readOut[j,i] = W_new

                I_syn = I_syn + I_syn_additional

    return V_neurons, Spikes, Weights_readOut


##############################
def classifier(Spikes_readout,synapes_read):
    No_of_spikes = np.sum(Spikes_readout,1)
    print(No_of_spikes)
    class_out = np.argmax(No_of_spikes)
    return synapes_read[class_out], class_out

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