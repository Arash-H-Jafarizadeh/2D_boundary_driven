import time as tt
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys
import glob
import os

sys.path.append('/gpfs01/home/ppzaj/python_projects/Quantinuum_Driven_Boundaries/source_code/')
from basic_functions import *  # type: ignore

if len(sys.argv) == 3:
    job_number = int(sys.argv[1])
    array_number = int(sys.argv[2])
if len(sys.argv) == 2:
    job_number = int(sys.argv[1])
    array_number = 0



folder_path = "/gpfs01/home/ppzaj/python_projects/Quantinuum_Driven_Boundaries/fermionic_runs/raw_data/"

L_list = [5]
p_list = [0.25] #[0.25, 0.5]
V_list = [1.0, 4.0]
B_list = [0.0]#*np.pi, 0.1*np.pi, 0.5*np.pi, 0.999*np.pi]
T_list = [75]#200, 400]
K_list = ['MPO']#'InfMag'] #"InfRect" #"normal" # "Hermitian" 
X_list = [100, 150, 200, 250]


# array_input = [(l, p, v, b, t, k) for l in L_list for p in p_list for v in V_list for b in B_list for t in T_list for k in K_list]
# L_input, P_input, V_input, B_input, T_input, krs_type = array_input[array_number] 
array_input = [(l, p, v, b, t, x, k) for l in L_list for p in p_list for v in V_list for b in B_list for t in T_list for x in X_list for k in K_list]
L_input, P_input, V_input, B_input, T_input, X_input, krs_type = array_input[array_number] 


Lx, Ly = L_input, L_input
L = Lx * Ly

dt = 0.1#P_input

time_steps = T_input

t_step_str, n_traj_str = f'{str(time_steps)[0]}e{int(np.log10(time_steps))}', '2e1'#'1e3'
print(t_step_str)

num_traj = 20 #1000
max_time = time_steps * dt


if __name__ == '__main__': ############################################ Read and save data from array runs - new algorithm for fixed step_size/L/V - 20250407

    print(L_input,",",P_input,",",V_input,",",B_input,",", T_input,",", X_input,",", krs_type)

    save_data = True 
        
    input_path = "fermionic_runs/raw_data/mpo/"
    # input_path = "fermionic_runs/raw_data/p_data/"
    # input_path = "fermionic_runs/raw_data/s_data/"
    
    save_path = "fermionic_runs/raw_data/"
    
    
    Ls = Lx*Ly
        
    # all_files =  sorted( glob.glob(f'data_L{L_input}_V{V_input:.1f}_B{B_input:.3f}_P{P_input:.2f}_T{t_step_str}_N{n_traj_str}'+'*.npy', root_dir=input_path) )
    all_files =  sorted( glob.glob(f'data_L{L_input}_V{V_input:.1f}_B{B_input:.3f}_P{P_input:.2f}_T{t_step_str}_X{X_input:03}_N{n_traj_str}'+'*.npy', root_dir=input_path) )
    num_files = len(all_files)
    print(all_files)
    print("")
     
        
    # N_avg = np.zeros((num_files, time_steps, Ls), dtype=np.float64)
    # N_sq_avg = np.zeros((num_files, time_steps, Ls), dtype=np.float64)
    # J_avg = np.zeros((num_files, time_steps, 2*Ls-Lx-Ly, 3), dtype=np.float64)
    # J_sq_avg = np.zeros((num_files, time_steps, 2*Ls-Lx-Ly, 3), dtype=np.float64)
    results_krs = np.empty((num_files, num_traj, time_steps), dtype=np.int8)

    C_avg = np.empty((num_files, time_steps, Ls, Ls), dtype=np.complex128)
    NN_avg = np.empty((num_files, time_steps, Ls, Ls), dtype=np.float64)
    NN_sq_avg = np.empty((num_files, time_steps, Ls, Ls), dtype=np.float64)
        
    # print("shape check 0: ", np.shape(N_avg),',',np.shape(N_sq_avg),',',np.shape(J_avg),',',np.shape(J_sq_avg),',',np.shape(results_krs),',')
    print("shape check 0: ", np.shape(C_avg),',',np.shape(NN_avg),',',np.shape(results_krs),',')
    print("")

    for indx, nome in enumerate(all_files):
        print(" - file name is ", nome)
        
        # if V_input < 0:
        #     array_nam = nome[33:36]
        # if V_input > 0:
        #     array_nam = nome[32:35]        
        # print('array is', array_nam)
        # print("")
        
        A_data = np.load(input_path + nome, allow_pickle=True).item()
        print("test key names A_data: ", A_data.keys())

        # N_avg[indx] = A_data['density'][0]        # N_sq_avg[indx] = A_data['density'][1]        
        # J_avg[indx] = A_data['current'][0]        # J_sq_avg[indx] = A_data['current'][1]
        
        print(" - ",np.shape(A_data['krauses']))
        results_krs[indx] = A_data['krauses']
        
        
        print(" - ",np.shape(A_data['hopping_correlation']))
        C_avg[indx] = A_data['hopping_correlation']#['correlation']
        
        print(" - ",np.shape(A_data['density_correlation']))
        NN_avg[indx] = A_data['density_correlation']#[0]
        # NN_sq_avg[indx] = A_data['density_correlation'][1]        
        
        
        print(f" -=-=-=-= {indx} =-=-=-=- ")
    
    
    # N_avg = np.average(N_avg, axis=0)    # N_sq_avg = np.average(N_sq_avg, axis=0)    # N_DATA = np.concatenate(([N_avg],[N_sq_avg]), axis=0)
    # J_avg = np.average(J_avg, axis=0)    # J_sq_avg = np.average(J_sq_avg, axis=0)    # J_DATA = np.concatenate(([J_avg],[J_sq_avg]), axis=0)

    results_krs = np.concatenate(results_krs, axis=0)
    
    C_avg = np.average(C_avg, axis=0)
    NN_avg = np.average(NN_avg, axis=0)
    
    N_avg = np.real( np.diagonal(C_avg, axis1=1, axis2=2) )
    J_avg = np.array([current_from_correlation(C_avg[t], (Lx,Ly)) for t in range(time_steps)]) # type: ignore
    
        
    # print("shape check1: ", np.shape(N_avg),',',np.shape(N_sq_avg),',',np.shape(J_avg),',',np.shape(J_sq_avg),',',np.shape(results_krs),',')
    # print("shape check2: ", np.shape(N_DATA),' , ',np.shape(J_DATA),' , ',np.shape(results_krs),',')
    
    
    
    if save_data: ###################################### SAVING ###############################################
        
        # arcivo = open(save_path + f'N_data_L{L_input:01}_V{V_input:.1f}_P{P_input:.2f}_T{t_step_str}_N{n_traj_str}_{krs_type}.npy', 'wb')  
        # np.save(arcivo, N_DATA) #N_avg)
        # arcivo.close()
        # # arcivo = open(save_path + f'N_sq_avg_L{L_input:01}_V{V_input:.1f}_P{P_input:.2f}_T{t_steps}_N{n_traj}_normal.npy', 'wb')  
        # # np.save(arcivo, N_sq_avg)
        # # arcivo.close()

        # arcivo = open(save_path + f'K_data_L{L_input:01}_V{V_input:.1f}_P{P_input:.2f}_T{t_step_str}_N{n_traj_str}_{krs_type}.npy', 'wb')  
        # np.save(arcivo, np.array(results_krs, dtype=np.int64))
        # arcivo.close()
        
        steps_str = t_step_str 
        trajs_str = f'{str(num_files * num_traj)[0]}e{int(np.log10(num_files * num_traj))}'
        
        dict_data = {
            'dt':dt,
            'chi_max':X_input,
            'time_steps':time_steps,
            'trajectory':num_files * num_traj,
            'density':N_avg,#N_DATA,
            'current':J_avg,#J_DATA,
            'hopping_correlation':C_avg,
            'density_correlation':NN_avg,
            'krauses':results_krs }

        # arcivo = open(save_path + f'Data_L{L_input:01}_V{V_input:.1f}_B{B_input:.3f}_P{P_input:.2f}_T{steps_str}_N{trajs_str}_{krs_type}_dict.npy', 'wb')
        arcivo = open(save_path + f'Data_L{L_input:01}_V{V_input:.1f}_B{B_input:.2f}_P{P_input:.2f}_T{steps_str}_X{X_input:03}_N{trajs_str}_{krs_type}_dict.npy', 'wb')    
        np.save(arcivo, dict_data) # type: ignore
        arcivo.close()

        print(f"data for L{L_input} V{V_input:.1f} P{P_input:.2f} T{t_step_str} N{trajs_str} saved")
        print("")
    

