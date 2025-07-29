
import time as tt
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys, os


os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_MAX_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'


sys.path.append('/gpfs01/home/ppzaj/python_projects/Quantinuum_Driven_Boundaries/source_code/exact_diagonalization/') 
import bosonic_sparse_function as sbos  # type: ignore

sys.path.append('/gpfs01/home/ppzaj/python_projects/Quantinuum_Driven_Boundaries/source_code/') 
import bosonic_mpo_function as mbos  # type: ignore
from basic_functions import *  # type: ignore


if len(sys.argv) == 3:
    job_number = int(sys.argv[1])
    array_number = int(sys.argv[2])
if len(sys.argv) == 2:
    job_number = int(sys.argv[1])
    array_number = 0


ploting = True
plot_path = "bosonic_runs/plots/hermit/"
saving = False
save_path = "bosonic_runs/raw_data/hermit/"



num_traj = 1
num_traj_exp=f'{str(num_traj)[0]}e{int(np.log10(num_traj))}'


L_list = [5]
p_list = [0.25]
V_list = [1.0]#, 1.0, 4.0]
B_list = [0.0]#0.1*np.pi, 0.5*np.pi, 0.999*np.pi]
T_list = [100]#, 400]
X_list = [101, 201, 301]

array_input = [(l, p, v, b, t, x) for l in L_list for p in p_list for v in V_list for b in B_list for t in T_list for x in X_list]
input_L, input_p, input_V, input_B, input_T, input_X = array_input[np.mod(array_number, len(array_input))] # input_p = p_list[array_number]


Vs, Js = input_V, 1.0
Lx, Ly = input_L, input_L
Ls = Lx*Ly

dt = 0.1 
time_steps = input_T
time_steps_exp=f'{str(time_steps)[0]}e{int(np.log10(time_steps))}' #int(np.log10(time_steps))



print(f" ***** array job {array_number} for p={input_p}, size=({Lx}x{Ly}), T={time_steps}, V={input_V}, B={input_B} and X={input_X} ***** ")
print("")

N_avg = np.empty((num_traj, time_steps, Ls), dtype=np.float64)
N_sq_avg = np.empty((num_traj, time_steps, Ls), dtype=np.float64)
J_avg = np.empty((num_traj, time_steps, 2*Ls-Lx-Ly, 3), dtype=np.float64)
J_sq_avg = np.empty((num_traj, time_steps, 2*Ls-Lx-Ly, 3), dtype=np.float64)

# C_avg = np.empty((num_traj, time_steps, Ls, Ls), dtype=np.float64)

results_krs = np.empty((num_traj, time_steps), dtype=np.int8)

t_1 = tt.time()
for s in range(num_traj):
    ini_conf = np.random.choice([0, 1], size=Ls)    
    print(f" - trajectory {s:04}th running with ini_state={ini_conf}", flush=True)
    
    t_0 = tt.time()
    (loop_N, loop_J, loop_K) = mbos.mpo_simulation( time_steps, (Lx, Ly), (input_V, Js), 
                                                    driving_rate=input_p, 
                                                    dt=dt, 
                                                    initial_state=ini_conf, #sfer.checkerboard((Lx,Ly))
                                                    magnetic_field = input_B,
                                                    show_memory_usage = True,
                                                    chi_max = input_X, 
                                                    max_trunc_err = 0.01,
                                                    )
    
    N_avg[s] = loop_N 
    N_sq_avg[s] = loop_N**2 
    J_avg[s] = loop_J 
    J_sq_avg[s] = loop_J**2 
    
    # C_avg[s] = loop_C**2 
    results_krs[s] = loop_K
    
    print(f" - Time for trajectory {s:04} size=({Lx},{Ly}), p={input_p} and {num_traj} was:", tt.time()-t_0)
    print("")

N_avg = np.average(N_avg, axis=0)
N_sq_avg = np.average(N_sq_avg, axis=0)
N_DATA = np.concatenate(([N_avg],[N_sq_avg]), axis=0)
J_avg = np.average(J_avg, axis=0)
J_sq_avg = np.average(J_sq_avg, axis=0)
J_DATA = np.concatenate(([J_avg],[J_sq_avg]), axis=0)
    
# C_avg = np.average(C_avg, axis=0)

print(f"- Time for array:{array_number}, V={input_V}, p={input_p} and {num_traj} run(s):", tt.time()-t_1)
print("")
print(" data shapes are:", np.shape(N_avg), " , ", np.shape(J_avg), " , ", np.shape(results_krs))
# print(" data shapes are:", np.shape(C_avg), " , ", np.shape(results_krs))
print("")


if ploting: ####################### PLOTING #################################

        fig, ax = plt.subplots(2, 1, figsize=(8, 12), dpi=300, sharex=True,  layout='constrained',)
        
        # N_avg = np.diagonal(C_avg, axis1=1, axis2=2)
        colors = distinct_colors(Ls) # type: ignore
        for pl in range(Ls):
            ax[0].plot(np.arange(time_steps), N_avg[:,pl], label=f'r={pl}', linestyle='-', color=colors[pl])#, linewidth=0.7, marker=markers[pl], markersize=3*(3*L-2*pl)/(3*L))
            # sem = 5* np.sqrt(np.abs( N_sq_avg[:,pl] - N_avg[:,pl]**2 ))/np.sqrt(num_traj)    
            # ax.fill_between(time_array, N_avg[:,pl] + sem, N_avg[:,pl] - sem, color=f"C{pl}", alpha=0.18) #, linewidth=1.5)    
        ax[0].legend(ncols=Lx, fontsize='small')
        ax[0].set_ylabel(r'$\langle n_{r} \rangle$',fontsize=14)
        ax[0].legend(ncol=Lx, fontsize=9)
        ax[0].grid(which='major', linestyle=':', linewidth=0.8)
        
        # ax.set_xlabel('time',fontsize=14)
        # ax.set_title(f"<n_r> plot for ({Lx}x{Ly}), p={input_p}, V={Vs}, B={input_B:.3f}, dt ={dt} (fermion)")
        # fig.savefig(plot_path + f"N_avg_L{input_L:01}_P{input_p:.2f}_V{input_V:.1f}_B{input_B:.3f}_T{time_steps_exp}_N{num_traj_exp}_A{array_number:03}_J{job_number}.png", dpi=300, bbox_inches = 'tight')

        # J_avg = np.array([current_from_correlation(C_avg[n]) for n in range(time_steps)]) # type: ignore
        colors = distinct_colors(2*input_L*(input_L-1))  # type: ignore
        for r in range(2*input_L*(input_L-1)):
            i=int(J_avg[0,r,0])
            f=int(J_avg[0,r,1])
            ax[1].plot(np.arange(time_steps), J_avg[:,r,2], label=f'{i}->{f}', color=colors[r], linewidth=1.8)
        
        ax[1].set_ylabel(r'$\langle J_{n \rightarrow m} \rangle$',fontsize=14)
        ax[1].legend(ncol=Lx, fontsize=9)
        ax[1].grid(which='major', linestyle=':', linewidth=0.8)
        ax[1].set_xlabel('time steps',fontsize=14)
 
        # ax.set_title(f"Evolution of Currents for p={input_p:.2f}, V={input_V:.2f}, B={input_B:.3f} (fermions)")
        # fig.savefig(plot_path + f"current_L{input_L:01}_P{input_p:.2f}_V{input_V:.1f}_B{input_B:.3f}_T{time_steps_exp}_N{num_traj_exp}_A{array_number:03}_J{job_number}.png", dpi=300, bbox_inches = 'tight')

        fig.suptitle("plot for ({Lx}x{Ly}), p={input_p}, V={Vs}, B={input_B:.3f}, dt ={dt} (fermion)")
        fig.savefig(plot_path + f"plot_L{input_L:01}_P{input_p:.2f}_V{input_V:.1f}_B{input_B:.2f}_T{time_steps_exp}_N{num_traj_exp}_A{array_number:03}_J{job_number}.png", dpi=300, bbox_inches = 'tight')   

if saving: ######################## SAVING ##################################

        dict_data = {
        'density':N_DATA,
        'current':J_DATA,
        # 'current':C_avg,
        'krauses':results_krs }

        arcivo = open(save_path + f'data_L{input_L:01}_V{input_V:.1f}_B{input_B:.2f}_P{input_p:.2f}_T{time_steps_exp}_N{num_traj_exp}_A{array_number:03}.npy', 'wb')  
        np.save(arcivo, dict_data) # type: ignore
        arcivo.close()


        print(f"data for {input_p} file saved")
        print("")