
import time as tt
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import multiprocessing as mp
import sys, os


os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_MAX_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'


sys.path.append('/gpfs01/home/ppzaj/python_projects/Quantinuum_Driven_Boundaries/source_code/')
import bosonic_mpo_function as mbos  # type: ignore
from basic_functions import *  # type: ignore


if len(sys.argv) == 3:
    job_number = int(sys.argv[1])
    array_number = int(sys.argv[2])
if len(sys.argv) == 2:
    job_number = int(sys.argv[1])
    array_number = 0


num_processes = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
print(f" **** Python script will create a pool of {num_processes} processes **** ", flush=True)


ploting = True
plot_path = "bosonic_runs/plots/mpo/"
# plot_path = "bosonic_runs/plots/single_process/"

saving = True
save_path = "bosonic_runs/raw_data/mpo/"
# save_path = "bosonic_runs/raw_data/s_data/"



num_traj = 20
num_traj_exp=f'{str(num_traj)[0]}e{int(np.log10(num_traj))}'

# time_steps = 200

L_list = [5]
p_list = [0.25]
V_list = [0.0, 4.0]
B_list = [0.0]#0.1*np.pi, 0.5*np.pi, 0.999*np.pi]
T_list = [75]#, 400]
X_list = [100, 150]

array_input = [(l, p, v, b, t, x) for l in L_list for p in p_list for v in V_list for b in B_list for t in T_list for x in X_list]
input_L, input_p, input_V, input_B, input_T, input_X = array_input[np.mod(array_number, len(array_input))] # input_p = p_list[array_number]


Vs, Js = input_V, 1.0
Lx, Ly = input_L, input_L
Ls = Lx*Ly

dt = 0.1 #input_p #max_time/time_steps

time_steps = input_T
time_steps_exp=f'{str(time_steps)[0]}e{int(np.log10(time_steps))}' #int(np.log10(time_steps))

print(f" ***** array job {array_number} for size={input_L}, p={input_p}, V={input_V}, B={input_B}, T={time_steps} and X={input_X} ***** ")
print("")


def init_pool_processes():
    np.random.seed()


def single_trajectory(seed):
    ini_conf = np.random.choice([0, 1], size=Ls)
    print(f"trajectory {seed:05}th running on PID {os.getpid()} with ini_state={ini_conf}", flush=True)
    loop_C, loop_NN, loop_k = mbos.mpo_simulation( time_steps, (Lx, Ly), (Vs, Js), 
                                                    dt=dt, 
                                                    driving_rate=input_p, 
                                                    initial_state=ini_conf, #sfer.checkerboard((Lx,Ly))
                                                    # magnetic_field = input_B,
                                                    chi_max = input_X,
                                                    total_error_order = 2,
                                                    # max_trunc_err = 0.09,
                                                    )
    
    return(loop_C, loop_NN, loop_k)



if __name__ == '__main__':
    
    Ls = Lx*Ly

    print(f"Python script will create a pool of {num_processes} processes.")
    t_0 = tt.time()        
    # N_avg = np.empty((num_traj, time_steps, Ls), dtype=np.float64)
    # N_sq_avg = np.empty((num_traj, time_steps, Ls), dtype=np.float64)
    # J_avg = np.empty((num_traj, time_steps, 2*Ls-Lx-Ly, 3), dtype=np.float64)
    # J_sq_avg = np.empty((num_traj, time_steps, 2*Ls-Lx-Ly, 3), dtype=np.float64)
    C_avg = np.empty((num_traj, time_steps, Ls, Ls), dtype=np.complex128)
    NN_avg = np.empty((num_traj, time_steps, Ls, Ls), dtype=np.float64)
    results_krs = np.empty((num_traj, time_steps), dtype=np.int8)
    
    with mp.Pool(processes=num_processes, initializer=init_pool_processes) as pool:
        for indx, (C,NN,K) in enumerate( pool.map(single_trajectory, range(num_traj)) ):
            print(f"  - got result of {indx}")
            # N_avg[indx] = N #/ num_traj
            # N_sq_avg[indx] = N**2 #/ num_traj
            # J_avg[indx] = J #/ num_traj
            # J_sq_avg[indx] = J**2 #/ num_traj
            C_avg[indx] = C
            NN_avg[indx] = NN
            results_krs[indx] = K
            
    print(f"- Time for {num_processes} CPUs and size=({Lx},{Ly}), p={input_p} and {num_traj} was:", tt.time()-t_0)
    print("")

    # N_avg = np.average(N_avg, axis=0)
    # N_sq_avg = np.average(N_sq_avg, axis=0)
    # N_DATA = np.concatenate(([N_avg],[N_sq_avg]), axis=0)

    # J_avg = np.average(J_avg, axis=0)
    # J_sq_avg = np.average(J_sq_avg, axis=0)
    # J_DATA = np.concatenate(([J_avg],[J_sq_avg]), axis=0)
    
    C_avg = np.average(C_avg, axis=0)
    NN_avg = np.average(NN_avg, axis=0)
    # NN_sq_avg = np.average(NN_sq_avg, axis=0)
    # NN_DATA = np.concatenate(([NN_avg],[NN_sq_avg]), axis=0)

    
    N_avg = np.diagonal(C_avg, axis1=1, axis2=2)
    J_avg = np.array([current_from_correlation(C_avg[t], (Lx,Ly)) for t in range(time_steps)]) # type: ignore
    
    if ploting: ################################################# PLOTING ########################################################

        def distinct_colors(n):
            cmap = plt.get_cmap('jet') #('viridis')
            return [cmap(i / (n - 1)) for i in range(n)]
    
    
        ###### early plot of densities ######
        fig, ax = plt.subplots(2, 1, figsize=(8, 12), dpi=300, sharex=True,  layout='constrained',)
        
        colors = distinct_colors(Ls)
        for pl in range(Ls):
            ax[0].plot(np.arange(time_steps), N_avg[:,pl], label=f'r={pl}', linestyle='-', color=colors[pl])#, linewidth=0.7, marker=markers[pl], markersize=3*(3*L-2*pl)/(3*L))
            # sem = 5* np.sqrt(np.abs( N_sq_avg[:,pl] - N_avg[:,pl]**2 ))/np.sqrt(num_traj)    
            # ax.fill_between(time_array, N_avg[:,pl] + sem, N_avg[:,pl] - sem, color=f"C{pl}", alpha=0.18) #, linewidth=1.5)
        
        ax[0].legend(ncols=Lx, fontsize='small')
        ax[0].set_ylabel(r'$\langle n_{r} \rangle$',fontsize=14)
        ax[0].legend(ncol=Lx, fontsize=9)
        ax[0].grid(which='major', linestyle=':', linewidth=0.8)
        
        
        colors = distinct_colors(2*input_L*(input_L-1)) 
        for r in range(2*input_L*(input_L-1)):
            i=int(J_avg[0,r,0])
            f=int(J_avg[0,r,1])
            ax[1].plot(np.arange(time_steps), J_avg[:,r,2], label=f'{i}->{f}', color=colors[r], linewidth=1.8)
        
        ax[1].set_ylabel(r'$\langle J_{n \rightarrow m} \rangle$',fontsize=14)
        ax[1].legend(ncol=Lx, fontsize=9)
        ax[1].grid(which='major', linestyle=':', linewidth=0.8)
        ax[1].set_xlabel('time steps',fontsize=14)
 
        # ax.set_title(f"Evolution of Currents for p={input_p:.2f}, V={input_V:.2f}, B={input_B:.3f} (bosons)")
        # fig.savefig(plot_path + f"current_L{input_L:01}_P{input_p:.2f}_V{input_V:.1f}_B{input_B:.3f}_T{time_steps_exp}_N{num_traj_exp}_A{array_number:03}_J{job_number}.png", dpi=300, bbox_inches = 'tight')

        fig.suptitle(f"plot for ({Lx}x{Ly}), p={input_p}, V={Vs}, B={input_B:.3f}, dt ={dt}, X={input_X:03} (boson)")
        fig.savefig(plot_path + f"plot_L{input_L:01}_P{input_p:.2f}_V{input_V:.1f}_B{input_B:.3f}_X{input_X:03}_T{time_steps_exp}_N{num_traj_exp}_A{array_number:03}_J{job_number}.png", dpi=300, bbox_inches = 'tight')    
    
    if saving: ######################## SAVING ##################################
        
        dict_data = {
            # 'density':N_DATA,
            # 'current':J_DATA,
            'density_correlation':NN_avg,#NN_DATA,
            'hopping_correlation':C_avg,
            'krauses':results_krs }

        arcivo = open(save_path + f'data_L{input_L:01}_V{input_V:.1f}_B{input_B:.3f}_P{input_p:.2f}_T{time_steps_exp}_X{input_X:03}_N{num_traj_exp}_A{array_number:03}.npy', 'wb')  
        np.save(arcivo, dict_data) # type: ignore
        arcivo.close()


        print(f"data for {input_p} file saved")
        print("")