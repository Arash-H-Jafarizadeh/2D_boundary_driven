
import time as tt
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import multiprocessing as mp
import sys, os


num_processes = os.cpu_count() 

os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_MAX_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'


sys.path.append('/gpfs01/home/ppzaj/python_projects/Quantinuum_Driven_Boundaries/source_code/exact_diagonalization/')  # Adjust the path to where fermionic_functions.py is located
import bosonic_sparse_function as sbos  # type: ignore


if len(sys.argv) == 3:
    job_number = int(sys.argv[1])
    array_number = int(sys.argv[2])
if len(sys.argv) == 2:
    job_number = int(sys.argv[1])
    array_number = 0


num_processes = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
print(f" **** Python script will create a pool of {num_processes} processes **** ", flush=True)


ploting = True
plot_path = "bosonic_runs/plots/hermit/"

saving = True
save_path = "bosonic_runs/raw_data/hermit/"



num_traj = 1000
time_steps = 10

size_list = [5]
p_list = [0.25]
V_list = [0.5]
B_list = [0.63]
array_input = [(l,p,v,b) for l in size_list for p in p_list for v in V_list for b in B_list]
input_L, input_p, input_V, input_B = array_input[np.mod(array_number, len(array_input))] 


Vs, Js = input_V, 1.0
Lx, Ly = input_L, input_L
Ls = Lx*Ly

dt = input_p #max_time/time_steps


print(f" ***** array job {array_number} for p={input_p}, size=({Lx}x{Ly}), T={time_steps*dt} and V={input_V} ***** ")
print("")


def init_pool_processes():
    np.random.seed()


def single_trajectory(seed):
    ############################################ creating random initial state
    ini_conf = np.random.choice([0, 1], size=Ls)

    ############################################ creating random states is a particular filling sector (i.e. half filled)
    # ini_conf = np.zeros(Ls, dtype=int)
    # ones_indices = np.random.choice(Ls, size=Ls//2, replace=False)
    # ini_conf[ones_indices] = 1

    print(f"trajectory {seed:05}th running on PID {os.getpid()} with ini_state={ini_conf}", flush=True)
    
    loop_N, loop_J, loop_k = sbos.normal_simulation( time_steps, (Lx, Ly), (input_V, Js), 
                                                    driving_rate = input_p, 
                                                    dt = dt, 
                                                    initial_state = ini_conf, #sbos.checkerboard((Lx,Ly))
                                                    magnetic_field = input_B
                                                    )

    return(loop_N, loop_J, loop_k)



if __name__ == '__main__':
    
    L = Lx*Ly
    time_array = np.arange(time_steps) #np.linspace(0, max_time, num = time_steps)

    print(f"Python script will create a pool of {num_processes} processes.")
    t_0 = tt.time()        
    N_avg = np.empty((num_traj, time_steps, L), dtype=np.float64)
    N_sq_avg = np.empty((num_traj, time_steps, L), dtype=np.float64)
    J_avg = np.empty((num_traj, time_steps, 2*L-Lx-Ly, 3), dtype=np.float64)
    J_sq_avg = np.empty((num_traj, time_steps, 2*L-Lx-Ly, 3), dtype=np.float64)
    results_krs = np.empty((num_traj, time_steps), dtype=np.int8)
    
    with mp.Pool(processes=num_processes, initializer=init_pool_processes) as pool:
        for indx, (N,J,K) in enumerate( pool.map(single_trajectory, range(num_traj)) ):
            print(f"  - got result of {indx}")
            N_avg[indx] = N #/ num_traj
            N_sq_avg[indx] = N**2 #/ num_traj
            J_avg[indx] = J #/ num_traj
            J_sq_avg[indx] = J**2 #/ num_traj
            results_krs[indx] = K
            
    print(f"- Time for {num_processes} CPUs and size=({Lx},{Ly}), p={input_p} and {num_traj} was:", tt.time()-t_0)
    print("")

    N_avg = np.average(N_avg, axis=0)
    N_sq_avg = np.average(N_sq_avg, axis=0)
    J_avg = np.average(J_avg, axis=0)
    J_sq_avg = np.average(J_sq_avg, axis=0)
    

    if ploting: ####################### PLOTING #################################
        
        def distinct_colors(n):
            cmap = plt.get_cmap('jet') #('viridis')
            return [cmap(i / (n - 1)) for i in range(n)]
        
        # markers=['o','s','d','^','v','>','<','*','p','']
        fig, ax = plt.subplots(1, 1, figsize=(8, 6),
                    subplot_kw=dict( 
                        # xscale = 'linear', #yscale ='log', # ylim = (1.e-17, 1e-2), #xlim = (1.e-12,100),
                        ylabel = r'$\langle n_{r} \rangle$', 
                        xlabel = r'$time$',#r'$\frac{ \# \text{states} }{ N_L }$',
                        title = f"density for ({Lx}x{Ly}), p={input_p}, V={Vs}, dt={dt} (boson) (inf_temperature_secor)"   
                        )                )
        
        colors = distinct_colors(L)
        for pl in range(L):
            ax.plot(time_array, N_avg[:,pl], label=f'r={pl}', linestyle='-', color=colors[pl]) #, linewidth=0.7, marker=markers[pl], markersize=3*(3*L-2*pl)/(3*L))
            # sem = 5* np.sqrt(np.abs( N_sq_avg[:,pl] - N_avg[:,pl]**2 ))/np.sqrt(num_traj)    
            # ax.fill_between(time_array, N_avg[:,pl] + sem, N_avg[:,pl] - sem, color=f"C{pl}", alpha=0.18) #, linewidth=1.5)


        ax.legend(ncols=Lx, fontsize='small')
        # fig.suptitle(f"Distances distribution for best |M| states + accuracy comparison - L={Ls}, V={Vs}", fontsize='medium', y=0.9001)    
        num_traj_exp=f'{str(num_traj)[0]}e{int(np.log10(num_traj))}'
        time_steps_exp=f'{str(time_steps)[0]}e{int(np.log10(time_steps))}' #int(np.log10(time_steps))
        fig.savefig(plot_path + f"N_avg_L{input_L:01}_P{input_p:.2f}_V{input_V:.1f}_T{time_steps_exp}_N{num_traj_exp}_A{array_number:03}_J{job_number}.png", dpi=300, bbox_inches = 'tight')
        # _N{num_traj:04}
        # _T{time_steps:05}
        # _T1e{time_steps_exp:01}_N1e{num_traj_exp:01}_

    if saving: ######################## SAVING ##################################
        
        num_traj_exp=f'{str(num_traj)[0]}e{int(np.log10(num_traj))}'
        time_steps_exp=f'{str(time_steps)[0]}e{int(np.log10(time_steps))}' #int(np.log10(time_steps))
        
        N_DATA = np.concatenate(([N_avg],[N_sq_avg]), axis=0)
        arcivo = open(save_path + f'N_data_L{input_L:01}_V{input_V:.1f}_P{input_p:.2f}_T{time_steps_exp}_N{num_traj_exp}_A{array_number:03}.npy', 'wb')  
        np.save(arcivo, N_DATA)
        arcivo.close()
        # arcivo = open(save_path + f'N_avg_L{input_L:01}_V{input_V:.1f}_P{input_p:.2f}_T{time_steps_exp}_N{num_traj_exp}_A{array_number:03}.npy', 'wb')  
        # np.save(arcivo, N_avg)
        # arcivo.close()
        # arcivo = open(save_path + f'N_qud_L{input_L:01}_V{input_V:.1f}_P{input_p:.2f}_T{time_steps_exp}_N{num_traj_exp}_A{array_number:03}.npy', 'wb')  
        # np.save(arcivo, N_sq_avg)
        # arcivo.close()

        J_DATA = np.concatenate(([J_avg],[J_sq_avg]), axis=0)
        arcivo = open(save_path + f'J_data_L{input_L:01}_V{input_V:.1f}_P{input_p:.2f}_T{time_steps_exp}_N{num_traj_exp}_A{array_number:03}.npy', 'wb')  
        np.save(arcivo, J_DATA)
        arcivo.close()
        # arcivo = open(save_path + f'J_avg_L{input_L:01}_V{input_V:.1f}_P{input_p:.2f}_T{time_steps_exp}_N{num_traj_exp}_A{array_number:03}.npy', 'wb')  
        # np.save(arcivo, J_avg)
        # arcivo.close()
        # arcivo = open(save_path + f'J_qud_L{input_L:01}_V{input_V:.1f}_P{input_p:.2f}_T{time_steps_exp}_N{num_traj_exp}_A{array_number:03}.npy', 'wb')  
        # np.save(arcivo, J_sq_avg)
        # arcivo.close()

        arcivo = open(save_path + f'K_data_L{input_L:01}_V{input_V:.1f}_P{input_p:.2f}_T{time_steps_exp}_N{num_traj_exp}_A{array_number:03}.npy', 'wb')  
        np.save(arcivo, np.array(results_krs, dtype=np.int64))
        arcivo.close()


        print(f"data for {input_p} file saved")
        print("")