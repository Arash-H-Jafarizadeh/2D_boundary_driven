import time as tt
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys

sys.path.append('/gpfs01/home/ppzaj/python_projects/Quantinuum_Driven_Boundaries/source_code/') 
from basic_functions import *  # type: ignore


if len(sys.argv) == 3:
    job_number = int(sys.argv[1])
    array_number = int(sys.argv[2])
if len(sys.argv) == 2:
    job_number = int(sys.argv[1])
    array_number = 0
        
        

folder_path = "/gpfs01/home/ppzaj/python_projects/Quantinuum_Driven_Boundaries/bosonic_runs/raw_data/"

L_list = [5]
p_list = [0.25]#[0.025, 0.05, 0.1, 0.2, 0.5]
V_list = [0.0, 4.0]# [0.0, 4.0]
B_list = [0.0]#0.1*np.pi, 0.5*np.pi, 0.999*np.pi]
T_list = [75]#[200, 400]
K_list = ["MPO"] # "InfRect" # "random" # "normal" # "hermitian" 
X_list = [100, 150, 200, 250]

# array_input = [(l,p,v,b,t,k) for l in L_list for p in p_list for v in V_list for b in B_list for t in T_list for k in K_list]
# L_input, P_input, V_input, B_input,T_input, K_type = array_input[array_number]  

array_input = [(l,p,v,b,t,k, x) for l in L_list for p in p_list for v in V_list for b in B_list for t in T_list for k in K_list for x in X_list]
L_input, P_input, V_input, B_input,T_input, K_type, X_input = array_input[array_number]  


Lx, Ly = L_input, L_input
L = Lx * Ly 


time_steps = T_input
steps_str = f'{str(T_input)[0]}e{int(np.log10(T_input))}'
trajs_str = '8e2'


# data = np.load(folder_path + f"Data_L{L_input}_V{V_input:.1f}_B{B_input:.3f}_P{P_input:.2f}_T{steps_str}_N{trajs_str}_{K_type}_dict_0.npy",  allow_pickle=True).item()
data = np.load(folder_path + f"Data_L{L_input}_V{V_input:.1f}_B{B_input:.3f}_P{P_input:.2f}_T{steps_str}_X{X_input:03}_N{trajs_str}_{K_type}_dict.npy",  allow_pickle=True).item()
    
N_data = data['density']#[0]
# N_data_std = data['density'][1]
J_data = data['current']#[0]
# J_data_std = data['current'][1]
dtt = 0.1#data['dt']
max_t = data['time_steps']
num_traj = data['trajectory']

max_time = time_steps * dtt

if False: ####################################### Density profile SnapShots #######################################################################

    N_data = np.load(folder_path +  f"_N_avg_(3x3)_P{p_input}_V0.15.npy", allow_pickle=True)
    
    time_steps = len(N_data)
    time_data = np.linspace(0, max_time, num = time_steps)

    time_snaps = [0,5,10,15,20,30,40,50,59]
    
    Fig, axs = plt.subplots(3, 3, figsize=(10, 8), sharex=True, sharey=True, )# layout='constrained', gridspec_kw=dict( wspace=0.001,),)
    # Fig.subplots_adjust(wspace=0.001, hspace=0.2)

    axs_list = [axs[0,0], axs[0,1], axs[0,2], axs[1,0], axs[1,1], axs[1,2], axs[2,0], axs[2,1], axs[2,2]]
    for plt_num, t_s in enumerate(time_snaps):
    
        AX = axs_list[plt_num]
        pcm = AX.pcolormesh( np.array(N_data[t_s,:,:], dtype=np.float64), cmap='RdBu_r', vmin=0, vmax=1)
        # AX.set_aspect(1)
        AX.set_xticks([])  # Remove x-axis ticks
        AX.set_yticks([])  # Remove y-axis ticks
        AX.tick_params(left=False, bottom=False)  # Remove tick lines
        AX.set_title(f't={time_data[time_snaps[plt_num]]:.2f}',  y=0.97)
        
    Fig.colorbar(pcm, ax=axs, location='right', shrink=0.99,  fraction=.1) 
    Fig.suptitle(f"time snapshots of particle counts for {Lx}x{Ly} {system}-system: p={p_input}, V={0.15}", y=0.95)
    Fig.savefig(f"bosonic_runs/{system}_N_TimeSnap_p{p_input}_({Lx}x{Ly}).png", dpi=600, bbox_inches = 'tight')

    
if True:################################# *IMPROVED* AVG Current SnapShots #######################################################################
    
    plt_type='Avg' #'Avg' #'Fin'
    
    def connected_pairs(Lx, Ly):
        """
        Returns a list of tuples (label1, label2) for all nearest-neighbor pairs
        in an Lx x Ly rectangular lattice labeled row-major from 0 to Lx*Ly-1,
        excluding any pair where either label is a corner (0 or Lx*Ly-1).
        """
        pairs = []
        corner_labels = {0, Lx*Ly-1}
        for i in range(Ly):
            for j in range(Lx):
                label = i * Lx + j
                if j + 1 < Lx: # Right neighbor
                    right_label = i * Lx + (j + 1)
                    if label not in corner_labels and right_label not in corner_labels:
                        pairs.append((label, right_label))
                if i + 1 < Ly: # Down neighbor
                    down_label = (i + 1) * Lx + j
                    if label not in corner_labels and down_label not in corner_labels:
                        pairs.append((label, down_label))
        return pairs

    # def connected_pairs(N, M):
    #     pairs = []
    #     for i in range(N):
    #         for j in range(N):
    #             label = i * N + j
    #             # if j + 1 < N and label != 0 and i*N+j+1 != N**2-1: # right neighbor 
    #             if j + 1 < N: # right neighbor 
    #                 pairs.append((label, i*N+(j+1)))
    #             # if i + 1 < N and label != 0 and (i+1)*N+j != N**2-1: # up neighbor
    #             if i + 1 < N: # up neighbor
    #                 pairs.append((label, (i+1)*N+j))
    #     return pairs


    if plt_type == 'Fin':
        j_data = J_data[-1] # 
        n_data = N_data[-1] # 
        # min_J, max_J = np.min(np.abs(J_data[:,2]), axis=None), np.max(np.abs(J_data[:,2]), axis=None)
        title_str = rf'Last time step $t={max_t}*\delta t$ (V={V_input}, B={B_input})'
        # file_name_str = f"BOS_final_time_snapp_L{L_input}_V{V_input}_B{B_input:.1f}_P{P_input:.2f}_T{steps_str}_N{trajs_str}_{K_type}.pdf"    
        file_name_str = f"BOS_final_time_snapp_L{L_input}_V{V_input}_B{B_input:.1f}_P{P_input:.2f}_T{steps_str}_X{X_input:03}_N{trajs_str}_{K_type}.pdf"    
        
    if plt_type == 'Avg':
        j_data = np.mean(J_data[-5:], axis=0)  
        n_data = np.mean(N_data[-5:], axis=0)  
        # min_J, max_J = np.min(np.abs(J_data[:,2]), axis=None), np.max(np.abs(J_data[:,2]), axis=None)
        title_str = rf'Last 5 time steps averaged (V={V_input}, B={B_input})'
        # file_name_str = f"BOS_averg_time_snapp_L{L_input}_V{V_input}_B{B_input:.1f}_P{P_input:.2f}_T{steps_str}_N{trajs_str}_{K_type}.pdf"
        file_name_str = f"BOS_averg_time_snapp_L{L_input}_V{V_input}_B{B_input:.1f}_P{P_input:.2f}_T{steps_str}_X{X_input:03}_N{trajs_str}_{K_type}.pdf"
    
    
    
    positions = connected_pairs(Lx, Ly) 

    J_dict = {(int(i), int(j)): v for i, j, v in j_data}
    print(" J_dict form:", len(J_dict), '\n')

    
    vertex_pos = {(i, j): (j, i) for i in range(Lx) for j in range(Ly)}  # (row, col): (x, y)
    vertex_numbers = np.arange(Lx*Ly).reshape(Lx, Ly)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 9), sharex=True, sharey=True, layout='constrained', gridspec_kw=dict( wspace=0.2,hspace = 0.) )

    arrow_kwargs = dict(width=0.07, head_width=0.182, head_length=0.2, length_includes_head=True, zorder=2)
 
    # min_J, max_J = 0.0, np.max(np.abs( list(J_dict.values()) ), axis=None) # ~~> this is wrong becausit values for all bonds
    min_J, max_J = 0.0, np.max(np.abs( [J_dict[(i,j)] for i , j in positions] ), axis=None)
 
    J_norm = plt.Normalize(min_J, max_J) # type: ignore
    J_cmap = plt.cm.viridis #summer # YlGn_r # binary #  type: ignore 

    N_norm = plt.Normalize(.00, 0.999) # type: ignore
    N_cmap = plt.cm.RdBu_r # type: ignore #Blues #binary


    l_off, h_off = 0.1, 0.25
    
    idx = {i: (i % Lx, i // Ly) for i in range(Lx*Ly)}
        

    for n, (x,y) in enumerate(positions):
        x1, y1 = idx[x]
        x2, y2 = idx[y]
        if y1 == y2 : ## Horizontal edge
            val = J_dict[(x,y)] #J_data[n, -1]
            color = J_cmap(J_norm(np.abs(val))) #cmap(norm(val))
            if val >= 0:
                dx = x2-x1
                ax.arrow(x1+l_off, y1, dx-h_off, 0, color=color, **arrow_kwargs)
            else:
                dx = x1-x2
                ax.arrow(x1+1-l_off, y1, dx+h_off, 0, color=color, **arrow_kwargs)
                # ax.arrow(x1+1-l_off, y1, dx+h_off, 0, **arrow_kwargs)
        if x1 == x2: ## Vertical Edges
            val = J_dict[(x,y)] #J_data[n, -1]
            color = J_cmap(J_norm(np.abs(val))) #cmap(norm(val))
            if val >= 0:
                dy = y2-y1
                ax.arrow(x1, y1+l_off, 0, dy-h_off, color=color, **arrow_kwargs)
                # ax.arrow(x1, y1+l_off, 0, dy-h_off, **arrow_kwargs)
            else:
                dy = y1-y2
                ax.arrow(x1, y2-l_off, 0, dy+h_off, color=color, **arrow_kwargs)
                # ax.arrow(x1, y2-l_off, 0, dy+h_off, **arrow_kwargs)
    
    
    for (i, j), (x, y) in vertex_pos.items():
        n_val =  n_data[i*Ly + j] #N_data[i, j]
        n_color = N_cmap(N_norm(np.abs(n_val))) 
        csm = ax.scatter(x, y, s=2000, color=n_color, edgecolor='black', linewidth=2, zorder=3)
        ax.text(x, y, str(vertex_numbers[i, j]), ha='center', va='center_baseline', fontsize=32, zorder=4)

    ax.axis('off')
    ax.set_aspect('equal')
    
        
    # Colorbar placement
    sm = plt.cm.ScalarMappable(cmap=J_cmap, norm=J_norm)
    sm.set_array([])
    J_cb = fig.colorbar(sm, ax=ax, shrink = 0.7, aspect=30)

    csm = plt.cm.ScalarMappable(cmap=N_cmap, norm=N_norm)
    csm.set_array([])
    N_cb = fig.colorbar(csm, ax=ax, location='bottom', orientation='horizontal', shrink=0.7, aspect=30) # type: ignore

    J_cb.ax.set_title(r'$J_{n,m}$', fontsize=14)
    N_cb.ax.set_title(r'$\langle n_i \rangle$', loc='right', y=0.9,fontsize=14)
    
    fig.text(0.9, 0.10, f'Bosons\n V={V_input}, p={P_input}', ha='center', va='center_baseline', fontsize=24, zorder=-4)


    ax.set_title(title_str,  y=0.98)
    fig.savefig("bosonic_runs/plots/" + file_name_str, dpi=500, bbox_inches ='tight')    
    


if True: ####################################### N time Plots all inputs ####################################################################
    
    # num_traj = 20000 
    # max_time = dtt * max_t
    
    t_data = np.linspace(0, max_time, num = time_steps)

    # sem = 5* np.sqrt( np.abs(N_data_std - N_data**2 )) / np.sqrt(num_traj, dtype=np.float64) #* np.sqrt(num_traj/(num_traj-1) )    
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharex=True, layout='constrained', 
                           gridspec_kw=dict( wspace=0., hspace = 0.)
                        )
    

    colors = distinct_colors(Lx*Ly)  # type: ignore
    for r in range(Lx*Ly):    
        ax.plot(t_data, N_data[:,r], label=f'j={r}', color=colors[r], linewidth=1.8)
        # ax.fill_between(t_data, N_data[:,r] + sem[:,r], N_data[:,r] - sem[:,r], color=colors[r], alpha=0.2) #, linewidth=1.5)

    ax.set_title(f"Evolution of densities for p={P_input}, V={V_input} (bosons)")
    ax.set_ylabel(r'$\langle n_j \rangle$',fontsize=14)
    ax.legend(ncol=Lx, fontsize=9)
    ax.grid(which='major', linestyle=':', linewidth=0.8)
    ax.set_xlabel('time',fontsize=14)
    
    # fig.savefig(f"bosonic_runs/plots/BOS_density_vs_time_L{L_input}_V{V_input:.1f}_B{B_input:.1f}_P{P_input:.2f}_T{steps_str}_N{trajs_str}_{K_type}.pdf", dpi=400, bbox_inches ='tight')
    fig.savefig(f"bosonic_runs/plots/BOS_density_vs_time_L{L_input}_V{V_input:.1f}_B{B_input:.1f}_P{P_input:.2f}_T{steps_str}_X{X_input:03}_N{trajs_str}_{K_type}.pdf", dpi=500, bbox_inches ='tight')
   
    


if True: ####################################### J time Plots all p's ####################################################################
    
    
    def lower_diag(N):
        pairs = []
        for i in range(N):
            for j in range(N):
                if i < j:
                    label = i * N + j
                    if j + 1 < N and i < j + 1:
                        pairs.append((label, i * N + (j + 1) ))
                    if i + 1 < N and i + 1 < j:
                        pairs.append((label, (i + 1) * N + j ))
        return pairs

    
    # max_time = dtt * max_t
    t_data = np.linspace(0, max_time, num = time_steps)

    # sem = 5* np.sqrt( np.abs(J_data_std - J_data**2 )) / np.sqrt(num_traj) #* np.sqrt(num_traj/(num_traj-1) )    
    
    # positions = [(0,1),(0,3),(1,2),(1,4),(2,5),(3,4),(3,6),(4,5),(4,7),(5,8),(6,7),(7,8)]
    # vertex_pos = {(i, j): (j, i) for i in range(Lx) for j in range(Ly)}  # (row, col): (x, y)
    # vertex_numbers = np.arange(Lx*Ly).reshape(Ly, Lx)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharex=True, layout='constrained')#, gridspec_kw=dict( wspace=0.0,hspace = 0.))
    
    dig_inds = np.array([r*(Lx+1) for r in range(Lx)]+[r*(Lx+1)+1 for r in range(Lx-1)])
    
   
    colors = distinct_colors(2*L_input*(L_input-1))  # type: ignore
    for r in range(2*L_input*(L_input-1)):
        i=int(J_data[0,r,0])
        f=int(J_data[0,r,1])
        ax.plot(t_data, J_data[:,r,2], label=f'{i}->{f}', color=colors[r], linewidth=1.8)
        # ax[0].fill_between(t_data, J_data[:,r,2] + sem[:,r,2], J_data[:,r,2] - sem[:,r,2], color=f"C{r}", alpha=0.2) #, linewidth=1.5)
        # pl_num += 1
    
    ax.set_title(f"Evolution of Currents for p={P_input}, V={V_input} (bosons)")
    ax.set_ylabel(r'$\langle J_{n \rightarrow m} \rangle$',fontsize=14)
    ax.legend(ncol=Lx, fontsize=9)
    ax.grid(which='major', linestyle=':', linewidth=0.8)
    ax.set_xlabel('time',fontsize=14)
     
    
    # fig.suptitle(f"Current time evolution for {Lx}x{Ly} system, p={P_input}, V={V_input} and {K_type} kraus")
    # fig.savefig(f"bosonic_runs/plots/BOS_current_vs_time_L{L_input}_V{V_input:.1f}_B{B_input:.1f}_P{P_input:.2f}_T{steps_str}_N{trajs_str}_{K_type}.pdf", dpi=400, bbox_inches ='tight')
    fig.savefig(f"bosonic_runs/plots/BOS_current_vs_time_L{L_input}_V{V_input:.1f}_B{B_input:.1f}_P{P_input:.2f}_T{steps_str}_X{X_input:03}_N{trajs_str}_{K_type}.pdf", dpi=500, bbox_inches ='tight')


if False: ####################################### Diagonal Density Profile (Fin or Avg) ####################################################################    
    data_type = "Fin" #"Avg"

    distance_on_diagonal = np.zeros((L_input*L_input,))
    for i in range(L_input*L_input):
        x = i % L_input
        y = i // L_input
        distance_on_diagonal[i] = x+y
    
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharex=True, layout='constrained', 
                           gridspec_kw=dict( wspace=0., hspace = 0.)
                        )
    



    for pl_num, B_loop in enumerate(B_list): # to have all the B vlues ploted in one plot
    
        new_data = np.load(folder_path + f"Data_L{L_input}_V{V_input:.1f}_B{B_loop:.3f}_P{P_input:.2f}_T{steps_str}_N{trajs_str}_{K_type}_dict.npy",  allow_pickle=True).item()
        
        n_data = new_data['density'][0]
        # n_data_std = new_data['density'][1]

        y_data = n_data[-1] if data_type == 'Fin' else np.mean(n_data[-5:], axis=0)  

        ax.scatter(distance_on_diagonal, y_data, label=rf'B=$2\pi\times{B_loop/(2*np.pi):.1f} $', marker='o')#, linestyle='--', linewidths=0.8)
        

    ax.set_xlabel(r"Diagonal distance $r=(x+y)$", fontsize=15)
    ax.set_ylabel(r"Density $\langle n_r \rangle$", fontsize=15)
    ax.set_title(f"bosons: {L_input}x{L_input}, dt={dtt}, p={P_input}, V={V_input} and random initial state")
    ax.legend()
    ax.grid(which='major', linestyle=':', linewidth=0.7, zorder=-10)

    # plt.tight_layout()
    fig.savefig(f"bosonic_runs/plots/BOS_diag_density_profile_L{L_input}_V{V_input:.1f}_Bs_P{P_input:.2f}_T{steps_str}_N{trajs_str}_{K_type}.pdf", dpi=400, bbox_inches ='tight')
    # fig.savefig(f"bosonic_runs/plots/BOS_diag_density_profile_L{L_input}_V{V_input:.1f}_Bs_P{P_input:.2f}_T{steps_str}_X{X_input:03}_N{trajs_str}_{K_type}.pdf", dpi=500, bbox_inches ='tight')




if False: ####################################### Density Deviation LOG plot for all inputs ####################################################################
     
    def distinct_colors(n):
        cmap = plt.get_cmap('jet_r') #('viridis')
        return [cmap(i / (n - 1)) for i in range(n)]
    
    tstep_str, ntraj_str = '4e2', '2e4'
    
    data = np.load(folder_path + f"Data_L{L_input}_V{V_input:.1f}_P{P_input:.2f}_T{tstep_str}_N{ntraj_str}_{K_type}_dict.npy",  allow_pickle=True).item()
    
    N_data = data['density'][0]
    # N_data_std = data['density'][1]
    
    dtt = data['dt']
    max_t = data['time_steps']
     
    max_time = dtt * max_t
    
    t_data = np.linspace(0, max_time, num = time_steps)

    # sem = 5* np.sqrt( np.abs(N_data_std - N_data**2 )) / np.sqrt(num_traj, dtype=np.float64) #* np.sqrt(num_traj/(num_traj-1) )    
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharex=True, layout='constrained', 
                           gridspec_kw=dict( wspace=0., hspace = 0.)
                        )
    

    colors = distinct_colors(L_input*L_input) 
    for r in range(L_input*L_input):    
        ax.plot(t_data, np.abs( N_data[:,r] - 0.5), label=f'j={r}', color=colors[r], linewidth=1.8)
        # ax.fill_between(t_data, N_data[:,r] + sem[:,r], N_data[:,r] - sem[:,r], color=colors[r], alpha=0.2) #, linewidth=1.5)

    ax.set_xscale('log')
    ax.set_title(f"log-density for p={P_input}, V={V_input} (bosons)")
    ax.set_ylabel("Occupation deviation from 0.5",fontsize=14)
    ax.set_xlabel('time',fontsize=15)
    
    ax.legend(ncol=Lx, fontsize=9)
    ax.grid(which='major', linestyle=':', linewidth=0.8)
    
    # fig.savefig(f"bosonic_runs/plots/BOS_N_evolution_L{L_input}_V{V_input:.1f}_P{P_input:.2f}_T{tstep_str}_N{ntraj_str}_{K_type}.pdf", dpi=500, bbox_inches ='tight')
    fig.savefig(f"bosonic_runs/plots/BOS_log_density_L{L_input}_V{V_input:.1f}_P{P_input:.2f}_T{tstep_str}_N{ntraj_str}_{K_type}.pdf", dpi=500, bbox_inches ='tight')
      
    
