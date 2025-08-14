import time as tt
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys


if len(sys.argv) == 3:
    job_number = int(sys.argv[1])
    array_number = int(sys.argv[2])
if len(sys.argv) == 2:
    job_number = int(sys.argv[1])
    array_number = 0



# folder_path = "/gpfs01/home/ppzaj/python_projects/Quantinuum_Driven_Boundaries/fermionic_runs/raw_data/"
folder_path = "/gpfs01/home/ppzaj/python_projects/Quantinuum_Driven_Boundaries/fermionic_runs/raw_data/s_data/"

L_list = [4]
p_list = [0.0] #[0.25, 0.5]
V_list = [0.0]
B_list = [0.0*np.pi, 0.5*np.pi, 0.9999*np.pi]
T_list = [12]#[200, 400]
K_list = ['VAR'] #'MPO' #"InfRect" #"normal" # "Hermitian" 
X_list = [3,4,5]#, 150, 200, 250]

# array_input = [(l, p, v, b, t, k) for l in L_list for p in p_list for v in V_list for b in B_list for t in T_list for k in K_list]
# L_input, P_input, V_input, B_input, T_input, krs_type = array_input[np.mod(array_number, len(array_input))] 
array_input = [(l, p, v, b, t, k, x) for l in L_list for p in p_list for v in V_list for b in B_list for t in T_list for k in K_list for x in X_list]
L_input, P_input, V_input, B_input, T_input, krs_type, X_input = array_input[np.mod(array_number, len(array_input))] 


Lx, Ly = L_input, L_input
L = Lx * Ly

# dt = P_input

steps_str = f'{str(T_input)[0]}e{int(np.log10(T_input))}'
trajs_str = '2e3'


# data = np.load(folder_path + f"Data_L{L_input}_V{V_input:.1f}_B{B_input:.3f}_P{P_input:.2f}_T{steps_str}_N{trajs_str}_{krs_type}_dict.npy",  allow_pickle=True).item()
# data = np.load(folder_path + f"Data_L{L_input}_V{V_input:.1f}_B{B_input:.2f}_P{P_input:.2f}_T{steps_str}_X{X_input:03}_N{trajs_str}_{krs_type}_dict.npy",  allow_pickle=True).item()
data = np.load(folder_path + f"data_L{L_input}_V{V_input:.1f}_B{B_input:.3f}_P{P_input:.2f}_T{steps_str}_N{trajs_str}_{krs_type}_A{X_input:04}.npy",  allow_pickle=True).item()
    
N_data = data['density']#[0]
# N_data_std = data['density'][1]
J_data = data['current']#[0]
# J_data_std = data['current'][1]

num_traj = 2000 #data['trajectory']

dtt = P_input / 2.0 #data['dt']
# max_t = T_input #data['time_steps']
# t_data = np.linspace(0, T_input * dtt, num = T_input)
t_data = np.cumsum([1,1,1,0.5,0.5,0.5,0.25,0.25,0.25,0.1,0.1,0.1])


if False: ####################################### Density profile SnapShots #######################################################################

    N_data = np.load(folder_path + system + f"_N_avg_(3x3)_P{p_input}_V0.15.npy", allow_pickle=True)
    
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
    Fig.savefig(f"test_runs/{system}_N_TimeSnap_p{p_input}_({Lx}x{Ly}).png", dpi=600, bbox_inches = 'tight')

    
if True:################################# *IMPROVED* AVG Current SnapShots #######################################################################
    
    plt_type='Fin' #'Avg' #'Fin'
    
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


    
    # min_J, max_J = 0.0, np.max(np.abs(J_data[:,:,2]), axis=None)
    # time_steps = len(N_data)
    # time_data = np.linspace(0, max_time, num = time_steps)


    if plt_type == 'Fin':
        max_t = -1
        j_data = J_data[max_t] # 
        n_data = N_data[max_t] # 
        # min_J, max_J = np.min(np.abs(J_data[:,2]), axis=None), np.max(np.abs(J_data[:,2]), axis=None)
        title_str = rf'Last time step $t={T_input}*\delta t$ (V={V_input}, $B=\pi\times${B_input/(np.pi):.2f})'
        # title_str = rf'time $t={max_t}*\delta t$ (V={V_input}, $B=\pi\times${B_input/(np.pi):.2f})'
        # file_name_str = f"FER_FinSnap_L{L_input}_V{V_input:.1f}_B{B_input:.2f}_P{P_input:.2f}_T{steps_str}_N{trajs_str}_{krs_type}.pdf"    
        file_name_str = f"FER_FinSnap_L{L_input}_V{V_input:.1f}_B{B_input:.2f}_P{P_input:.2f}_T{steps_str}_X{X_input:03}_N{trajs_str}_{krs_type}.pdf"    
        # file_name_str = f"FER_FinSnap_L{L_input}_V{V_input:.1f}_B{B_input:.2f}_P{P_input:.2f}_T{steps_str}_t{max_t}_N{trajs_str}_{krs_type}_1.pdf"    
        
    if plt_type == 'Avg':
        j_data = np.mean(J_data[-5:], axis=0)  
        n_data = np.mean(N_data[-5:], axis=0)  
        # min_J, max_J = np.min(np.abs(J_data[:,2]), axis=None), np.max(np.abs(J_data[:,2]), axis=None)
        title_str = rf'Last 5 time steps averaged (V={V_input}, B={B_input})'
        # file_name_str = f"FER_AvgSnap_L{L_input}_V{V_input:.1f}_B{B_input:.2f}_P{P_input:.2f}_T{steps_str}_N{trajs_str}_{krs_type}.pdf"
        file_name_str = f"FER_AvgSnap_L{L_input}_V{V_input:.1f}_B{B_input:.2f}_P{P_input:.2f}_T{steps_str}_X{X_input:03}_N{trajs_str}_{krs_type}.pdf"
    
    
    
    positions = connected_pairs(Lx, Ly) 

    J_dict = {(int(i), int(j)): v for i, j, v in j_data}
    print(" J_dict form:", len(J_dict)) #print(J_dict)
    
    vertex_pos = {(i, j): (j, i) for i in range(Lx) for j in range(Ly)}  # (row, col): (x, y)
    vertex_numbers = np.arange(Lx*Ly).reshape(Ly, Lx)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 9), sharex=True, sharey=True, layout='constrained', gridspec_kw=dict( wspace=0.2,hspace = 0.
                                                                                                                   ),)

    arrow_kwargs = dict(width=0.07, head_width=0.182, head_length=0.2, length_includes_head=True, zorder=2)
 
 
    min_J, max_J = 0.0, np.max(np.abs( [J_dict[(i,j)] for i , j in positions] ), axis=None)
    J_norm = plt.Normalize(min_J, max_J) # type: ignore
    J_cmap = plt.cm.viridis #summer #YlGn_r #binary # type: ignore 

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
        if x1 == x2: ## Vertical Edges
            val = J_dict[(x,y)] #J_data[n, -1]
            color = J_cmap(J_norm(np.abs(val))) #cmap(norm(val))
            if val >= 0:
                dy = y2-y1
                ax.arrow(x1, y1+l_off, 0, dy-h_off, color=color, **arrow_kwargs)
            else:
                dy = y1-y2
                ax.arrow(x1, y2-l_off, 0, dy+h_off, color=color, **arrow_kwargs)
    
    
    for (i, j), (x, y) in vertex_pos.items():
        n_val =  n_data[i*Ly + j] #N_data[i, j]
        n_color = N_cmap(N_norm(np.abs(n_val))) 
        csm = ax.scatter(x, y, s=2800, color=n_color, edgecolor='black', linewidth=2, zorder=3)
        ax.text(x, y, str(vertex_numbers[i, j]), ha='center', va='center_baseline', fontsize=32, zorder=4)

    ax.axis('off')
    ax.set_aspect('equal')
    
        
    # Colorbar placement
    sm = plt.cm.ScalarMappable(cmap=J_cmap, norm=J_norm)
    sm.set_array([])
    J_cb = fig.colorbar(sm, ax=ax, shrink = 0.7, aspect=30)

    csm = plt.cm.ScalarMappable(cmap=N_cmap, norm=N_norm)
    csm.set_array([])
    N_cb = fig.colorbar(csm, ax=ax, cmap=N_cmap, location='bottom', orientation='horizontal', shrink=0.7, aspect=30) # type: ignore

    J_cb.ax.set_title(r'$J_{n,m}$', fontsize=14)
    N_cb.ax.set_title(r'$\langle n_i \rangle$', loc='right', y=0.9,fontsize=14)
    
    fig.text(0.9, 0.10, f'Fermions\n V={V_input}, p={P_input}', ha='center', va='center_baseline', fontsize=24, zorder=-4)


    ax.set_title(title_str,  y=0.98)
    fig.savefig("fermionic_runs/plots/" + file_name_str, dpi=500, bbox_inches ='tight')    
    

if True:####################################### N time Plots all inputs ####################################################################
    
    def distinct_colors(n):
        cmap = plt.get_cmap('jet_r') #('viridis')
        return [cmap(i / (n - 1)) for i in range(n)]
        
    
    # sem = 5* np.sqrt( np.abs(N_data_std - N_data**2 )) / np.sqrt(num_traj, dtype=np.float64) #* np.sqrt(num_traj/(num_traj-1) )    
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharex=True, layout='constrained', 
                           gridspec_kw=dict( wspace=0., hspace = 0.)
                        )
    
    colors = distinct_colors(L_input*L_input) 
    for r in range(L_input*L_input):    
        ax.plot(t_data, N_data[:,r], label=f'j={r}', color=colors[r], linewidth=1.8)
        # ax.fill_between(t_data, N_data[:,r] + sem[:,r], N_data[:,r] - sem[:,r], color=colors[r], alpha=0.2) #, linewidth=1.5)


    ax.set_title(f"Evolution of densities for p={P_input:.2f}, V={V_input:.2f}, B={B_input:.2f} (fermions).")
    ax.set_xlabel('time',fontsize=14)
    ax.set_ylabel(r'$\langle n_j \rangle$',fontsize=14)
    ax.legend(ncol=Lx, fontsize=10)
    ax.grid(which='major', linestyle=':', linewidth=0.8)

    # fig.suptitle(f"Density time evolution for {L_input}x{L_input}, p={P_input}, V={0.15} and {krs_type} kraus")
    # fig.savefig(f"fermionic_runs/plots/FER_density_vs_time_L{L_input}_V{V_input:.1f}_B{B_input:.1f}_P{P_input:.2f}_T{steps_str}_N{trajs_str}_{krs_type}.pdf", dpi=400, bbox_inches ='tight')
    fig.savefig(f"fermionic_runs/plots/FER_density_vs_time_L{L_input}_V{V_input:.1f}_B{B_input:.1f}_P{P_input:.2f}_T{steps_str}_X{X_input:03}_N{trajs_str}_{krs_type}.pdf", dpi=400, bbox_inches ='tight')
   

if True:####################################### J time Plots all p's ####################################################################
    
    def distinct_colors(n):
        cmap = plt.get_cmap('jet_r') #('viridis')
        return [cmap(i / (n - 1)) for i in range(n)]
    
    
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

    

    # sem = 5* np.sqrt( np.abs(J_data_std - J_data**2 )) / np.sqrt(num_traj) #* np.sqrt(num_traj/(num_traj-1) )    
        
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharex=True, layout='constrained')#, gridspec_kw=dict( wspace=0.0,hspace = 0.))
    
    colors = distinct_colors(2*L_input*(L_input-1)) 
    for r in range(2*L_input*(L_input-1)):
        i=int(J_data[0,r,0])
        f=int(J_data[0,r,1])
        ax.plot(t_data, J_data[:,r,2], label=f'{i}->{f}', color=colors[r], linewidth=1.8)
        # ax[0].fill_between(t_data, J_data[:,r,2] + sem[:,r,2], J_data[:,r,2] - sem[:,r,2], color=f"C{r}", alpha=0.2) #, linewidth=1.5)
        # pl_num += 1
    
    ax.set_title(f"Evolution of Currents for p={P_input:.2f}, V={V_input:.2f}, B={B_input:.3f} (fermions)")
    ax.set_ylabel(r'$\langle J_{n \rightarrow m} \rangle$',fontsize=14)
    ax.legend(ncol=Lx, fontsize=9)
    ax.grid(which='major', linestyle=':', linewidth=0.8)
    ax.set_xlabel('time',fontsize=14)

    # dig_inds = np.array([r*(Lx+1) for r in range(Lx)]+[r*(Lx+1)+1 for r in range(Lx-1)])
    # pl_num = 0
    # for r in range(2*L_input*(L_input-1)):
    #     i=int(J_data[0,r,0])
    #     f=int(J_data[0,r,1])
    #     if i in dig_inds and f in dig_inds:
    #     # ilr = r*(Lx+1)    
    #         ax[0].plot(t_data, J_data[:,r,2], label=f'{i}->{f}', color=f"C{pl_num}", linewidth=1.8)
    #         # ax[0].fill_between(t_data, J_data[:,r,2] + sem[:,r,2], J_data[:,r,2] - sem[:,r,2], color=f"C{r}", alpha=0.2) #, linewidth=1.5)
    #         pl_num += 1
    #     pl_num = 0
    #     if (i, f) in lower_diag(Lx):
    #         ax[1].plot(t_data, J_data[:,r,2], label=f'{i}->{f}', color=f"C{pl_num}", linewidth=1.8)
    #         # ax[1].fill_between(t_data, J_data[:,r,2] + sem[:,r,2], J_data[:,r,2] - sem[:,r,2], color=f"C{r}", alpha=0.2) #, linewidth=1.5)
    #         pl_num += 1
    # ax[0].set_title('next to diagonal', y=0.91)
    # ax[1].set_title('further from diagonal', y=0.99, fontsize=12)

    
    # fig.suptitle(f"Current time evolution for {Lx}x{Ly} system, p={P_input}, V={V_input} and {krs_type} kraus")
    # fig.savefig(f"fermionic_runs/plots/FER_current_vs_time_L{L_input}_V{V_input:.1f}_B{B_input:.1f}_P{P_input:.2f}_T{steps_str}_N{trajs_str}_{krs_type}.pdf", dpi=500, bbox_inches ='tight')
    fig.savefig(f"fermionic_runs/plots/FER_current_vs_time_L{L_input}_V{V_input:.1f}_B{B_input:.1f}_P{P_input:.2f}_T{steps_str}_X{X_input:03}_N{trajs_str}_{krs_type}.pdf", dpi=500, bbox_inches ='tight')


if False: ####################################### Diagonal Density Profile (Fin or Avg) ####################################################################
    
    final_data = False #True  

    distance_on_diagonal = np.zeros((L_input*L_input,))
    for i in range(L_input*L_input):
        x = i % L_input
        y = i // L_input
        distance_on_diagonal[i] = x+y
    
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharex=True, layout='constrained', 
                           gridspec_kw=dict( wspace=0., hspace = 0.)
                        )
    



    for pl_num, B_loop in enumerate(B_list): # to have all the B vlues ploted in one plot
    
        new_data = np.load(folder_path + f"Data_L{L_input}_V{V_input:.1f}_B{B_loop:.3f}_P{P_input:.2f}_T{steps_str}_N{trajs_str}_{krs_type}_dict.npy",  allow_pickle=True).item()
        
        n_data = new_data['density'][0]
        # n_data_std = new_data['density'][1]

        y_data = n_data[-1] if final_data else np.mean(n_data[-5:], axis=0)  

        ax.scatter(distance_on_diagonal, y_data, label=rf'B=$2\pi\times{B_loop/(2*np.pi):.1f} $', marker='o')#, linestyle='--', linewidths=0.8)
        

    ax.set_xlabel(r"Diagonal distance $r=(x+y)$", fontsize=15)
    ax.set_ylabel(r"Density $\langle n_r \rangle$", fontsize=15)
    ax.set_title(f"fermions: {L_input}x{L_input}, dt={dtt:.2f}, p={P_input}, V={V_input} and random initial state")
    ax.legend()
    ax.grid(which='major', linestyle=':', linewidth=0.7, zorder=-10)



    plt.tight_layout()
    fig.savefig(f"fermionic_runs/plots/FER_diag_density_profile_L{L_input}_V{V_input:.1f}_P{P_input:.2f}_T{steps_str}_N{trajs_str}_{krs_type}.pdf", dpi=500, bbox_inches ='tight')
   
    


if False: ####################################### Density Deviation LOG plot for all inputs ####################################################################
     
    def distinct_colors(n):
        cmap = plt.get_cmap('jet_r') #('viridis')
        return [cmap(i / (n - 1)) for i in range(n)]
    
    tstep_str, ntraj_str = '4e2', '2e4'
    
    data = np.load(folder_path + f"Data_L{L_input}_V{V_input:.1f}_P{P_input:.2f}_T{tstep_str}_N{ntraj_str}_{krs_type}_dict.npy",  allow_pickle=True).item()
    
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
    ax.set_title(f"log-density for p={P_input}, V={V_input} (fermions)")
    ax.set_ylabel("Occupation deviation from 0.5",fontsize=14)
    ax.set_xlabel('time',fontsize=15)
    
    ax.legend(ncol=Lx, fontsize=9)
    ax.grid(which='major', linestyle=':', linewidth=0.8)
    
    # fig.savefig(f"bosonic_runs/plots/BOS_N_evolution_L{L_input}_V{V_input:.1f}_P{P_input:.2f}_T{tstep_str}_N{ntraj_str}_{K_type}.pdf", dpi=500, bbox_inches ='tight')
    fig.savefig(f"fermionic_runs/plots/FER_log_density_L{L_input}_V{V_input:.1f}_P{P_input:.2f}_T{tstep_str}_N{ntraj_str}_{krs_type}.pdf", dpi=500, bbox_inches ='tight')
      

    


























    
if False: ####################################### Current SnapShots #######################################################################
    
    
    def connected_pairs(N):
        pairs = []
        for i in range(N):
            for j in range(N):
                label = i * N + j
                if j + 1 < N and label != 0 and i*N+j+1 != N**2-1: # right neighbor 
                    pairs.append((label, i*N+(j+1)))
                if i + 1 < N and label != 0 and (i+1)*N+j != N**2-1: # up neighbor
                    pairs.append((label, (i+1)*N+j))
        return pairs
    

    N_data = np.load(folder_path + f"N_data_L{L_input}_V{V_input:.1f}_P{P_input:.2f}_T{t_step_str}_N{n_traj_str}_{krs_type}.npy")[0]
    J_data = np.load(folder_path + f"J_data_L{L_input}_V{V_input:.1f}_P{P_input:.2f}_T{t_step_str}_N{n_traj_str}_{krs_type}.npy")[0]

    
     
    time_steps = len(N_data)
    time_data = np.linspace(0, max_time, num = time_steps)

    # NTH = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 59]
    # NTH = [0, 100, 200, 600, 700, 800, 850, 900, 925, 950, 975, 999]
    NTH = np.linspace(0, 999, 12, dtype=int)


    positions = connected_pairs(Lx) #[(0,1),(0,3),(1,2),(1,4),(2,5),(3,4),(3,6),(4,5),(4,7),(5,8),(6,7),(7,8)]
    vertex_pos = {(i, j): (j, i) for i in range(Lx) for j in range(Ly)}  # (row, col): (x, y)
    
    vertex_numbers = np.arange(Lx*Ly).reshape(Ly, Lx)
    
    fig, axs = plt.subplots(3, 4, figsize=(10, 9), sharex=True, sharey=True, layout='constrained', gridspec_kw=dict( wspace=0.15, hspace=0.05),
                            )
    #  
    # fig.subplots_adjust(wspace=10.51, hspace=1.92)
    arrow_kwargs = dict(width=0.08, head_width=0.2, head_length=0.2, length_includes_head=True, zorder=2)
 
 
    # Draw edges with color mapping
    min_J, max_J = 0.0, np.max(np.abs(J_data[:,:,2]), axis=None)
    J_norm = plt.Normalize(min_J, max_J)
    J_cmap = plt.cm.YlGn #binary #Reds #jet #viridis

    N_norm = plt.Normalize(.00, 0.99)
    N_cmap = plt.cm.RdBu_r #Blues #binary


    l_off, h_off = 0.1, 0.3
    
    idx = {i: (i % Lx, i // Ly) for i in range(Lx*Ly)}
    
    axs_list = [ axs[xx,yy] for xx in range(3) for yy in range(4)] #[axs[0,0], axs[0,1], axs[0,2], axs[1,0], axs[1,1], axs[1,2], axs[2,0], axs[2,1], axs[2,2]]
    
    for plt_num, nth in enumerate(NTH):
        print(plt_num)
        ax = axs_list[plt_num]
        for n, (x,y) in enumerate(positions):
            x1, y1 = idx[x]
            x2, y2 = idx[y]
            if y1 == y2: ## Horizontal edge
                # val = J_data[nth, x, y]
                val = J_data[nth, n, -1]
                color = J_cmap(J_norm(np.abs(val))) #cmap(norm(val))
                if val >= 0:
                    dx = x2-x1
                    ax.arrow(x1+l_off, y1, dx-h_off, 0, color=color, **arrow_kwargs)
                else:
                    dx = x1-x2
                    ax.arrow(x1+1-l_off, y1, dx+h_off, 0, color=color, **arrow_kwargs)
            if x1 == x2: ## Vertical Edges
                # val = J_data[nth, x,y]#edge_values[i, j, 1]
                val = J_data[nth, n, -1]
                color = J_cmap(J_norm(np.abs(val))) #cmap(norm(val))
                if val >= 0:
                    dy = y2-y1
                    ax.arrow(x1, y1+l_off, 0, dy-h_off, color=color, **arrow_kwargs)
                else:
                    dy = y1-y2
                    ax.arrow(x1, y2-l_off, 0, dy+h_off, color=color, **arrow_kwargs)
    
    
        for (i, j), (x, y) in vertex_pos.items():
            n_val = N_data[nth, i*Ly + j] #N_data[i, j]
            n_color = N_cmap(N_norm(np.abs(n_val))) 
            csm = ax.scatter(x, y, s=450, color=n_color, edgecolor='black', linewidth=2, zorder=3)
            ax.text(x, y, str(vertex_numbers[i, j]), ha='center', va='center_baseline', fontsize=14, zorder=4)

        ax.axis('off')
        ax.set_aspect('equal')
        ax.set_title(f't={time_data[nth]:.2f}',  y=0.97)
    
        
        # Colorbar
    sm = plt.cm.ScalarMappable(cmap=J_cmap, norm=J_norm)
    sm.set_array([])
    J_cb = fig.colorbar(sm, ax=axs, shrink = 0.7, aspect=30)

    csm = plt.cm.ScalarMappable(cmap=N_cmap, norm=N_norm)
    csm.set_array([])
    N_cb = fig.colorbar(csm, ax=axs, cmap=plt.cm.binary, location='bottom', orientation='horizontal', shrink=0.7, aspect=30)

    J_cb.ax.set_title(r'$J_{n,m}$', fontsize=14)
    N_cb.ax.set_title(r'$\langle n_i \rangle$', loc='right', y=0.9,fontsize=14)
    
    fig.text(0.93, 0.071, f"fermion \n p={P_input}", ha='center', va='center_baseline', fontsize=26, zorder=-4)

    fig.savefig(f"fermionic_runs/plots/NJ_TimeSnap_L{L_input}_V{V_input:.1f}_p{P_input:.2f}_T{t_step_str}_N{n_traj_str}_{krs_type}.png", dpi=600, bbox_inches ='tight')
    # plt.tight_layout()

    
    
if False:####################################### AVG Current SnapShots #######################################################################
    
    plt_type='Avg' #'Fin'
    tstep_str, ntraj_str = '4e2', '1e4'
    
    def connected_pairs(N):
        pairs = []
        for i in range(N):
            for j in range(N):
                label = i * N + j
                # if j + 1 < N and label != 0 and i*N+j+1 != N**2-1: # right neighbor 
                if j + 1 < N: # right neighbor 
                    pairs.append((label, i*N+(j+1)))
                # if i + 1 < N and label != 0 and (i+1)*N+j != N**2-1: # up neighbor
                if i + 1 < N: # up neighbor
                    pairs.append((label, (i+1)*N+j))
        return pairs
    

    # N_data = np.load(folder_path + f"N_data_L{L_input}_V{V_input:.1f}_P{P_input:.2f}_T{t_step_str}_N{n_traj_str}_{krs_type}.npy")[0]
    # J_data = np.load(folder_path + f"J_data_L{L_input}_V{V_input:.1f}_P{P_input:.2f}_T{t_step_str}_N{n_traj_str}_{krs_type}.npy")[0]


    data = np.load(folder_path + f"Data_L{L_input}_V{V_input:.1f}_P{P_input:.2f}_T{t_step_str}_N{n_traj_str}_{krs_type}_dict.npy",  allow_pickle=True).item()
    
    J_data = data['current'][0]
    N_data = data['density'][0]
    dtt = data['dt']
    max_t = data['time_steps']

    
    min_J, max_J = 0.0, np.max(np.abs(J_data[:,:,2]), axis=None)
     
    time_steps = len(N_data)
    time_data = np.linspace(0, max_time, num = time_steps)


    if plt_type == 'Fin':
        J_data = J_data[-1] # 
        N_data = N_data[-1] # 
        title_str = rf'Last time step $t={max_t}*\delta t$'
        file_name_str = f"FER_NJ_FinSnap_L{L_input}_V{V_input}_P{P_input:.2f}_T{tstep_str}_N{ntraj_str}_{krs_type}.pdf"    
        
    if plt_type == 'Avg':
        J_data = np.mean(J_data[-5:], axis=0)  
        N_data = np.mean(N_data[-5:], axis=0)  
        title_str = rf'Last 5 time steps averaged'
        file_name_str = f"FER_NJ_AvgSnap_L{L_input}_V{V_input}_P{P_input:.2f}_T{tstep_str}_N{ntraj_str}_{krs_type}.pdf"
    
    
    positions = connected_pairs(Lx) #[(0,1),(0,3),(1,2),(1,4),(2,5),(3,4),(3,6),(4,5),(4,7),(5,8),(6,7),(7,8)]
    vertex_pos = {(i, j): (j, i) for i in range(Lx) for j in range(Ly)}  # (row, col): (x, y)
    
    vertex_numbers = np.arange(Lx*Ly).reshape(Ly, Lx)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 9), sharex=True, sharey=True, layout='constrained', gridspec_kw=dict( wspace=0.2,hspace = 0.
                                                                                                                   ),)
    # fig.subplots_adjust(wspace=10.51, hspace=1.92)
    arrow_kwargs = dict(width=0.07, head_width=0.182, head_length=0.2, length_includes_head=True, zorder=2)
 
 
    # Draw edges with color mapping
    # min_J, max_J = 0.0, np.max(np.abs(J_data[:,2]), axis=None)
    J_norm = plt.Normalize(min_J, max_J) # type: ignore
    J_cmap = plt.cm.summer #YlGn_r #binary #viridis # type: ignore 

    N_norm = plt.Normalize(.00, 0.999) # type: ignore
    N_cmap = plt.cm.RdBu_r # type: ignore #Blues #binary


    l_off, h_off = 0.1, 0.25
    
    idx = {i: (i % Lx, i // Ly) for i in range(Lx*Ly)}
        

    for n, (x,y) in enumerate(positions):
        x1, y1 = idx[x]
        x2, y2 = idx[y]
        if y1 == y2 : ## Horizontal edge
            # val = J_data[nth, x, y]
            val = J_data[n, -1]
            color = J_cmap(J_norm(np.abs(val))) #cmap(norm(val))
            if val >= 0:
                dx = x2-x1
                ax.arrow(x1+l_off, y1, dx-h_off, 0, color=color, **arrow_kwargs)
            else:
                dx = x1-x2
                ax.arrow(x1+1-l_off, y1, dx+h_off, 0, color=color, **arrow_kwargs)
        if x1 == x2: ## Vertical Edges
            # val = J_data[nth, x,y]#edge_values[i, j, 1]
            val = J_data[n, -1]
            color = J_cmap(J_norm(np.abs(val))) #cmap(norm(val))
            if val >= 0:
                dy = y2-y1
                ax.arrow(x1, y1+l_off, 0, dy-h_off, color=color, **arrow_kwargs)
            else:
                dy = y1-y2
                ax.arrow(x1, y2-l_off, 0, dy+h_off, color=color, **arrow_kwargs)
    
    
    for (i, j), (x, y) in vertex_pos.items():
        n_val =  N_data[i*Ly + j] #N_data[i, j]
        n_color = N_cmap(N_norm(np.abs(n_val))) 
        csm = ax.scatter(x, y, s=2800, color=n_color, edgecolor='black', linewidth=2, zorder=3)
        ax.text(x, y, str(vertex_numbers[i, j]), ha='center', va='center_baseline', fontsize=32, zorder=4)

    ax.axis('off')
    ax.set_aspect('equal')
    
        
    # Colorbar placement
    sm = plt.cm.ScalarMappable(cmap=J_cmap, norm=J_norm)
    sm.set_array([])
    J_cb = fig.colorbar(sm, ax=ax, shrink = 0.7, aspect=30)

    csm = plt.cm.ScalarMappable(cmap=N_cmap, norm=N_norm)
    csm.set_array([])
    N_cb = fig.colorbar(csm, ax=ax, cmap=plt.cm.binary, location='bottom', orientation='horizontal', shrink=0.7, aspect=30) # type: ignore

    J_cb.ax.set_title(r'$J_{n,m}$', fontsize=14)
    N_cb.ax.set_title(r'$\langle n_i \rangle$', loc='right', y=0.9,fontsize=14)
    
    fig.text(0.9, 0.10, f'Fermions\n V={V_input}, p={P_input}', ha='center', va='center_baseline', fontsize=24, zorder=-4)


    ax.set_title(title_str,  y=0.98)
    fig.savefig("fermionic_runs/plots/" + file_name_str, dpi=500, bbox_inches ='tight')    
    
            