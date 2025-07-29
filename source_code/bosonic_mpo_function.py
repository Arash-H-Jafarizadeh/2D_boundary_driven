import numpy as np # type: ignore
import tenpy # type: ignore

# sys.path.append('/gpfs01/home/ppzaj/python_projects/Quantinuum_Driven_Boundaries/source_code/') 
from basic_functions import *



def kraus_operator(indx, Ls):

    if indx == 0:#
        return [('Id',0)]#
    if indx == 1:#
        return [('Bd',0)]#
    if indx == 2:#
        return [('N',0)]#
    if indx == 3:#
        return [('B',Ls-1),('Bd',Ls-1)]# 
    if indx == 4:#
        return [('Bd',0),('B',Ls-1),('Bd',Ls-1)]#
    if indx == 5:#
        return [('N',0),('B',Ls-1),('Bd',Ls-1)]#
    if indx == 6:#
        return [('B',Ls-1)]#sps.csr_array( X_L @ N_L )
    if indx == 7:#
        return [('Bd',0),('B',Ls-1)]#
    if indx == 8:#
        return [('N',0),('B',Ls-1)]#[('Bd',0),('B',0),('B',Ls-1)]#



def currents(input_mps, dimensions, magnetic_field = 0.0):

    Lx, Ly = dimensions
    L = Lx*Ly

    peier = lambda x: np.exp( 1j * magnetic_field * x )

    indexes = lattice_edges(Lx, Ly)

    J_array = np.empty((2*L-Lx-Ly, 3)) 
    for indx, (row, x, y) in enumerate(indexes): 
        var_l = input_mps.expectation_value_term([('Bd',x),('B',y)], autoJW=True)
        var_r = input_mps.expectation_value_term([('Bd',y),('B',x)], autoJW=True)
        exp_val =  -1j*(peier(row)*var_l - peier(row).conj()* var_r ) 
        J_array[indx] = np.array([int(x), int(y), np.real( exp_val )])
        
    return( J_array )


def hopping_correlation(input_mps, dimensions):
    
    Lx, Ly = dimensions
    L = Lx * Ly
    
    C_Mat = np.zeros((L,L) , dtype=np.complex128) 
    for x in range(L):    
        for y in range(L):
            exp_val = input_mps.expectation_value_term([('Bd',x),('B',y)], autoJW=False)
            C_Mat[x,y] += exp_val
        
    return(C_Mat)


def density_correlation(input_mps, dimensions):
    
    Lx, Ly = dimensions
    L = Lx * Ly
    
    NN_Mat = np.zeros((L,L) , dtype=np.float64) 
    for x in range(L):    
        for y in range(L):
            var_N = input_mps.expectation_value('N')
            exp_x, exp_y = var_N[x], var_N[y]
            exp_val = input_mps.expectation_value_term([('N',x),('N',y)], autoJW=False)
            NN_Mat[x,y] += np.real( exp_val - exp_x * exp_y )
        
    return(NN_Mat)



def mpo_simulation(time_steps, system_dimensions, physical_couplings, **Kwargs):

    ini_config = Kwargs.get('initial_state', checkerboard(system_dimensions))
    drive_rate = Kwargs.get('driving_rate', 0.001)
    dt = Kwargs.get('dt', 0.03) 
    B_field = 0.0 #Kwargs.get('magnetic_field', 0.0)
    
    max_chi = Kwargs.get('chi_max', 100)
    min_svd = Kwargs.get('svd_min', 1.e-16)
    the_order = Kwargs.get('total_error_order', 2)
    trunc_err = Kwargs.get('max_trunc_err', 0.879)
    
    show_memory = Kwargs.get('show_memory_usage', False)
    
    
    Vs, Js = physical_couplings
    Lx, Ly = system_dimensions
    Ls = Lx*Ly


    model_params = dict(
        bc_MPS='finite', bc_y='open', bc_x='open', 
        Lx = Lx, Ly = Ly, 
        lattice='Square', # order = 'Fstyle',
        n_max = 1,
        t=Js, V=Vs, mu = 0.0, U = 0.0,
        conserve= 'parity', #'parity' 'best' 
        )

    simulation_options = dict(
        compression_method = 'SVD', #| 'variational' | 'zip_up'
        trunc_params=dict(
                chi_max = max_chi,
                svd_min = min_svd,
                ),
        max_trunc_err = trunc_err,   
        order = int(the_order),
        N_steps = 1, 
        dt = dt,
        preserve_norm=True,
        )


    mps_config = np.reshape(ini_config, (Lx,Ly,1))
    
    model = tenpy.BoseHubbardModel(model_params) 

    psi = tenpy.MPS.from_lat_product_state(model.lat, mps_config )

    time_evolver = tenpy.algorithms.mpo_evolution.ExpMPOEvolution(psi, model, simulation_options)


    if show_memory: 
        print("  ****** memory needed: ",
              tenpy.algorithms.mpo_evolution.ExpMPOEvolution.estimate_RAM(time_evolver),
              " MB")

    # mpo_N = np.zeros((time_steps, Ls), dtype=np.float64)
    # mpo_J = np.zeros((time_steps, 2*Ls-Lx-Ly, 3), dtype=np.float64)
    mpo_C = np.zeros((time_steps, Ls, Ls), dtype=np.complex128)
    mpo_NN = np.zeros((time_steps, Ls, Ls), dtype=np.float64)
    mpo_k = np.zeros(time_steps, dtype=np.int16)

    for t_indx in range(time_steps):
        
        # mpo_N[t_indx] = psi.expectation_value('N')
        # mpo_J[t_indx] = currents(psi, (Lx, Ly), magnetic_field=B_field)

        mpo_C[t_indx] = hopping_correlation(psi, (Lx, Ly))
        mpo_NN[t_indx] = density_correlation(psi, (Lx, Ly))


        time_evolver.run()
        
        n0, nL = psi.expectation_value(['N','N'], sites=[0, Ls-1])# type: ignore #
        n0nL = psi.expectation_value_term([('N',0),('N',Ls-1)], autoJW=False)

        krs_probs = [(1-drive_rate)**2, drive_rate*(1-drive_rate)*(1-n0), drive_rate*(1-drive_rate)*n0,
                    drive_rate*(1-drive_rate)*(1-nL), drive_rate*drive_rate*(1-n0-nL+n0nL), drive_rate*drive_rate*(n0-n0nL),
                    drive_rate*(1-drive_rate)*nL, drive_rate*drive_rate*(nL-n0nL), drive_rate*n0nL ]
        
        krs_rates = np.cumsum(krs_probs)
        coin = np.random.uniform(0,1)  
        index = len(krs_rates[krs_rates < coin])
        
        
        psi.apply_local_term(kraus_operator(index, Ls), autoJW=False, renormalize=True)
        mpo_k[t_indx] = index   

        # print(f"    -=-=-= step {t_indx:03} =-=-=- ")

    return(mpo_C, mpo_NN, mpo_k)#(mpo_N, mpo_J, mpo_k)# 