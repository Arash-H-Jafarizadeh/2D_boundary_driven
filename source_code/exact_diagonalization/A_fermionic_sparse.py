import numpy as np
import scipy.sparse as sparse
import functools as ft



















plotting = True
saving = not plotting


dt = 0.5
Nx = 3
Ny = 3
N = Nx * Ny

p = 0.5
V = 1.0
drive_type = "current"  # "current", "dephasing"
particle_type = "bosons"  # "fermions", "bosons"

num_iterations = 200
batch_size = 10
steps = 20
initial_state = "random"  # "checkerboard", "empty", "random", "custom"

even_parity = False  # Only used for random state
occupation_list = [0,1,0,1,1,0,1,0,0]  # Only used for custom state

t_list = np.linspace(0, steps*dt, steps+1)



if not drive_type in ["current", "dephasing"]:
    raise ValueError(f"Invalid drive_type: {drive_type}")

if not initial_state in ["checkerboard", "empty", "random", "custom"]:
    raise ValueError(f"Invalid initial_state: {initial_state}")


# Extract currents
bonds = []
for y in range(Ny):
    for x in range(Nx-1):
        n1 = x % Nx + y*Nx
        n2 = (x+1) % Nx + y*Nx

        if not (n1==0) and not (n2==Nx*Ny-1):
            bonds.append((n1, n2))

for x in range(Nx):
    for y in range(Ny-1):
        n1 = x % Nx + y*Nx
        n2 = x % Nx + (y+1)*Nx

        if not (n1==0) and not (n2==Nx*Ny-1):
            bonds.append((n1, n2))




X = csr_matrix([[0, 1], [1, 0]])
Y = csr_matrix([[0, -1j], [1j, 0]])
Z = csr_matrix([[1, 0], [0, -1]])
Identity = eye(2)
c_local = csr_matrix([[0,1],[0,0]])
cdag_local = csr_matrix([[0,0],[1,0]])


def c(j):
    op_list = []
    if particle_type == "fermions":
        op_list = [Z]*(j) + [c_local] + [Identity]*(N-j-1)
    elif particle_type == "bosons":
        op_list = [Identity]*(j) + [c_local] + [Identity]*(N-j-1)

    return ft.reduce(kron, op_list)


def cdag(j):
    op_list = []
    if particle_type == "fermions":
        op_list = [Z]*(j) + [cdag_local] + [Identity]*(N-j-1)
    elif particle_type == "bosons":
        op_list = [Identity]*(j) + [cdag_local] + [Identity]*(N-j-1)

    return ft.reduce(kron, op_list)


def hamiltonian():
    H = csr_matrix((2**N, 2**N))

    # horizontal 
    for y in range(Ny):
        for x in range(Nx-1):
            n1 = x % Nx + y * Nx 
            n2 = (x + 1) % Nx + y * Nx

            # hopping
            H += cdag(n1) @ c(n2) + cdag(n2) @ c(n1)

            # nearest neighbor interactions
            if V != 0:
                H += V * cdag(n1) @ c(n1) @ cdag(n2) @ c(n2)

    # vertical hopping
    for x in range(Nx):
        for y in range(Ny-1):
            n1 = x % Nx + y * Nx 
            n2 = x % Nx + (y + 1) * Nx

            # hopping
            H += cdag(n1) @ c(n2) + cdag(n2) @ c(n1)

            # nearest neighbor interactions
            if V != 0:
                H += V * cdag(n1) @ c(n1) @ cdag(n2) @ c(n2)

    print("Built Hamiltonian")

    return H

def initial_state_vector(occupation_list):

    state = 1.0
    for occ in occupation_list:
        if occ == 0:
            state = np.kron(state, np.array([1, 0]))
        elif occ == 1:
            state = np.kron(state, np.array([0, 1]))

    return state.reshape((2**N,))


def checkerboard_state():
    occupation_list = []
    # Create a checkerboard pattern of occupations
    for y in range(Ny):
        for x in range(Nx):
            if (x + y) % 2 == 1:
                occupation_list.append(1)  # filled site
            else:
                occupation_list.append(0)  # empty site

    return initial_state_vector(occupation_list)


def empty_state():
    occupation_list = [0] * N
    return initial_state_vector(occupation_list)


def random_state():
    occupation_list = np.random.choice([0, 1], size=N)
    if even_parity:
        occupation_list[-1] = 1 - occupation_list[-1] # Ensure even parity by flipping the last site if necessary
    return initial_state_vector(occupation_list)


def custom_state(occupation_list):
    return initial_state_vector(occupation_list)


def pick_kraus(state, site):

    n = np.vdot(state, cdag(site) @ c(site) @ state).real

    probs = np.cumsum([1-p, p*(1-n), p*n])

    coin = np.random.rand(1)

    K = sum(coin > probs)

    return K


def trajectory(procid, data):

    H = data["H"].copy()

    if initial_state == "random":
        state = random_state()
    else:
        state = data["state"].copy()

    # state = state_checkerboard.copy()

    K_list = np.zeros((steps,), dtype=int)
    n_list = np.zeros((steps+1, N), dtype=float)
    currents_list = np.zeros((steps+1, len(bonds)), dtype=complex)

    n_list[0] = [np.vdot(state, cdag(i) @ c(i) @ state).real for i in range(N)]
    currents_list[0] = [(-1j*np.vdot(state, cdag(n2) @ (c(n1) @ state) - cdag(n1) @ (c(n2) @ state))).real for n1, n2 in bonds]
    for step in range(steps):

        state = expm_multiply(-1j * H * dt, state)

        # Kraus operator for inflow
        K_in = pick_kraus(state, 0)
        if K_in == 1:
            state = cdag(0) @ state
            if drive_type == "dephasing":
                state = c(0) @ state
            state /= np.linalg.norm(state)
        elif K_in == 2:
            state = cdag(0) @ c(0) @ state
            state /= np.linalg.norm(state)

        # Kraus operator for outflow
        K_out = pick_kraus(state, N-1)
        if K_out == 1:
            state = c(N-1) @ cdag(N-1) @ state
            state /= np.linalg.norm(state)
        elif K_out == 2:
            state = c(N-1) @ state
            if drive_type == "dephasing":
                state = cdag(N-1) @ state
            state /= np.linalg.norm(state)

        K_list[step] = K_in + 3*K_out
        n_list[step+1] = [np.vdot(state, cdag(i) @ c(i) @ state).real for i in range(N)]
        currents_list[step+1] = [(-1j*np.vdot(state, cdag(n2) @ (c(n1) @ state) - cdag(n1) @ (c(n2) @ state))).real for n1, n2 in bonds]

    trajectory_data = {"n_list": n_list, "currents_list": currents_list, "K_list": K_list}

    data["completed"] += 1
    # print(f"Finished trajectory {procid}")
    print(f"Completed {data['completed']} / {num_iterations}", end="\r") # May not give correct value!

    data[procid] = trajectory_data
    # return trajectory_data




if __name__ == "__main__":

    H = hamiltonian()

    
    state = None
    if initial_state == "checkerboard":
        state = checkerboard_state()
    elif initial_state == "empty":
        state = empty_state()
    elif initial_state == "random":
        state = random_state()
    elif initial_state == "custom":
        state = custom_state()

    manager = Manager()
    data = manager.dict()
    data["H"] = H
    data["state"] = state 
    data["completed"] = 0

    t1 = time.perf_counter()

    print(f"CPU count: {cpu_count()}")

    # with Pool(processes=cpu_count()-1) as pool:
    #     results = pool.map(trajectory, [x for x in range(num_iterations)])

    pool = Pool(processes=num_processes)
    for procid in range(num_iterations):
        pool.apply_async(trajectory, args=(procid, data))
    pool.close()
    pool.join()

    t2 = time.perf_counter()
    print(f"\n Finished all trajectories")
    print(f"Time taken (parallel): {t2 - t1} seconds")

    K_avg = 0.
    n_avg = 0.
    avg_currents = 0.
    for i in range(num_iterations):
        res = data[i]
        K_avg += res["K_list"] / num_iterations
        n_avg += res["n_list"]/num_iterations
        avg_currents += res["currents_list"] / num_iterations


    time_averaged_currents = np.mean(avg_currents[7*steps//8:-1], axis=0)
    time_averaged_n = np.mean(n_avg[7*steps//8:-1], axis=0)

    if saving:
        data = {'t_list': t_list, "n_avg": n_avg, "current_avg": avg_currents, "K_avg": K_avg}

        with open(f'data/QC_{Nx}x{Ny}_dt{dt:,.1g}_p_{p:,.1g}_steps{steps}_trajectories{num_iterations}_{initial_state}'.replace('.','-') + '.pickle', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    elif plotting:

        # Plotting the occupation numbers
        for i in range(N):
            plt.plot(t_list, n_avg[:,i], label=f"n_{i}")
        plt.plot(t_list, np.sum(n_avg, axis=1)/N, '--', c='k', label="n_avg")
        plt.xlabel("Time")
        plt.ylabel("Occupation")
        plt.legend()
        # plt.xscale("log")
        plt.show()



        # plotting the currents
        for i in range(len(bonds)):
            plt.plot(t_list, avg_currents[:,i], label=f"current_{i}")
        plt.xlabel("Time")
        plt.ylabel("Current")
        plt.legend()
        plt.show()



        # single line definition of empty lists for X, Y, U, V, C
        X = []; Y = []; U = []; V = []; C = []
        for i, bond in enumerate(bonds):
            # convert back from n to x,y coordinates
            x1, y1 = bond[0] % Nx, bond[0] // Nx
            x2, y2 = bond[1] % Nx, bond[1] // Nx

            C.append(np.abs(avg_currents[-1,i]))

            if np.real(avg_currents[-1,i]) > 0:
                X.append(x1)
                Y.append(y1)
                U.append(x2-x1)
                V.append(y2-y1)
            else:
                X.append(x2)
                Y.append(y2)
                U.append(x1-x2)
                V.append(y1-y2)



        fig, ax = plt.subplots()

        p1 = ax.quiver(X, Y, U, V, C, angles='xy', scale_units='xy', scale=1, width=0.03)
            # plt.scatter([x1, x2], [y1, y2], c=[avg_currents[-1,i]], marker='o', cmap="RdBu_r", s=1, vmin=0, vmax=1, linewidths=2)
        cb1 = plt.colorbar(p1, ax=ax)

        X = []; Y = []; C = []
        for x in range(Nx):
            for y in range(Ny):
                n = x % Nx + y*Nx
                X.append(x)
                Y.append(y)
                C.append(n_avg[-1,n])


        p2 = ax.scatter(X, Y, c=C, cmap="RdBu_r", s=500, edgecolors= "black", vmin=0, vmax=1)

        cb2 = plt.colorbar(p2, ax=ax)
        ax.set_axis_off()
        ax.set_aspect('equal')
        plt.show()



        fig, ax = plt.subplots()

        # single line definition of empty lists for X, Y, U, V, C
        X = []; Y = []; U = []; V = []; C = []
        for i, bond in enumerate(bonds):
            # convert back from n to x,y coordinates
            x1, y1 = bond[0] % Nx, bond[0] // Nx
            x2, y2 = bond[1] % Nx, bond[1] // Nx

            C.append(np.abs(time_averaged_currents[i]))

            if np.real(time_averaged_currents[i]) > 0:
                X.append(x1)
                Y.append(y1)
                U.append(x2-x1)
                V.append(y2-y1)
            else:
                X.append(x2)
                Y.append(y2)
                U.append(x1-x2)
                V.append(y1-y2)

        p1 = ax.quiver(X, Y, U, V, C, angles='xy', scale_units='xy', scale=1, width=0.03)
            # plt.scatter([x1, x2], [y1, y2], c=[avg_currents[-1,i]], marker='o', cmap="RdBu_r", s=1, vmin=0, vmax=1, linewidths=2)
        cb1 = plt.colorbar(p1, ax=ax)

        X = []; Y = []; C = []
        for x in range(Nx):
            for y in range(Ny):
                n = x % Nx + y*Nx
                X.append(x)
                Y.append(y)
                C.append(time_averaged_n[n])


        p2 = ax.scatter(X, Y, c=C, cmap="RdBu_r", s=500, edgecolors= "black", vmin=0, vmax=1)

        cb2 = plt.colorbar(p2, ax=ax)
        ax.set_axis_off()
        ax.set_aspect('equal')
        plt.show()