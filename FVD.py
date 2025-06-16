import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.colors as mcolors

config = {"delta_t": 0.1, "xc":2.0, "xc1":3.2, "xc2":4.0, "a":5, "b":1, "L":500}

class FVDModel:
    def __init__(self, N,k,lamb, night=False, use_stochastic=False, A=0.0,config=config,seed=None):
        # N means number of vehicles
        self.N = N
        self.k = k
        self.lamb = lamb
        self.night = night
        self.use_stochastic = use_stochastic
        self.config = config
        self.A = A
        self.dt = self.config["delta_t"]
        self.rng = np.random.default_rng(seed)

        if self.night:
            self.optimal_velocity = self.night_optimal_velocity
            self.v_max = self.night_optimal_velocity(self.config["xc1"])
        else:
            self.optimal_velocity = self.normal_optimal_velocity
            self.v_max = 5.0       # ??

    def normal_optimal_velocity(self, dx):    #dx:headway
        # Formula (5)
        xc = self.config["xc"]   # 2.0
        return np.tanh(dx - xc) + np.tanh(xc)
    
    def night_optimal_velocity(self, dx):
        # Formula (6)
        xc,xc1,xc2,a,b = self.config["xc"],self.config["xc1"],self.config["xc2"],self.config["a"],self.config["b"]
        
        # if headway too close, like normal
        if dx<xc1:      
            return np.tanh(dx - xc) + np.tanh(xc)
        elif xc1 <= dx < xc2:
            return a-dx       # (1.8 down to 1)
        else:
            return b
    
    def set_initial_conditions(self):
        """Initialize vehicles with uniform spacing"""
        L = self.config["L"]
        headway = L/self.N     # spacing
        self.positions = np.array([i * headway for i in range(self.N)])
        self.velocities = np.array([self.optimal_velocity(headway)] * self.N)
        self.accelerations = np.zeros(self.N)
        
        # Perturbation state tracking
        self.perturbed_vehicle = None
        self.ndec_remaining = 0
        self.stopped = False
        
    def apply_perturbation(self, perturbed_vehicle, ndec):
        """
        Apply perturbation to specific vehicle (here all set to leader) in ndec steps
        The perturbations are exerted as follows. In the initial homogeneous traffic, one vehicle decelerates with constant deceleration
        a = -1 within ndec time steps. Then it moves according to the FVD model. 
        If the vehicle reaches zero velocity within mdec time steps (mdec < ndec), the vehicle stops for the remaining ndec mdec time steps. 
        In this way, the magnitude of the perturbations is quantified: a small ndec corresponds to small perturbations, a large ndec corresponds to large perturbations.
        """
        self.perturbed_vehicle = perturbed_vehicle
        self.ndec_remaining = ndec
        self.stopped = False
    
    def update(self):
        new_velocities = np.zeros(self.N)
        new_positions = np.zeros(self.N)
        new_accelerations = np.zeros(self.N)
        
        # first calculate all accelerations
        for i in range(self.N):
            leader = (i-1) % self.N     # car n is ahead of car n+1
            headway = self.positions[leader] - self.positions[i]
            # periodic boundary condition
            if headway < 0:
                headway += self.config["L"]
            
            
            if i == self.perturbed_vehicle and self.ndec_remaining > 0:
                # Check if already stopped or velocity too low
                if self.stopped or self.velocities[i] <= 0.1:  # Paper's threshold cite[10]
                    new_accelerations[i] = 0
                    if not self.stopped:
                        self.stopped = True
                else:
                    new_accelerations[i] = -1.0  
            else:   
                V_dx = self.optimal_velocity(headway)
                v_leader = self.velocities[leader] 
                # Formula (1)
                new_accelerations[i] = self.k * (V_dx - self.velocities[i]) + self.lamb*(v_leader - self.velocities[i])
        for i in range(self.N):
            '''update velocities and positions for each vehicle'''
            # Formula (3)
            v_tentative = self.velocities[i] + new_accelerations[i] * self.dt
            
            # update velocities, # Handle perturbation stopping condition
            if i == self.perturbed_vehicle and self.ndec_remaining > 0:
                if v_tentative <= 0:
                    v_tentative = 0
                    self.stopped = True
            if self.use_stochastic:
                # Formlula (7)
                rand_val = self.rng.uniform(-0.5, 0.5)
                v_tentative += rand_val * self.A
                # V(3.2) is v_max like in the paper page 6
                v_new = min(max(0,v_tentative), self.v_max)
            else:
                v_new = max(0, v_tentative)
            
            # update location
            if self.use_stochastic:
                # Formula (9)
                x_new = self.positions[i] + 0.5 * (self.velocities[i] + v_new) * self.dt
            else:
                # Formula (4)
                x_new = self.positions[i] + self.velocities[i] * self.dt + 0.5 * new_accelerations[i] * (self.dt**2)
            
            # if x_new < 0: 
            #     x_new += self.config["L"]
            # if x_new >= self.config["L"]:
            #     x_new -= self.config["L"]
            x_new = x_new % self.config["L"]
                
            new_velocities[i] = v_new
            new_positions[i] = x_new
    
        if self.perturbed_vehicle is not None and self.ndec_remaining > 0:
            self.ndec_remaining -= 1

        self.velocities = new_velocities
        self.positions = new_positions
        self.accelerations = new_accelerations
        return self.positions, self.velocities
    
    def run(self, T, record_interval=1):
        positions_history = []
        velocities_history = []
        times = []
        
        for t in range(T):
            pos, vel = self.update()
            if t % record_interval == 0:      # store enough data for plotting
                positions_history.append(pos.copy())
                velocities_history.append(vel.copy())
                times.append(t * self.dt)
        return np.array(positions_history), np.array(velocities_history), np.array(times)

def plot_space_time(positions_history,times,title):
    plt.figure(figsize=(10, 6))
    for i in range(len(times)):
        plt.scatter(positions_history[i], [times[i]] * len(positions_history[i]), 
                    c='k', s=0.1, alpha=0.5)
    plt.xlabel('Position (m)')
    plt.ylabel('Time (s)')
    plt.title(title)
    plt.grid(True)
    plt.show()
    
def plot_fundamental_diagram(densities, flows, title, night_theoretical=None):
    plt.figure(figsize=(10, 6))
    plt.plot(densities, flows, 'bo-', label='Simulation')
    
    if night_theoretical is not None:
        # Plot theoretical night driving curve
        plt.plot(densities, night_theoretical, 'r--', label='Theoretical Night')
    
    plt.xlabel('Density (veh/m)')
    plt.ylabel('Flow (veh/s)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def calculate_theoretical_night_flow(density, config):
    """
    Calculate theoretical flow for night driving based on optimal velocity
    Paper doesnt point exactly but it is clear that headway is inversely proportional to density
    """
    headway = 1.0 / density if density > 0 else float('inf')
    
    xc, xc1, xc2, a, b = config["xc"], config["xc1"], config["xc2"], config["a"], config["b"]
    
    if headway < xc1:
        v_opt = np.tanh(headway - xc) + np.tanh(xc)
    elif xc1 <= headway < xc2:
        v_opt = a - headway
    else:
        v_opt = b
    
    return density * v_opt
    
if __name__ == "__main__":
    seed = 1

    print("Generating Fundamental Diagrams...")
    param_sets = [
        (1.0, 0.2, "Fig 4"),  # Figure 4
        (1.0, 0.1, "Fig 6"),  # Figure 6
    ]
    for k_val, lamb_val, fig_name in param_sets:
        densities = []
        flows = []
        theoretical_flows = []
    N_values = np.arange(20, 400, 20)  # More data points
        
    for N in N_values:
            density = N / config["L"]
            
            # Run simulation
            sim = FVDModel(N=N, k=k_val, lamb=lamb_val, night=True, use_stochastic=False, seed=seed)
            sim.set_initial_conditions()
            sim.apply_perturbation(perturbed_vehicle=0, ndec=1)  # Small perturbation

            pos_hist, vel_hist, times = sim.run(T=8000, record_interval=5)
            
            # Calculate average velocity from steady state (discard first 50% of data)
            steady_state_start = len(vel_hist) // 2
            avg_velocity = np.mean(vel_hist[steady_state_start:])
            flow = density * avg_velocity
            
            densities.append(density)
            flows.append(flow)
            
            # Calculate theoretical flow
            theoretical_flow = calculate_theoretical_night_flow(density, config)
            theoretical_flows.append(theoretical_flow)
        
    # Plot fundamental diagram
    plot_fundamental_diagram(densities, flows, 
                                f'{fig_name}: Fundamental Diagram (Night, k={k_val}, λ={lamb_val})',
                                theoretical_flows)
    
    
    # #Figure 3: Stable clusters (Night, k=1.0, λ=0.5, N=150)
    # sim = FVDModel(N=150, k=1.0,lamb=0.5,night=True, use_stochastic=False, seed=seed)
    # sim.set_initial_conditions()
    # sim.apply_perturbation(perturbed_vehicle=0, ndec=1)
    # pos_hist, vel_hist, times = sim.run(T=1000, record_interval=1)
    # plot_space_time(pos_hist, times, 'Fig 3: Stable Clusters (Night, k=1.0, λ=0.5, N=150)')
    
    # # # Figure 5: Kink-antikink waves (Night, k=1.0, λ=0.2, N=250)
    # # sim = FVDModel(N=250, k=1.0, lamb=0.2, night=True, use_stochastic=False, seed=seed)
    # # sim.set_initial_conditions()
    # # sim.apply_perturbation(perturbed_vehicle=0, ndec=1)
    # # pos_hist, vel_hist, times = sim.run(T=1000, record_interval=1)
    # # plot_space_time(pos_hist, times, 'Fig 5: Kink-Antikink Waves (Night, k=1.0, λ=0.2, N=250)')
    
    # # # Figure 7: Stable clusters (Night, k=1.0, λ=0.1, N=230)
    # # sim.set_initial_conditions()
    # # sim.apply_perturbation(perturbed_vehicle=0, ndec=1)
    # # pos_hist, vel_hist, times = sim.run(T=1000, record_interval=1)
    # # plot_space_time(pos_hist, times, 'Fig 7: Stable Clusters (Night, k=1.0, λ=0.1, N=230)')
    
    # # Figure 8a: Unstable clusters (Night, k=1.0, λ=0.1, N=300)
    # # Same parameters as Fig7 but different initial density/perturbation
    # sim = FVDModel(N=300, k=1.0, lamb=0.1, night=True,use_stochastic=False, seed=seed)
    # sim.set_initial_conditions()
    # sim.apply_perturbation(perturbed_vehicle=0, ndec=80)  # Large perturbation
    # pos_hist, vel_hist, times = sim.run(T=1000, record_interval=1)
    # plot_space_time(pos_hist, times, 'Fig 8a: Unstable Clusters (Night, k=1.0, λ=0.1, N=300)')
    
    # # # Figure 9: Randomness effects (A=0.05, 0.1, 0.2)
    # # for A in [0.01, 0.05, 0.1]:
    # #     sim = FVDModel(N=300, k=1.0, lamb=0.1, night=True, use_stochastic=True, A=A, seed=seed)
    # #     sim.set_initial_conditions()
    # #     sim.apply_perturbation(perturbed_vehicle=0, ndec=80)
    # #     pos_hist, vel_hist, times = sim.run(T=1000, record_interval=1)
    # #     plot_space_time(pos_hist, times, f'Fig 9: Randomness Effect (A={A})')
    
    # # Fundamental diagram example