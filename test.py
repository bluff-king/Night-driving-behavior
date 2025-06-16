import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class NightDrivingModel:
    def __init__(self, N=150, L=500, dt=0.1):
        """
        Initialize the night driving car-following model
        
        Parameters:
        N: Number of vehicles
        L: System length (periodic boundary)
        dt: Time step
        """
        self.N = N
        self.L = L
        self.dt = dt
        
        # Model parameters from paper
        self.x_c = 2
        self.x_c1 = 3.2
        self.x_c2 = 4
        self.a = 5
        self.b = 1
        
        # Initialize positions and velocities
        self.positions = np.linspace(0, L, N, endpoint=False)
        self.velocities = np.ones(N)  # Initial velocity = 1
        
        # Perturbation parameters
        self.perturbed_vehicle = 0  # Index of vehicle to perturb
        self.perturbation_active = False
        self.perturbation_counter = 0
        
    def normal_velocity(self, Dx):
        """Normal optimal velocity function (Equation 5)"""
        return np.tanh(Dx - self.x_c) + np.tanh(self.x_c)
    
    def night_velocity(self, Dx):
        """Night traffic optimal velocity function (Equation 6)"""
        V = np.zeros_like(Dx)
        
        # For Dx < x_c1
        mask1 = Dx < self.x_c1
        V[mask1] = np.tanh(Dx[mask1] - self.x_c) + np.tanh(self.x_c)
        
        # For x_c1 <= Dx < x_c2
        mask2 = (Dx >= self.x_c1) & (Dx < self.x_c2)
        V[mask2] = self.a - Dx[mask2]
        
        # For Dx >= x_c2
        mask3 = Dx >= self.x_c2
        V[mask3] = self.b
        
        return V
    
    def velocity_derivative(self, Dx):
        """Derivative of night velocity function for stability analysis"""
        V_prime = np.zeros_like(Dx)
        
        # For Dx < x_c1: derivative of tanh(Dx - x_c) + tanh(x_c)
        mask1 = Dx < self.x_c1
        V_prime[mask1] = 1 - np.tanh(Dx[mask1] - self.x_c)**2
        
        # For x_c1 <= Dx < x_c2: derivative of (a - Dx)
        mask2 = (Dx >= self.x_c1) & (Dx < self.x_c2)
        V_prime[mask2] = -1  # This is where V' < 0, causing instability
        
        # For Dx >= x_c2: derivative of constant b
        mask3 = Dx >= self.x_c2
        V_prime[mask3] = 0
        
        return V_prime
    
    def get_headways(self):
        """Calculate headways (spacing between consecutive vehicles)"""
        headways = np.zeros(self.N)
        for i in range(self.N):
            # Vehicle i follows vehicle (i-1)
            # Headway of vehicle i = position of vehicle (i-1) - position of vehicle i
            prev_idx = (i - 1) % self.N
            headway = self.positions[prev_idx] - self.positions[i]
            if headway <= 0:
                headway += self.L  # Periodic boundary condition
            headways[i] = headway
        return headways
    
    def apply_perturbation(self, n_dec=1, deceleration=1.0):
        """
        Apply perturbation to one vehicle
        
        Parameters:
        n_dec: Number of time steps for perturbation
        deceleration: Deceleration rate
        """
        self.perturbation_active = True
        self.perturbation_counter = 0
        self.n_dec = n_dec
        self.deceleration = deceleration
        self.m_dec = 0  # Counter for when vehicle reaches zero velocity
        self.vehicle_stopped = False
    
    def update_with_perturbation(self, kappa, lambda_param):
        """Update system with perturbation logic"""
        headways = self.get_headways()
        
        # Calculate accelerations using FVD model
        accelerations = np.zeros(self.N)
        
        for i in range(self.N):
            # Vehicle i follows vehicle (i-1)
            prev_idx = (i - 1) % self.N
            headway = headways[i]
            
            # FVD model acceleration (Equation 1)
            # dv_{i}/dt = κ[V(Δx) - v_i] + λ(v_{i-1} - v_i)
            optimal_vel = self.night_velocity(np.array([headway]))[0]
            velocity_diff = self.velocities[prev_idx] - self.velocities[i]
            
            accelerations[i] = (kappa * (optimal_vel - self.velocities[i]) + 
                              lambda_param * velocity_diff)
        
        # Apply perturbation to specific vehicle
        if self.perturbation_active and self.perturbation_counter < self.n_dec:
            if not self.vehicle_stopped:
                # Apply constant deceleration
                accelerations[self.perturbed_vehicle] = -self.deceleration
                
                # Check if vehicle will reach zero velocity
                new_velocity = (self.velocities[self.perturbed_vehicle] + 
                              accelerations[self.perturbed_vehicle] * self.dt)
                
                if new_velocity <= 0:
                    self.velocities[self.perturbed_vehicle] = 0
                    self.vehicle_stopped = True
                    self.m_dec = self.perturbation_counter + 1
            else:
                # Vehicle remains stopped for remaining time steps
                accelerations[self.perturbed_vehicle] = 0
                self.velocities[self.perturbed_vehicle] = 0
            
            self.perturbation_counter += 1
            
            if self.perturbation_counter >= self.n_dec:
                self.perturbation_active = False
        
        # Update velocities (except for stopped vehicle during perturbation)
        for i in range(self.N):
            if not (self.perturbation_active and i == self.perturbed_vehicle and self.vehicle_stopped):
                self.velocities[i] += accelerations[i] * self.dt
                self.velocities[i] = max(0, self.velocities[i])  # Ensure non-negative velocity
        
        # Update positions
        for i in range(self.N):
            self.positions[i] += self.velocities[i] * self.dt + 0.5 * accelerations[i] * self.dt**2
            self.positions[i] = self.positions[i] % self.L  # Periodic boundary
    
    def check_stability(self, kappa, lambda_param):
        """Check linear stability condition"""
        headways = self.get_headways()
        V_prime = self.velocity_derivative(headways)
        stability_threshold = kappa/2 + lambda_param
        
        # Traffic is unstable where V' >= stability_threshold
        unstable_mask = V_prime >= stability_threshold
        
        return {
            'headways': headways,
            'V_prime': V_prime,
            'stability_threshold': stability_threshold,
            'unstable_regions': unstable_mask,
            'is_stable': np.all(V_prime < stability_threshold)
        }
    
    def simulate(self, steps, kappa=1.0, lambda_param=0.5, apply_perturbation=True, n_dec=1):
        """
        Run simulation
        
        Parameters:
        steps: Number of simulation steps
        kappa, lambda_param: FVD model parameters
        apply_perturbation: Whether to apply perturbation
        n_dec: Perturbation duration
        """
        # Store trajectory data
        position_history = np.zeros((steps, self.N))
        velocity_history = np.zeros((steps, self.N))
        headway_history = np.zeros((steps, self.N))
        
        # Apply initial perturbation
        if apply_perturbation:
            self.apply_perturbation(n_dec=n_dec)
        
        for step in range(steps):
            # Update system
            self.update_with_perturbation(kappa, lambda_param)
            
            # Store data
            position_history[step] = self.positions.copy()
            velocity_history[step] = self.velocities.copy()
            headway_history[step] = self.get_headways()
        
        return {
            'positions': position_history,
            'velocities': velocity_history,
            'headways': headway_history
        }
    
    def plot_space_time(self, position_history, title="Space-Time Plot"):
        """Plot space-time diagram"""
        plt.figure(figsize=(12, 8))
        
        # Plot trajectories
        steps, N = position_history.shape
        time_steps = np.arange(steps) * self.dt
        
        for i in range(N):
            plt.plot(position_history[:, i], time_steps, 'b-', alpha=0.7, linewidth=0.5)
        
        plt.xlabel('Position')
        plt.ylabel('Time')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def analyze_clusters(self, headway_history, velocity_history):
        """Analyze cluster formation"""
        final_headways = headway_history[-1]
        final_velocities = velocity_history[-1]
        
        # Identify clusters (vehicles with headways in unstable range)
        cluster_vehicles = []
        for i in range(self.N):
            if self.x_c1 <= final_headways[i] <= self.x_c2:
                cluster_vehicles.append(i)
        
        # Find leading vehicles (headways > x_c2)
        leading_vehicles = []
        for i in range(self.N):
            if final_headways[i] > self.x_c2:
                leading_vehicles.append(i)
        
        print(f"Vehicles in clusters (x_c1 < headway < x_c2): {len(cluster_vehicles)}")
        print(f"Leading vehicles (headway > x_c2): {len(leading_vehicles)}")
        print(f"Average velocity of leading vehicles: {np.mean([final_velocities[i] for i in leading_vehicles]):.3f}")
        print(f"Expected velocity for leading vehicles: {self.b}")
        
        return {
            'cluster_vehicles': cluster_vehicles,
            'leading_vehicles': leading_vehicles,
            'final_headways': final_headways,
            'final_velocities': final_velocities
        }

# Example usage and demonstrations
def demonstrate_night_driving():
    """Demonstrate different scenarios from the paper"""
    
    # Scenario 1: Small perturbation with kappa=1.0, lambda=0.5
    print("=== Scenario 1: Small Perturbation (n_dec=1), kappa=1.0, lambda=0.5 ===")
    model1 = NightDrivingModel(N=150, L=500)
    
    # Check stability condition
    stability = model1.check_stability(kappa=1.0, lambda_param=0.5)
    print(f"Stability threshold: {stability['stability_threshold']:.3f}")
    print(f"System is stable: {stability['is_stable']}")
    
    # Run simulation
    results1 = model1.simulate(steps=1000, kappa=1.0, lambda_param=0.5, n_dec=1)
    
    # Analyze results
    clusters1 = model1.analyze_clusters(results1['headways'], results1['velocities'])
    model1.plot_space_time(results1['positions'], 
                          "Small Perturbation: kappa=1.0, lambda=0.5")
    
    print("\n" + "="*60 + "\n")
    
    # Scenario 2: Small perturbation with kappa=1.0, lambda=0.2 (kink-antikink waves)
    print("=== Scenario 2: Small Perturbation (n_dec=1), kappa=1.0, lambda=0.2 ===")
    model2 = NightDrivingModel(N=250, L=500)
    
    stability2 = model2.check_stability(kappa=1.0, lambda_param=0.2)
    print(f"Stability threshold: {stability2['stability_threshold']:.3f}")
    
    results2 = model2.simulate(steps=1500, kappa=1.0, lambda_param=0.2, n_dec=1)
    clusters2 = model2.analyze_clusters(results2['headways'], results2['velocities'])
    model2.plot_space_time(results2['positions'], 
                          "Kink-Antikink Waves: kappa=1.0, lambda=0.2")
    
    print("\n" + "="*60 + "\n")
    
    # Scenario 3: Large perturbation
    print("=== Scenario 3: Large Perturbation (n_dec=80), kappa=1.0, lambda=0.5 ===")
    model3 = NightDrivingModel(N=220, L=500)
    
    results3 = model3.simulate(steps=2000, kappa=1.0, lambda_param=0.5, n_dec=80)
    clusters3 = model3.analyze_clusters(results3['headways'], results3['velocities'])
    model3.plot_space_time(results3['positions'], 
                          "Large Perturbation: One Cluster Formation")

# Run demonstration
if __name__ == "__main__":
    demonstrate_night_driving()