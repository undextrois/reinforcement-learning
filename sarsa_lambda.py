import numpy as np
import matplotlib.pyplot as plt
from blackjack import ModifiedBlackjackEnv
from collections import defaultdict
import pickle

class SarsaLambda:
    """
    Sarsa(λ) with forward-view implementation.
    No eligibility traces - uses n-step returns directly.
    """
    
    def __init__(self, lambda_value, N0=100):
        """
        Initialize Sarsa(λ).
        
        Args:
            lambda_value: Lambda parameter (0 to 1)
            N0: Constant for epsilon-greedy exploration schedule
        """
        self.lambda_value = lambda_value
        self.N0 = N0
        
        # Q-values: Q[state][action]
        self.Q = defaultdict(lambda: np.zeros(2))
        
        # Visit counts
        self.N_s = defaultdict(int)  # N(s) - state visits
        self.N_sa = defaultdict(lambda: np.zeros(2))  # N(s,a) - state-action visits
        
        self.env = ModifiedBlackjackEnv()
    
    def get_epsilon(self, state):
        """Calculate epsilon for epsilon-greedy policy."""
        return self.N0 / (self.N0 + self.N_s[state])
    
    def get_action(self, state, epsilon):
        """Select action using epsilon-greedy policy."""
        if np.random.random() < epsilon:
            return np.random.randint(2)
        else:
            return np.argmax(self.Q[state])
    
    def generate_episode(self):
        """
        Generate an episode and store trajectory.
        
        Returns:
            trajectory: List of (state, action, reward) tuples
        """
        trajectory = []
        state = self.env.reset()
        done = False
        
        while not done:
            epsilon = self.get_epsilon(state)
            action = self.get_action(state, epsilon)
            next_state, reward, done = self.env.step(action)
            
            trajectory.append((state, action, reward))
            
            if not done:
                state = next_state
        
        return trajectory
    
    def compute_lambda_return(self, trajectory, t):
        """
        Compute λ-return for time step t using forward view.
        
        G_t^λ = (1-λ) * Σ_{n=1}^{∞} λ^{n-1} * G_t^{(n)}
        
        where G_t^{(n)} is the n-step return.
        
        Args:
            trajectory: List of (state, action, reward) tuples
            t: Time step to compute return for
            
        Returns:
            lambda_return: The λ-return for time step t
        """
        T = len(trajectory)  # Episode length
        
        # For terminal episodes (no discounting, gamma=1)
        # G_t^{(n)} = R_{t+1} + R_{t+2} + ... + R_{t+n}
        
        lambda_return = 0.0
        
        for n in range(1, T - t + 1):
            # Calculate n-step return
            n_step_return = sum([trajectory[t + k][2] for k in range(n)])
            
            # Weight by (1-λ) * λ^{n-1}
            if n < T - t:
                weight = (1 - self.lambda_value) * (self.lambda_value ** (n - 1))
            else:
                # Last term: λ^{T-t-1} (no (1-λ) factor for terminal return)
                weight = self.lambda_value ** (n - 1)
            
            lambda_return += weight * n_step_return
        
        return lambda_return
    
    def train_episode(self):
        """
        Train on one episode using forward-view Sarsa(λ).
        """
        # Generate episode
        trajectory = self.generate_episode()
        
        # Update Q-values for each step using λ-return
        for t in range(len(trajectory)):
            state, action, _ = trajectory[t]
            
            # Update visit counts
            self.N_s[state] += 1
            self.N_sa[state][action] += 1
            
            # Calculate step-size
            alpha = 1.0 / self.N_sa[state][action]
            
            # Compute λ-return
            G_lambda = self.compute_lambda_return(trajectory, t)
            
            # Update Q-value
            self.Q[state][action] += alpha * (G_lambda - self.Q[state][action])
    
    def train(self, num_episodes=1000):
        """
        Train using Sarsa(λ).
        
        Args:
            num_episodes: Number of episodes to train
        """
        for episode in range(num_episodes):
            self.train_episode()
    
    def compute_mse(self, Q_star):
        """
        Compute mean squared error against optimal Q*.
        
        Args:
            Q_star: Dictionary of optimal Q-values
            
        Returns:
            mse: Mean squared error
        """
        mse = 0.0
        count = 0
        
        # Get all states that exist in Q_star
        for state in Q_star:
            for action in range(2):
                q_star_value = Q_star[state][action]
                q_value = self.Q[state][action]
                
                mse += (q_value - q_star_value) ** 2
                count += 1
        
        return mse / count if count > 0 else 0.0


def run_sarsa_experiment(lambda_values, num_episodes=1000, num_runs=10):
    """
    Run Sarsa(λ) for different λ values and compute MSE.
    
    Args:
        lambda_values: List of λ values to test
        num_episodes: Number of episodes per run
        num_runs: Number of independent runs to average
        
    Returns:
        results: Dictionary with MSE values and learning curves
    """
    # Load optimal Q-values
    print("Loading optimal Q-values...")
    with open('optimal_Q_values.pkl', 'rb') as f:
        Q_star = pickle.load(f)
    
    # Convert to defaultdict for easier access
    Q_star_default = defaultdict(lambda: np.zeros(2))
    for state, values in Q_star.items():
        Q_star_default[state] = values
    
    results = {
        'lambda_values': lambda_values,
        'mse_values': [],
        'learning_curves': {}  # For λ=0 and λ=1
    }
    
    print(f"\nRunning Sarsa(λ) experiment with {num_runs} runs per λ value...")
    print("=" * 60)
    
    for lambda_val in lambda_values:
        print(f"\nTesting λ = {lambda_val:.1f}")
        
        mse_runs = []
        learning_curves = []
        
        for run in range(num_runs):
            # Create agent
            agent = SarsaLambda(lambda_value=lambda_val, N0=100)
            
            # Track learning curve (only for λ=0 and λ=1)
            if lambda_val == 0.0 or lambda_val == 1.0:
                curve = []
                for ep in range(1, num_episodes + 1):
                    agent.train_episode()
                    
                    # Compute MSE every 50 episodes
                    if ep % 50 == 0 or ep == 1:
                        mse = agent.compute_mse(Q_star_default)
                        curve.append((ep, mse))
                
                learning_curves.append(curve)
            else:
                # Just train without tracking
                agent.train(num_episodes)
            
            # Compute final MSE
            final_mse = agent.compute_mse(Q_star_default)
            mse_runs.append(final_mse)
            
            if (run + 1) % 5 == 0:
                print(f"  Run {run + 1}/{num_runs} complete")
        
        # Average MSE across runs
        avg_mse = np.mean(mse_runs)
        results['mse_values'].append(avg_mse)
        print(f"  Average MSE: {avg_mse:.6f}")
        
        # Store learning curves for λ=0 and λ=1
        if lambda_val == 0.0 or lambda_val == 1.0:
            # Average learning curves across runs
            max_len = max(len(curve) for curve in learning_curves)
            episodes = [curve[i][0] for i in range(len(learning_curves[0]))]
            mse_avg = []
            
            for i in range(len(episodes)):
                mses = [curve[i][1] for curve in learning_curves]
                mse_avg.append(np.mean(mses))
            
            results['learning_curves'][lambda_val] = (episodes, mse_avg)
    
    return results


def plot_results(results):
    """
    Plot MSE vs λ and learning curves.
    
    Args:
        results: Dictionary with experimental results
    """
    # Plot 1: MSE vs λ
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(results['lambda_values'], results['mse_values'], 
             'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('λ', fontsize=12)
    ax1.set_ylabel('Mean Squared Error', fontsize=12)
    ax1.set_title('MSE vs λ (Sarsa)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(results['lambda_values'])
    
    # Plot 2: Learning curves for λ=0 and λ=1
    for lambda_val in [0.0, 1.0]:
        if lambda_val in results['learning_curves']:
            episodes, mse = results['learning_curves'][lambda_val]
            label = f'λ = {lambda_val:.1f}'
            ax2.plot(episodes, mse, linewidth=2, label=label, marker='o', 
                    markersize=4, markevery=5)
    
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Mean Squared Error', fontsize=12)
    ax2.set_title('Learning Curves (λ=0 and λ=1)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sarsa_lambda_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nPlot saved as 'sarsa_lambda_results.png'")


# Run the experiment
if __name__ == "__main__":
    # Lambda values to test
    lambda_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Run experiment
    results = run_sarsa_experiment(
        lambda_values=lambda_values,
        num_episodes=1000,
        num_runs=10  # Average over 10 runs for stability
    )
    
    # Plot results
    print("\nGenerating plots...")
    plot_results(results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"{'λ':<10} {'Average MSE':<15}")
    print("-" * 25)
    for lambda_val, mse in zip(results['lambda_values'], results['mse_values']):
        print(f"{lambda_val:<10.1f} {mse:<15.6f}")
    
    best_lambda = results['lambda_values'][np.argmin(results['mse_values'])]
    best_mse = min(results['mse_values'])
    print("-" * 25)
    print(f"Best λ: {best_lambda:.1f} (MSE = {best_mse:.6f})")