import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from blackjack import ModifiedBlackjackEnv
from collections import defaultdict

class MonteCarloControl:
    """
    Monte Carlo Control with epsilon-greedy exploration.
    Uses time-varying step-size and exploration rate.
    """
    
    def __init__(self, N0=100):
        """
        Initialize Monte Carlo Control.
        
        Args:
            N0: Constant for epsilon-greedy exploration schedule
        """
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
            # Explore: random action
            return np.random.randint(2)
        else:
            # Exploit: greedy action
            return np.argmax(self.Q[state])
    
    def generate_episode(self):
        """
        Generate an episode using current policy.
        
        Returns:
            episode: List of (state, action, reward) tuples
        """
        episode = []
        state = self.env.reset()
        done = False
        
        while not done:
            # Get epsilon for current state
            epsilon = self.get_epsilon(state)
            
            # Select action
            action = self.get_action(state, epsilon)
            
            # Take action
            next_state, reward, done = self.env.step(action)
            
            # Store transition
            episode.append((state, action, reward))
            
            if not done:
                state = next_state
        
        return episode
    
    def update_Q_values(self, episode):
        """
        Update Q-values using Monte Carlo update with first-visit.
        
        Args:
            episode: List of (state, action, reward) tuples
        """
        # Calculate return (no discounting, gamma=1)
        G = sum([reward for _, _, reward in episode])
        
        # Track which state-action pairs we've already updated (first-visit)
        visited = set()
        
        for state, action, _ in episode:
            sa_pair = (state, action)
            
            # First-visit Monte Carlo
            if sa_pair not in visited:
                visited.add(sa_pair)
                
                # Update visit counts
                self.N_s[state] += 1
                self.N_sa[state][action] += 1
                
                # Calculate step-size
                alpha = 1.0 / self.N_sa[state][action]
                
                # Update Q-value
                self.Q[state][action] += alpha * (G - self.Q[state][action])
    
    def train(self, num_episodes=500000, print_every=50000):
        """
        Train using Monte Carlo Control.
        
        Args:
            num_episodes: Number of episodes to train
            print_every: Print progress every N episodes
        """
        print(f"Training Monte Carlo Control for {num_episodes} episodes...")
        print(f"N0 = {self.N0}")
        print()
        
        for episode_num in range(1, num_episodes + 1):
            # Generate episode
            episode = self.generate_episode()
            
            # Update Q-values
            self.update_Q_values(episode)
            
            # Print progress
            if episode_num % print_every == 0:
                avg_return = np.mean([sum([r for _, _, r in self.generate_episode()]) 
                                     for _ in range(1000)])
                print(f"Episode {episode_num}: Avg return over 1000 episodes = {avg_return:.3f}")
        
        print("\nTraining complete!")
    
    def get_optimal_value_function(self):
        """
        Extract optimal value function V*(s) = max_a Q*(s,a).
        
        Returns:
            V: Dictionary mapping states to optimal values
        """
        V = {}
        for state in self.Q:
            V[state] = np.max(self.Q[state])
        return V
    
    def plot_value_function(self):
        """
        Plot the optimal value function as a 3D surface.
        Similar to Sutton & Barto Figure 5.2.
        """
        # Get optimal value function
        V = self.get_optimal_value_function()
        
        # Create grid for plotting
        dealer_cards = range(1, 14)  # 1-13
        player_sums = range(1, 22)   # 1-21
        
        # Create meshgrid
        X, Y = np.meshgrid(dealer_cards, player_sums)
        Z = np.zeros_like(X, dtype=float)
        
        # Fill in values
        for i, player_sum in enumerate(player_sums):
            for j, dealer_card in enumerate(dealer_cards):
                state = (dealer_card, player_sum)
                if state in V:
                    Z[i, j] = V[state]
                else:
                    Z[i, j] = 0  # Unvisited states
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot surface
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', 
                              edgecolor='none', alpha=0.8)
        
        # Labels and title
        ax.set_xlabel('Dealer Showing', fontsize=12)
        ax.set_ylabel('Player Sum', fontsize=12)
        ax.set_zlabel('Value', fontsize=12)
        ax.set_title('Optimal Value Function V*(s)', fontsize=14, fontweight='bold')
        
        # Set axis limits
        ax.set_xlim(1, 13)
        ax.set_ylim(1, 21)
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Adjust viewing angle
        ax.view_init(elev=30, azim=45)
        
        plt.tight_layout()
        plt.savefig('optimal_value_function.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Plot saved as 'optimal_value_function.png'")
    
    def get_optimal_policy(self):
        """
        Extract optimal policy from Q-values.
        
        Returns:
            policy: Dictionary mapping states to optimal actions
        """
        policy = {}
        for state in self.Q:
            policy[state] = np.argmax(self.Q[state])
        return policy
    
    def print_policy_sample(self):
        """Print a sample of the optimal policy."""
        policy = self.get_optimal_policy()
        
        print("\nSample of Optimal Policy (0=stick, 1=hit):")
        print("=" * 50)
        
        # Show policy for dealer card = 7 (middle value)
        dealer_card = 7
        print(f"\nWhen dealer shows {dealer_card}:")
        for player_sum in range(12, 22):
            state = (dealer_card, player_sum)
            if state in policy:
                action = policy[state]
                action_name = "stick" if action == 0 else "hit"
                print(f"  Player sum {player_sum:2d}: {action_name}")


# Run Monte Carlo Control
if __name__ == "__main__":
    # Create and train agent
    mc = MonteCarloControl(N0=100)
    mc.train(num_episodes=500000, print_every=50000)
    
    # Print sample policy
    mc.print_policy_sample()
    
    # Plot value function
    print("\nGenerating plot...")
    mc.plot_value_function()
    
    # Save Q-values for later use in Sarsa
    import pickle
    with open('optimal_Q_values.pkl', 'wb') as f:
        pickle.dump(dict(mc.Q), f)
    print("\nOptimal Q-values saved to 'optimal_Q_values.pkl'")