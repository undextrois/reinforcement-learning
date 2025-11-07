import random

class ModifiedBlackjackEnv:
    """
    Modified Blackjack environment following the assignment rules:
    - Infinite deck with replacement
    - Cards 1-13, red (prob 1/3) or black (prob 2/3)
    - Black cards add, red cards subtract
    - Player busts if sum > 21 or < 1
    - Dealer sticks on 17+
    """
    
    def __init__(self):
        self.reset()
    
    def draw_card(self):
        """Draw a card from the infinite deck."""
        value = random.randint(1, 13)
        color = 'red' if random.random() < 1/3 else 'black'
        return value, color
    
    def reset(self):
        """Start a new game. Returns initial state."""
        # Both player and dealer draw one black card (guaranteed)
        player_card = random.randint(1, 13)  # guaranteed black
        dealer_card = random.randint(1, 13)  # guaranteed black
    
        self.player_sum = player_card
        self.dealer_first_card = dealer_card
        self.done = False
    
        return (self.dealer_first_card, self.player_sum)
    
    def step(self, action):
        """
        Take an action in the environment.
        
        Args:
            action: 0 = stick, 1 = hit
            
        Returns:
            state: (dealer_first_card, player_sum) or None if terminal
            reward: -1 (loss), 0 (draw), +1 (win)
            done: True if game is over
        """
        if self.done:
            raise ValueError("Game is already over. Call reset() to start a new game.")
        
        if action == 1:  # hit
            card_value, card_color = self.draw_card()
            
            if card_color == 'black':
                self.player_sum += card_value
            else:  # red
                self.player_sum -= card_value
            
            # Check if player busts
            if self.player_sum > 21 or self.player_sum < 1:
                self.done = True
                return None, -1, True
        
        else:  # stick (action == 0)
            # Player sticks, now dealer plays
            dealer_sum = self.dealer_first_card
            
            # Dealer hits until sum >= 17
            while dealer_sum < 17:
                card_value, card_color = self.draw_card()
                
                if card_color == 'black':
                    dealer_sum += card_value
                else:  # red
                    dealer_sum -= card_value
                
                # Check if dealer busts
                if dealer_sum > 21 or dealer_sum < 1:
                    self.done = True
                    return None, 1, True
            
            # Compare sums
            self.done = True
            if self.player_sum > dealer_sum:
                return None, 1, True
            elif self.player_sum < dealer_sum:
                return None, -1, True
            else:
                return None, 0, True
        
        # Game continues after hit without bust
        return (self.dealer_first_card, self.player_sum), 0, False


# Test the environment
if __name__ == "__main__":
    print("Testing Modified Blackjack Environment\n")
    
    env = ModifiedBlackjackEnv()
    
    # Test a few games
    for game in range(3):
        print(f"Game {game + 1}:")
        state = env.reset()
        print(f"  Initial state: Dealer={state[0]}, Player sum={state[1]}")
        
        done = False
        steps = 0
        
        while not done:
            # Simple strategy: hit if sum < 15, stick otherwise
            action = 1 if state[1] < 15 else 0
            action_name = "hit" if action == 1 else "stick"
            
            next_state, reward, done = env.step(action)
            steps += 1
            
            print(f"  Step {steps}: {action_name} (player sum = {state[1]})")
            
            if not done:
                state = next_state
                print(f"    New state: Player sum={state[1]}")
            else:
                result = "win" if reward == 1 else "lose" if reward == -1 else "draw"
                print(f"    Game over! Result: {result} (reward: {reward})")
        
        print()