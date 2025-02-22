# 2048 AI Game

This repository contains an implementation of the classic 2048 game we all used to play enhanced with an AI that uses the Expectimax algorithm. The AI leverages a variety of heuristic strategies to make optimal moves, including a snake-like weighted pattern, bonuses for empty cells and monotonicity, clustering penalties, and a hard constraint to keep the highest tile locked in the top-left corner.

## Features

- **AI Player with Expectimax:**  
  The AI simulates future moves and evaluates board states using a combination of heuristics, ensuring strong play.

- **Manual and AI Modes:**  
  Play manually using the arrow keys or let the AI play automatically.

- **Simulation Mode:**  
  Run automated simulations from the command line to test performance, win rate, and debugging output.

- **Graphical Interface:**  
  The game uses Pygame to display a 4×4 board, current score, high score, and move history.

- **Clean, Commented Code:**  
  The code is organized into clear sections with detailed comments explaining each function and strategy.

## How It Works

### Game Mechanics

- **Board Setup:**  
  The game board is a 4×4 grid. At the start, two random tiles (2 or 4) are placed on the board.

- **Moves and Merging:**  
  When a move is executed (UP, DOWN, LEFT, or RIGHT), tiles slide in that direction, merge with matching neighbors, and then a new tile is spawned in a random empty cell.

### AI Algorithm

- **Expectimax Search:**  
  The AI uses an expectimax algorithm that simulates both player moves (max nodes) and random tile additions (chance nodes). A memoization technique is used to speed up repeated state evaluations.

- **Heuristic Evaluation:**  
  The evaluation function combines multiple strategies:
  - **Snake-like Weighted Pattern:**  
    Encourages high-value tiles to stay in the top-left corner.
  - **Empty Cell Bonus:**  
    Rewards states with more empty cells, helping maintain mobility.
  - **Order and Monotonicity Bonuses:**  
    Rewards boards where high-value tiles are in descending order and rows/columns decrease consistently.
  - **Clustering Penalty:**  
    Penalizes large differences between adjacent tiles.
  - **Forced Highest Tile:**  
    Ensures the highest tile remains in the top-left corner by returning a very low score if not.

- **Adaptive Depth:**  
  The AI adjusts its search depth based on the number of empty cells to balance decision quality and performance.

### Code Structure

- **Helper Functions:**  
  Functions such as `can_move()`, `clone_board()`, and drawing functions (`set_board()`, `set_pieces()`) manage the game state and GUI.

- **Move Processing:**  
  Functions like `process_move()`, `make_move()`, and `spawn_pieces()` handle moves, merging, and tile spawning.

- **Evaluation and Search:**  
  The core AI logic is implemented in `evaluate()`, `expectimax()`, and `get_best_move()`.

- **Simulation Functions:**  
  `simulate_game()` and `run_simulations()` enable automated play and debugging output to evaluate the AI's performance.

- **Main Loop:**  
  The main game loop supports both interactive play and simulation mode based on command-line arguments.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/anishbanda/2048.git
   cd 2048

2. **Run the AI:**

   ```bash
   python3 game.py

3. **Simulate the AI:**

   ```bash
   python3 game.py simulate

4. **Play 2048 Yourself:**
   Find the line in `game.py` where it says `ai_mode = True` and set it to `False`
   Before:
   ```bash
   ai_mode = True
   ```
   After:
   ```bash
   ai_mode = False
   ```
   Then, run the following line:
   ```bash
   python3 game.py
