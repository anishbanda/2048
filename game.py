import pygame
import random
import time
import sys
import numpy as np

# Initialize Pygame
pygame.init()

# --------------------------
# Global Configuration
# --------------------------

DEBUG = False # Enables debug printing during simulations
node_count = 0 # Counter for nodes evaluated with expectimax

# Game dimensions and screen setup
WIDTH = 400
HEIGHT = 500
screen = pygame.display.set_mode([WIDTH, HEIGHT])
pygame.display.set_caption('2048 AI')
timer = pygame.time.Clock()
fps = 60
font = pygame.font.Font('freesansbold.ttf', 24)

# Color Library
colors = {0: (204, 192, 179),
          2: (238, 228, 218),
          4: (237, 224, 200),
          8: (242, 177, 121),
          16: (245, 149, 99),
          32: (246, 124, 95),
          64: (246, 94, 59),
          128: (237, 207, 114),
          256: (237, 204, 97),
          512: (237, 200, 80),
          1024: (237, 197, 63),
          2048: (237, 194, 46),
          'light text': (249, 246, 242),
          'dark text': (119, 110, 101),
          'other': (0, 0, 0),
          'bg': (187, 173, 160)}

# --------------------------
# Game State Variables
# --------------------------

board_values = [[0 for _ in range(4)] for _ in range(4)]
game_over = False # Flag for game status
spawn_new = True # Flag to spawn new game tiles at start
init_count = 0 # Counter for intitial tile spawn
direction = '' # Current move direction
score = 0 # Current game score
init_high = 0 
high_score = init_high # Highest score recorded
memo = {} # Memoization dictionary for expectimax
move_history = [] # History of moves
arrow_map = {
    'UP': '^',
    'DOWN': 'v',
    'LEFT': '<',
    'RIGHT': '>'
} # Map move directions for UI

# --------------------------
# Helper Functions
# --------------------------


def can_move(board):
    '''
    Checks if there is any valid move left.
    Returns True if there is at least one empty cell or adjacent cells that can merge.
    '''
    for i in range(4):
        for j in range(4):
            # Move if Cell Empty
            if board[i][j] == 0:
                return True
            # Check Above
            if i > 0 and board[i][j] == board[i-1][j]:
                return True
            # Check Below
            if i < 3 and board[i][j] == board[i+1][j]:
                return True
            # Check Left
            if j > 0 and board[i][j] == board[i][j-1]:
                return True
            # Check Right
            if j < 3 and board[i][j] == board[i][j+1]:
                return True
    return False

def set_over():
    '''
    Display the 'Game Over' or 'You Win' message.
    '''
    # If any tile is 2048, assume player won
    if any(cell == 2048 for row in board_values for cell in row):
        message = "You Win!"
    else:
        message = "Game Over!"
    pygame.draw.rect(screen, 'black', [50, 50, 300, 100], 0, 10)
    game_over_text1 = font.render(message, True, 'white')
    game_over_text2 = font.render('Press Enter to Restart', True, 'white')
    screen.blit(game_over_text1, (130, 65))
    screen.blit(game_over_text2, (70, 105))

def process_move(board, direction, update_score=False):
    '''
    Process a move in the given direction on the board.
    Optionally update the score for any merge.
    
    Returns the new board state and the delta score from merges.
    '''
    score_delta = 0
    # Create a 4x4 grid to track merged tiles during this move
    merged = [[False for _ in range(4)] for _ in range(4)]
    
    if direction == 'UP':
        for j in range(4):
            # Start from second row, first row can't move up
            for i in range(1, 4):
                if board[i][j] != 0:
                    k = i
                    # Shift tile upwards until edge or other tile reached
                    while k > 0 and board[k-1][j] == 0:
                        board[k-1][j] = board[k][j]
                        board[k][j] = 0
                        k -= 1
                    # Merge With Tile
                    if k > 0 and board[k-1][j] == board[k][j] and not merged[k-1][j]:
                        board[k-1][j] *= 2
                        if update_score:
                            score_delta += board[k-1][j]
                        board[k][j] = 0
                        merged[k-1][j] = True
    
    elif direction == 'DOWN':
        for j in range(4):
            # Start from third row to top.
            for i in range(2, -1, -1):
                if board[i][j] != 0:
                    k = i
                    # Shift tile downwards
                    while k < 3 and board[k+1][j] == 0:
                        board[k+1][j] = board[k][j]
                        board[k][j] = 0
                        k += 1
                    # Merge with tile below if possible
                    if k < 3 and board[k+1][j] == board[k][j] and not merged[k+1][j]:
                        board[k+1][j] *= 2
                        if update_score:
                            score_delta += board[k+1][j]
                        board[k][j] = 0
                        merged[k+1][j] = True
    
    elif direction == 'LEFT':
        for i in range(4):
            # Process from second column to last
            for j in range(1, 4):
                if board[i][j] != 0:
                    k = j
                    # Shift tile leftwards
                    while k > 0 and board[i][k-1] == 0:
                        board[i][k-1] = board[i][k]
                        board[i][k] = 0
                        k -= 1
                    # Merge with left neighbor if possible
                    if k > 0 and board[i][k-1] == board[i][k] and not merged[i][k-1]:
                        board[i][k-1] *= 2
                        if update_score:
                            score_delta += board[i][k-1]
                        board[i][k] = 0
                        merged[i][k-1] = True
    
    elif direction == 'RIGHT':
        for i in range(4):
            # Process from third column to left
            for j in range(2, -1, -1):
                if board[i][j] != 0:
                    k = j
                    # Shift tile rightwards
                    while k < 3 and board[i][k+1] == 0:
                        board[i][k+1] = board[i][k]
                        board[i][k] = 0
                        k += 1
                    # Merge with right neighbor if possible
                    if k < 3 and board[i][k+1] == board[i][k] and not merged[i][k+1]:
                        board[i][k+1] *= 2
                        if update_score:
                            score_delta += board[i][k+1]
                        board[i][k] = 0
                        merged[i][k+1] = True
                        
    return board, score_delta

def make_move(direc, board):
    '''
    Execute the given move direction on the board.
    Updates the global score and returns the new board.
    '''
    global score
    board, delta = process_move(board, direc, update_score=True)
    score += delta
    return board

def spawn_pieces(board):
    '''
    Add a new tile (2 or 4) to a random empty cell on the board.
    Only one tile is added per call.
    There is a 90% chance of 2 and a 10% chance of 4.
    '''
    count = 0
    while any(0 in row for row in board) and count < 1:
        row = random.randint(0, 3)
        col = random.randint(0, 3)
        if board[row][col] == 0:
            count += 1
            tile = 4 if random.randint(1, 10) == 10 else 2
            board[row][col] = tile
    return board

def set_board():
    '''
    Draw the background board and display the score and high score.
    '''
    pygame.draw.rect(screen, colors['bg'], [0, 0, 400, 400], 0, 10)
    score_text = font.render(f'Score: {score}', True, 'black')
    high_score_text = font.render(f'High Score: {high_score}', True, 'black')
    screen.blit(score_text, (10, 410))
    screen.blit(high_score_text, (10, 450))

def set_pieces(board):
    '''
    Draw the game tiles on the board.
    '''
    for i in range(4):
        for j in range(4):
            value = board[i][j]
            value_color = colors['light text'] if value > 8 else colors['dark text']
            color = colors[value]
            pygame.draw.rect(screen, color, [j * 95 + 20, i * 95 + 20, 75, 75], 0, 5)
            if value > 0:
                value_len = len(str(value))
                num_font = pygame.font.Font('freesansbold.ttf', 48 - (5 * value_len))
                value_text = num_font.render(str(value), True, value_color)
                text_rect = value_text.get_rect(center=(j * 95 + 57, i * 95 + 57))
                screen.blit(value_text, text_rect)
                pygame.draw.rect(screen, 'black', [j * 95 + 20, i * 95 + 20, 75, 75], 2, 5)

def clone_board(board):
    '''
    Create and return a copy of the board.
    '''
    return [row[:] for row in board]

def simulate_move(direction, board):
    '''
    Simulate a move in the given direction without updating the global score.
    Returns the new board state after the move.
    '''
    new_board = clone_board(board)
    new_board, _ = process_move(new_board, direction, update_score=False)
    return new_board

# --------------------------
# Evaluation and Search
# --------------------------

def evaluate(board):
    '''
    Heuristic evaluation of the board.
    Combines several strategies:
        1. Snake-like weighted pattern.
        2. Bonus for empty cells.
        3. Bonus for having the highest tile in order
        4. Bonus for monotonic rows/columns
        5. Penalty for high differences between adjacent tiles.
    Also forces the highest tile to remain in the top-left corner.
    
    Returns a numerical value for the board state.
    '''
    # Strategy 1. Snake-like weights favoring the top-left corner
    weights = [
        [65536, 32768, 16384, 8192],
        [512, 1024, 2048, 4096],
        [256, 128, 64, 32],
        [2, 4, 8, 16]
    ]
    
    weighted_score = sum(board[i][j] * weights[i][j] for i in range(4) for j in range(4))
    
    # Strategy 2. Bonus for empty cells to maintain stability
    empty_cells = sum(row.count(0) for row in board)
    empty_bonus = empty_cells * 300
    
    # Strategy 3. Order bonus for having highest tiles in the right order
    cells = [cell for row in board for cell in row]
    unique_cells = sorted(list(set(cells)), reverse=True)
    if len(unique_cells) < 5:
        order_bonus = 0
    else:
        first, second, third, fourth = unique_cells[:4]
        order_bonus1 = first * 250 if board[0][0] == first else 0
        order_bonus2 = second * 250 if board[0][1] == second else 0
        order_bonus3 = third * 250 if board[0][2] == third else 0
        order_bonus4 = fourth * 250 if board[0][3] == fourth else 0
        order_bonus5 = fourth * 100 if board[1][3] == fourth else 0
        order_bonus = order_bonus1 + order_bonus2 + order_bonus3 + order_bonus4 + order_bonus5
    
    # Strategy 4. Monotonicity bonus for rows and columns that are in order
    mono_bonus = 0
    for row in board:
        if all(row[i] >= row[i+1] for i in range(3)):
            mono_bonus += sum(row)
    for j in range(4):
        col = [board[i][j] for i in range(4)]
        if all(col[i] >= col[i+1] for i in range(3)):
            mono_bonus += sum(col)
            
    # Strategy 5. Clustering penalty for large differences between adjacent tiles
    clustering_penalty = 0
    for i in range(4):
        for j in range(4):
            if board[i][j] != 0:
                for di, dj in [(0, 1), (1, 0), (1, 1), (-1, 1), (1, -1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < 4 and 0 <= nj < 4 and board[ni][nj] != 0:
                        clustering_penalty += abs(board[i][j] - board[ni][nj])
                        
    evaluation = weighted_score + empty_bonus + order_bonus + mono_bonus - clustering_penalty

    # Penalty for when last empty cell has no merge possibilities
    if empty_cells == 1:
        for i in range(4):
            for j in range(4):
                if board[i][j] == 0:
                    has_merge = False # Try to find possible merge
                    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < 4 and 0 <= nj < 4 and board[ni][nj] in [2, 4]:
                            has_merge = True
                    if not has_merge:
                        evaluation -= 100000
                    
    # Force highest tile in top-left corner.
    max_tile = max(max(row) for row in board)
    highest_tile_pos = [(i, j) for i in range(4) for j in range(4) if board[i][j] == max_tile]
    if highest_tile_pos and highest_tile_pos[0] != (0, 0):
        return -1000000             
               
    # Merge bonus for potential adjacent merges.
    merge_bonus = 0   
    for i in range(4):
        for j in range(4):
            if board[i][j] > 0:
                for di, dj in [(0, 1), (1, 0)]:
                    ni, nj = i + di, j + dj
                    if ni < 4 and nj < 4 and board[ni][nj] == board[i][j]:
                        merge_bonus += board[i][j] * 5

    return evaluation + (merge_bonus * 100)
    
def generate_chance_nodes(board):
    '''
    Simulate all possible board states resulting from adding a new tile (2 or 4).
    Returns a list of tuples (new_board, probability).
    '''
    nodes = []
    empty_cells = [(i, j) for i in range(4) for j in range(4) if board[i][j] == 0]
    if not empty_cells:
        return nodes
    prob2 = 0.9 / len(empty_cells)
    prob4 = 0.1 / len(empty_cells)
    # Probabilities for tile values
    for (i, j) in empty_cells:
        for tile, p in [(2, prob2), (4, prob4)]:
            new_board = clone_board(board)
            new_board[i][j] = tile
            nodes.append((new_board, p))
    return nodes

def expectimax(board, depth, is_max):
    '''
    Perform an expectimax search on the board.
    
     - When is_max is True, choose the move that maximizes the evaluation.
     - When is_max is False, compute the expected value over random tile additions.
     
    Uses memoization to speed up repeated computations.
    '''
    global node_count
    node_count += 1
    
    # Use board state, depth, and if is max node as key for memoization
    key = (tuple(tuple(row) for row in board), depth, is_max)
    if key in memo:
        return memo[key]
    
    # Terminal condition: maximum depth reached or no valid moves
    if depth == 0 or not can_move(board):
        value = evaluate(board)
        memo[key] = value
        return value
    
    if is_max:
        best_value = -float('inf')
        for move in ['UP', 'LEFT', 'RIGHT', 'DOWN']:
            new_board = simulate_move(move, board)
            # Only consider when board changes
            if new_board != board:
                value = expectimax(new_board, depth - 1, False)
                best_value = max(best_value, value)   
        memo[key] = best_value 
        return best_value
    else:
        expected_value = 0
        nodes = generate_chance_nodes(board)
        if not nodes:
            value = evaluate(board)
            memo[key] = value
            return value
        for (new_board, probability) in nodes:
            value = expectimax(new_board, depth - 1, True)
            expected_value += probability * value
        memo[key] = expected_value
        return expected_value
    
def get_best_move(board):
    '''
    Iterate over possible moves to select the one with the highest expectimax value.
    Uses the adaptive depth (currently based on number of empty cells).
    '''
    best_move = None
    best_value = -float('inf')
    adaptive_depth = get_adaptive_depth(board)
    # Try fixed depth and one deeper
    for d in range(adaptive_depth, adaptive_depth + 2):
        for move in ['UP', 'LEFT', 'RIGHT', 'DOWN']:
            new_board = simulate_move(move, board)
            if new_board != board:
                value = expectimax(new_board, d-1, False)
                if value > best_value:
                    best_value = value
                    best_move = move
    return best_move

def get_adaptive_depth(board):
    '''
    Determine the search depth based on the number of empty cells.
    For example, return 4 if there are more than 4 empty cells, else return 5.
    '''
    empty_cells = sum(row.count(0) for row in board)
    '''
    if empty_cells > 6:
        return 4
    elif empty_cells > 3:
        return 5
    else:
        return 6
    '''
    return 4 if empty_cells > 4 else 5

# --------------------------
# Simulation and Main Loop
# --------------------------

def simulate_game():
    '''
    Simulate a single game using the AI.
    Returns a tuple (win, score) where win is True if 2048 was reached.
    '''
    global memo, node_count, score
    memo = {}
    node_count = 0
    
    # Create new board and add two pieces
    board = [[0 for _ in range(4)] for _ in range(4)]
    board = spawn_pieces(board)
    board = spawn_pieces(board)
    local_score = 0
    game_over_sim = False
    move_count = 0
    move_history_sim = []
    move_debug_info = []
    simulation_start_time = time.time()
    
    while not game_over_sim:
        move = get_best_move(board)
        if move is None:
            break # No valid moves
        
        # Build debug info for this move.
        current_debug = f"\nMove {move_count+1}: {move}\n"
        current_debug += "Board before move:\n" + "\n".join(str(row) for row in board) + "\n"
        current_debug += f"Score before move: {local_score}\n"
            
        board_before = clone_board(board)
        board, delta = process_move(board, move, update_score=True)
        local_score += delta
        move_count += 1
        move_history_sim.append(move)
        
        current_debug += f"Delta score from move: {delta}\n"
        current_debug += "Board after move (before new piece):\n" + "\n".join(str(row) for row in board) + "\n"
        current_debug += f"Score after move: {local_score}\n"
        
        # Add new tile if board changed
        if board != board_before:
            old_board = clone_board(board)
            board = spawn_pieces(board)
                        
        move_debug_info.append(current_debug)
                            
        # Check for win
        if any(cell == 2048 for row in board for cell in row):
            current_debug += "Reached 2048! Win condition met.\n"
            simulation_time = time.time() - simulation_start_time
            return True, local_score
        
        if not can_move(board):
            game_over_sim = True

    simulation_time = time.time() - simulation_start_time
    
    # Debug info for last 20 moves
    if DEBUG:
        print("\n--- DEBUG INFO FOR LAST 20 MOVES ---")
        for info in move_debug_info[-20:]:
            print(info)
        print("\n--- SIMULATION SUMMARY ---")
        print(f"Simulation finished in {simulation_time:.2f} seconds")
        print(f"Moves: {move_count}")
        print(f"Final Score: {local_score}")
        print(f"Expectimax Nodes Evaluated: {node_count}")
        
    return False, local_score

# Run Multiple Simulations
def run_simulations(num_games=1000, depth=4):
    '''
    Run multiple game simulations and report the win rate and average score.
    '''
    wins = 0
    total_score = 0
    for i in range(num_games):
        print(f"\nStarting Simulation {i+1}")
        win, sim_score = simulate_game()
        total_score += sim_score
        if win:
            wins += 1
        result_str = "Win" if win else "Loss"
        print(f"Simulation {i+1}: {result_str}, Score: {sim_score}")
    win_rate = wins / num_games * 100
    average_score = total_score / num_games
    print(f"Simulated {num_games} games...")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Average Score: {average_score:.2f}")
    
# --------------------------
# Main Game Loop
# --------------------------

if __name__ == '__main__':
    # Run simulations if command line argument 'simulate' given
    if len(sys.argv) > 1 and sys.argv[1] == 'simulate':
        DEBUG = True
        run_simulations(num_games=10,depth=4)
    else:
        # Main loop
        ai_mode = True # Toggle between AI and Manual Gameplay
        move_delay = 10 # Delay between moves in ms
        last_move_time = pygame.time.get_ticks() # Save last time move was made
        ai_chosen_move = '' # Display AI chosen move

        run = True
        while run:
            timer.tick(fps)
            screen.fill('gray')
            set_board()
            set_pieces(board_values)
    
            # Update game logic if game not over
            if not game_over:
                if ai_mode:
                    current_time = pygame.time.get_ticks()
                    if current_time - last_move_time > move_delay:
                        best_move = get_best_move(board_values)
                        if best_move is not None:
                            direction = best_move
                            ai_chosen_move = best_move
                        last_move_time = current_time
        
                # Make move if direction given
                if direction != '':
                    # Save copy of game pre-move
                    board_copy = clone_board(board_values)
                    board_values = make_move(direction, board_values)

                    # Update move history
                    arrow = arrow_map.get(direction, '')
                    move_history.append(arrow)
                    
                    direction = ''
                    # Generate new tile if board changed
                    if board_values != board_copy:
                        board_values = spawn_pieces(board_values)
                    # Check for win
                    if any(cell == 2048 for row in board_values for cell in row):
                        game_over = True
                    # Check if possible move
                    elif not can_move(board_values):
                        game_over = True

                # Spawn new piece if game not over
                if not game_over and (spawn_new or init_count < 2):
                    board_values = spawn_pieces(board_values)
                    spawn_new = False
                    init_count += 1
                    if not can_move(board_values):
                        game_over = True

            # Display move chosen by AI
            if ai_mode and not game_over:
                small_font = pygame.font.Font('freesansbold.ttf', 15)
                max_moves_per_line = 12
                lines = [move_history[i:i + max_moves_per_line] for i in range(0, len(move_history), max_moves_per_line)]
                lines = lines[-5:]
                starting_y = 400
                line_height = 20
                for idx, line in enumerate(lines):
                    line_text = ' '.join(line)
                    rendered_line = small_font.render(line_text, True, (0, 0, 0))
                    screen.blit(rendered_line, (230, starting_y + idx * line_height))

            # Show message when game ends
            if game_over:
                set_over()
                if high_score < score:
                    high_score = score

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                if event.type == pygame.KEYUP:
                    # Allow Moves if Game Not Over
                    if not game_over:
                        if event.key == pygame.K_UP:
                            direction = 'UP'
                        elif event.key == pygame.K_DOWN:
                            direction = 'DOWN'
                        elif event.key == pygame.K_LEFT:
                            direction = 'LEFT'
                        elif event.key == pygame.K_RIGHT:
                            direction = 'RIGHT'
                    # Allow restart if game over
                    if game_over:
                        if event.key == pygame.K_RETURN:
                            board_values = [[0 for _ in range(4)] for _ in range(4)]
                            spawn_new = True
                            init_count = 0
                            score = 0
                            direction = ''
                            game_over = False

            pygame.display.flip()
    
        pygame.quit()