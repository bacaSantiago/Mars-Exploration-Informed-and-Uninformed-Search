# Import the required libraries
import copy
import heapq
import numpy as np
from collections import deque
from skimage.transform import downscale_local_mean
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import plotly.graph_objects as px


# Function to preprocess and visualize the height map of the landscape
def process_height_map(input_file, output_file):
    # Open the data file
    data_file = open(input_file, "rb")

    # Parse header information
    endHeader = False
    while not endHeader:
        line = data_file.readline().rstrip().lower()

        sep_line = line.split(b'=')
        
        if len(sep_line) == 2:
            itemName = sep_line[0].rstrip().lstrip()
            itemValue = sep_line[1].rstrip().lstrip()

            # Extract information from header
            if itemName == b'valid_maximum':
                maxV = float(itemValue)
            elif itemName == b'valid_minimum':
                minV = float(itemValue)
            elif itemName == b'lines':
                n_rows = int(itemValue)
            elif itemName == b'line_samples':
                n_columns = int(itemValue)
            elif itemName == b'map_scale':
                scale_str = itemValue.split()
                if len(scale_str) > 1:
                    scale = float(scale_str[0])

        elif line == b'end':
            endHeader = True
            # Move to the start of actual data
            char = 0
            while char == 0 or char == 32:
                char = data_file.read(1)[0] 
            pos = data_file.seek(-1, 1)      

    # Read and preprocess image data
    image_size = n_rows * n_columns
    data = data_file.read(4 * image_size)
    image_data = np.frombuffer(data, dtype=np.dtype('f'))
    image_data = image_data.reshape((n_rows, n_columns)).astype('float64')
    image_data -= minV
    image_data[image_data < -10000] = -1

    # Subsample the image
    sub_rate = round(10 / scale) 
    image_data = downscale_local_mean(image_data, (sub_rate, sub_rate))
    image_data[image_data < 0] = -1

    print('Sub-sampling:', sub_rate)
    new_scale = scale * sub_rate
    print('New scale:', new_scale, 'meters/pixel')

    # Save preprocessed map
    np.save(output_file, image_data)

    # Load preprocessed height map
    image_data = np.load(output_file)
    
    return image_data, scale, new_scale, maxV, minV, n_rows, n_columns


# Function to plot the map in 3D using Plotly
def plot3D(image_data, new_scale, maxV, minV, n_rows, n_columns):
    # Plot 3D surface
    x = np.arange(image_data.shape[1]) * new_scale
    y = np.arange(image_data.shape[0]) * new_scale
    X, Y = np.meshgrid(x, y)

    # Plot 3D surface using Plotly
    fig = px.Figure(data = px.Surface(x=X, y=Y, z=np.flipud(image_data), colorscale='hot', cmin=0, 
                                    lighting=dict(ambient=0.0, diffuse=0.8, fresnel=0.02, roughness=0.4, specular=0.2), 
                                    lightposition=dict(x=0, y=n_rows/2, z=2*maxV)),
                    layout = px.Layout(scene_aspectmode='manual', 
                                        scene_aspectratio=dict(x=1, y=n_rows/n_columns, z=max((maxV-minV)/x.max(), 0.2)), 
                                        scene_zaxis_range=[0, maxV-minV]))
    
    fig.show()


# Function to plot the map in 2D using Matplotlib
def plot2D(image_data, n_rows, n_columns, scale, new_scale, start=None, target=None, path=None, ax=None):    
    # Plot surface image
    cmap = copy.copy(plt.cm.get_cmap('autumn'))
    cmap.set_under(color='black')   

    ls = LightSource(315, 45)
    rgb = ls.shade(image_data, cmap=cmap, vmin=0, vmax=image_data.max(), vert_exag=2, blend_mode='hsv')
    
    im = ax.imshow(rgb, cmap=cmap, vmin=0, vmax=image_data.max(), 
                   extent=[0, scale * n_columns, 0, scale * n_rows], 
                   interpolation='nearest', origin='upper')
    
    # Plot initial point and path
    if start and target and path:
        start_point = ax.scatter(start[1] * new_scale, n_rows - start[0] * new_scale, color='blue', label='Start Point', marker='o', s=40)
        end_point = ax.scatter(target[1] * new_scale, n_rows - target[0] * new_scale, color='green', label='End Point', marker='o', s=40)

        path_x, path_y = zip(*path)
        path_line = ax.plot(np.array(path_y) * new_scale, n_rows - np.array(path_x) * new_scale , color='deeppink', label='Path', linewidth=2)

        # Add legends
        ax.legend(handles=[start_point, end_point, path_line[0]], loc='best', fontsize=6)

    # Add color bar for height and set title and labels
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Height (m)')
    ax.set_title('Mars surface')
    ax.set_xlabel('X (m)', fontstyle='italic')
    ax.set_ylabel('Y (m)', fontstyle='italic')


# Define a class to represent a state in the navigation problem
class NavigationState:
    def __init__(self, x, y, target, parent=None, cost=0):
        self.x = x
        self.y = y
        self.parent = parent
        self.cost = cost
        self.target = target  # Store the target position
        self.heuristic = self.heuristicCost(target)  # Precompute heuristic cost
    
    # Calculate the estimated cost of reaching the target position from this position using a heuristic
    def heuristicCost(self, target):
        # Manhattan distance between current and target positions
        return abs(self.x - target[0]) + abs(self.y - target[1])
        
    # Define state comparison based on estimated cost plus actual cost
    def __lt__(self, other):
        return self.cost + self.heuristic < other.cost + other.heuristic

    # Function to get the successors of a given state
    def getSuccessors(self, height_map, max_height_diff, target):
        successors = []
        
        # Define the possible actions including diagonal movements
        actions = [
            (self.x - 1, self.y),      # Move Up
            (self.x + 1, self.y),      # Move Down
            (self.x, self.y - 1),      # Move Left
            (self.x, self.y + 1),      # Move Right
            (self.x - 1, self.y - 1),  # Move Up-Left (Diagonal)
            (self.x - 1, self.y + 1),  # Move Up-Right (Diagonal)
            (self.x + 1, self.y - 1),  # Move Down-Left (Diagonal)
            (self.x + 1, self.y + 1)   # Move Down-Right (Diagonal)
        ]

        # Iterate over all possible actions
        for new_x, new_y in actions:
            # Check if the action is valid (new position is within the map boundaries)
            if 0 <= new_x < height_map.shape[0] and 0 <= new_y < height_map.shape[1]:
                # Check if the new position is valid (not -1) and height difference is within limit
                if height_map[new_x, new_y] != -1 and height_map[new_x, new_y] - height_map[self.x, self.y] <= max_height_diff:
                    successors.append(NavigationState(new_x, new_y, target, self))

        return successors


# Function to recover the solution path from the target state to the initial state
def getSolutionPath(state):
    path = []
    while state:
        path.append((state.x, state.y))
        state = state.parent
    return list(reversed(path))


# Function to calculate row and column from given coordinate
def calculate_row_col(x, y, new_scale, n_rows):
    row = n_rows - (y / new_scale)
    column = x / new_scale
    return (round(row), round(column))


# A* Informed Search algorithm
def astar_search(height_map, start, target, max_height_diff):
    # Define initial state and target state
    initial_state = NavigationState(start[0], start[1], target)
    target_state = NavigationState(target[0], target[1], target)

    # Initialize the A* search algorithm
    open_list = []
    closed_set = set()

    # Add the initial state to the open list
    heapq.heappush(open_list, initial_state)

    # Perform A* search until reaching the target state
    while open_list:
        current_state = heapq.heappop(open_list)

        # Check if the target state has been reached
        if (current_state.x, current_state.y) == (target_state.x, target_state.y):
            solution_path = getSolutionPath(current_state)
            return solution_path

        # Mark current state as visited
        closed_set.add((current_state.x, current_state.y))

        # Get the successors of the current state
        successors = current_state.getSuccessors(height_map, max_height_diff, target)

        # Add successors to the open list if they have not been visited
        for successor in successors:
            if (successor.x, successor.y) not in closed_set:
                successor.cost = current_state.cost + 1
                heapq.heappush(open_list, successor)

    # Return None if no solution is found
    return None, None


# Breadth-First Search algorithm
def bfs(height_map, start, target, max_height_diff):
    # Initialize queue with starting state
    queue = deque([NavigationState(start[0], start[1], target)]) 

    # Set to keep track of visited states
    visited = set()

    while queue:
        current_state = queue.popleft()  # Pop the leftmost node from the queue

        # Check if the current state is the goal state
        if (current_state.x, current_state.y) == target:
            solution_path = getSolutionPath(current_state)
            return solution_path

        # Mark the current state as visited
        visited.add((current_state.x, current_state.y))

        # Get the successors of the current state
        successors = current_state.getSuccessors(height_map, max_height_diff, target)

        # Add successors to the queue if they have not been visited
        for successor in successors:
            if (successor.x, successor.y) not in visited:
                queue.append(successor)

    # If the goal state is not found
    return None


# Depth-First Search algorithm
def dfs(height_map, start, target, max_height_diff):
    # Initialize stack with starting state
    stack = [NavigationState(start[0], start[1], target)]

    # Set to keep track of visited states
    visited = set()

    while stack:
        current_state = stack.pop()  # Pop the topmost node from the stack

        # Check if the current state is the goal state
        if (current_state.x, current_state.y) == target:
            solution_path = getSolutionPath(current_state)
            return solution_path

        # Mark the current state as visited
        visited.add((current_state.x, current_state.y))

        # Get the successors of the current state
        successors = current_state.getSuccessors(height_map, max_height_diff, target)

        # Add successors to the stack if they have not been visited
        for successor in successors:
            if (successor.x, successor.y) not in visited:
                stack.append(successor)
                visited.add((successor.x, successor.y))

    # If the goal state is not found
    return None


# Bidirectional Breadth-First Search algorithm
def bidirectional_bfs(height_map, start, target, max_height_diff):
    # Initialize queues for forward and backward search
    forward_queue = deque([NavigationState(start[0], start[1], target)])
    backward_queue = deque([NavigationState(target[0], target[1], start)])

    # Sets to keep track of visited states for forward and backward search
    forward_visited = set()
    backward_visited = set()

    while forward_queue and backward_queue:
        # Forward search
        forward_current_state = forward_queue.popleft()  # Pop the leftmost node from the forward queue
        if (forward_current_state.x, forward_current_state.y) in backward_visited:
            # If the forward state intersects with the backward search, a path is found
            forward_path = getSolutionPath(forward_current_state)
            backward_path = getSolutionPath(backward_visited[(forward_current_state.x, forward_current_state.y)])
            return forward_path + list(reversed(backward_path))

        # Mark the current state as visited in forward search
        forward_visited.add((forward_current_state.x, forward_current_state.y))

        # Get the successors of the current state in forward search
        forward_successors = forward_current_state.getSuccessors(height_map, max_height_diff, target)

        # Add successors to the forward queue if they have not been visited
        for forward_successor in forward_successors:
            if (forward_successor.x, forward_successor.y) not in forward_visited:
                forward_queue.append(forward_successor)

        # Backward search
        backward_current_state = backward_queue.popleft()  # Pop the leftmost node from the backward queue
        if (backward_current_state.x, backward_current_state.y) in forward_visited:
            # If the backward state intersects with the forward search, a path is found
            backward_path = getSolutionPath(backward_current_state)
            forward_path = getSolutionPath(forward_visited[(backward_current_state.x, backward_current_state.y)])
            return forward_path + list(reversed(backward_path))

        # Mark the current state as visited in backward search
        backward_visited.add((backward_current_state.x, backward_current_state.y))

        # Get the successors of the current state in backward search
        backward_successors = backward_current_state.getSuccessors(height_map, max_height_diff, start)

        # Add successors to the backward queue if they have not been visited
        for backward_successor in backward_successors:
            if (backward_successor.x, backward_successor.y) not in backward_visited:
                backward_queue.append(backward_successor)

    # If no path is found
    return None


# Uniform Cost Search (UCS) algorithm
def ucs(height_map, start, target, max_height_diff):
    # Initialize priority queue with starting state
    priority_queue = [(0, NavigationState(start[0], start[1], target))]

    # Set to keep track of visited states
    visited = set()

    while priority_queue:
        # Pop the state with the lowest cost from the priority queue
        current_cost, current_state = heapq.heappop(priority_queue)

        # Check if the current state is the goal state
        if (current_state.x, current_state.y) == target:
            solution_path = getSolutionPath(current_state)
            return solution_path

        # Mark the current state as visited
        visited.add((current_state.x, current_state.y))

        # Get the successors of the current state
        successors = current_state.getSuccessors(height_map, max_height_diff, target)

        # Add successors to the priority queue if they have not been visited
        for successor in successors:
            if (successor.x, successor.y) not in visited:
                heapq.heappush(priority_queue, (successor.cost, successor))

    # If the goal state is not found
    return None, None


def visualize_path_on_map(mars_map, path, new_scale, maxV, minV, n_rows, n_columns, scale, start, target, ax=None, three_D=False):
    # Create a copy of the original map
    path_map = np.copy(mars_map)

    # Mark the path positions on the map
    for x, y in path:
        path_map[x, y] = 0 # Mark the path position with the lowest height 
        # path_map[x, y] = mars_map.max()  # Mark the path position with the maximum height 
    
    # Plot the path map
    if three_D:
        plot3D(path_map, new_scale, maxV, minV, n_rows, n_columns)
    else:
        plot2D(mars_map, n_rows, n_columns, scale, new_scale, start, target, path, ax)


# Main function
def main():
    input_file = "mars_map.img"
    output_file = "mars_map.npy"

    # Process and visualize height map
    image_data, scale, new_scale, maxV, minV, n_rows, n_columns = process_height_map(input_file, output_file)
    
    # Load height map and calculate row and column from a given coordinate
    mars_map = np.load('mars_map.npy')
    
    # Define start and end points
    start = calculate_row_col(2850, 6400, new_scale, mars_map.shape[0])
    target = calculate_row_col(3150, 6800, new_scale, mars_map.shape[0])
    
    # Define maximum height difference allowed
    max_height_diff = 0.25  # meters
       
    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Find and visualize paths using different algorithms
    algorithms = [astar_search, dfs, ucs]
    algorithm_names = ['A* Search', 'DFS Searach', 'UCS Search']
    
    for i, (algorithm, name) in enumerate(zip(algorithms, algorithm_names)):
        path = algorithm(mars_map, start, target, max_height_diff)
        if path != (None, None):
            print(f"\nOptimal path found using {name} Algorithm. Distance traveled: {round((len(path) - 1) * new_scale)} meters.")        
            visualize_path_on_map(mars_map, path, new_scale, maxV, minV, n_rows, n_columns, scale, start, target, ax=axs[i])
            axs[i].set_title(name)
        else:
            print(f"\nNo valid navigation path found using {name}.")

    # Adjust layout and display the plot
    fig.suptitle('Mars Surface Navigation Paths', fontweight='bold')
    plt.tight_layout()
    plt.show() 

    # Visualize the best algorithm path in 3D
    path = astar_search(mars_map, start, target, max_height_diff)
    visualize_path_on_map(mars_map, path, new_scale, maxV, minV, n_rows, n_columns, scale, start, target, three_D=True)
    
    # Define different start and end points
    starts = [(1950, 8300), (4300, 15000), (2250, 100), (2850, 6400), (1010, 16000), (7000, 1500)]
    targets = [(2150, 8500), (4100, 15000), (3350, 4500), (5800, 6600), (2200, 8050), (100, 17500)]
    
    # Display Euclidian distance between each start and end point
    print()
    for start, target in zip(starts, targets):
        print(f"Distance between {start} and {target}: {round(np.sqrt((start[0] - target[0])**2 + (start[1] - target[1])**2))} meters")
    
    # Transform and combine the points into a single list
    starts = [calculate_row_col(x, y, new_scale, mars_map.shape[0]) for x,y in starts]
    targets = [calculate_row_col(x, y, new_scale, mars_map.shape[0]) for x,y in targets]
    combined_points = [(start, target) for start, target in zip(starts, targets)]
           
    # Create subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 5))
    
    # Find and visualize paths using A* algorithm   
    for i, (start, target) in enumerate(combined_points):
        path = astar_search(mars_map, start, target, max_height_diff)
        if path != (None,None):
            print(f"\nOptimal path found using A* Search Algorithm. Distance traveled: {round((len(path) - 1) * new_scale)} meters.")        
            visualize_path_on_map(mars_map, path, new_scale, maxV, minV, n_rows, n_columns, scale, start, target, ax=axs[i // 3, i % 3])
            axs[i // 3, i % 3].set_title('A* Search')
        else:
            print(f"\nNo valid navigation path found using A* Search.")

    # Adjust layout and display the plot
    fig.suptitle('Mars Surface Navigation Paths', fontweight='bold')
    plt.tight_layout()
    plt.show()
      
        
if __name__ == "__main__":
    main()