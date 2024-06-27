## Mars Exploration: Informed and Uninformed Search

---

### Overview

This Python program implements various search algorithms to navigate a simulated real Martian surface represented by a height map. It utilizes informed and uninformed search algorithms to find optimal paths for the traversal of a rover across the Martian terrain, from a starting point to a target destination, while considering the constraints imposed by the rover and the Martian landscape.

### Introduction

Exploring the surface of Mars is a crucial endeavor for understanding the planet's geology and potential habitability. To aid in this exploration, the program leverages high-resolution topographic data obtained from satellites orbiting Mars, specifically the HiRISE (High-Resolution Imaging Science Experiment) mission, to construct a detailed representation of the Martian surface. The height map, stored in an IMG file, provides information about the elevation of different regions on Mars. Using this data, the program replicates the journey of a rover as it moves across the surface of Mars.

---

### Problem 1: Testing Search Algorithms

**Terrain Representation**

The Martian surface is depicted as a height map, with each pixel in the image corresponding to a specific area on the planet. The color of each pixel represents the height at that location, with deeper areas shown in dark red and higher areas in light yellow. The height map is processed to ensure valid data, with areas lacking information represented in gray.

**Search Algorithms**

The program evaluates the performance of five search algorithms:

1. A* Search (Informed)
2. Depth-First Search (Uninformed)
3. Uniform Cost Search (Uninformed)
4. Breadth-First Search (Uninformed)
5. Bidirectional Search (Uninformed)

These algorithms are tasked with finding a viable navigation route from a starting position to a target destination on the Martian surface. The rover can move between adjacent pixels, subject to height differentials not exceeding a specified threshold.

**Results and Analysis**

The search algorithms are assessed based on their ability to identify valid routes and the efficiency of pathfinding. A* Search, utilizing heuristic information, demonstrates effectiveness in navigating short distances with optimal paths. However, DFS and UCS algorithms also prove capable, indicating that uninformed search approaches can yield viable solutions.

---

### Problem 2: Performance Evaluation

**Short, Medium, and Long Routes**

The program further evaluates the chosen search algorithm's performance across various route distances. It examines routes with Euclidean distances between the start and target points:

1. Less than 500 meters
2. Between 1000 and 5000 meters
3. Greater than 10,000 meters

The selected algorithm's ability to find viable paths within acceptable time frames is analyzed for each scenario.

**Findings**

The chosen search algorithm successfully navigates short-distance routes efficiently, exemplifying its suitability for localized exploration tasks. However, its performance degrades for longer routes, particularly those exceeding 500 meters, due to computational constraints and search space limitations. Optimization strategies, such as parameter tuning and advanced pruning techniques, are suggested to enhance the algorithm's efficiency and effectiveness in tackling more complex navigation challenges on Mars.

---

### Dependencies:
- Python 3.x
- numpy: For numerical operations and array manipulation.
- skimage: Used for downsampling the height map to improve computational efficiency.
- matplotlib: Enables visualization of the Martian surface and navigation paths.
- plotly: Utilized for creating interactive 3D visualizations of the surface.

**Resources:**
- [mars_map.IMG](https://drive.google.com/file/d/1BgLp5tIhpV_7NqhkYPgSOZj0MbB4VasI/view?usp=share_link) (topographic information of the MARS Landscape extracted from HiRISE)
- mars_2D.png (example image for the introduction of problem 1)
- mars_3D.png (example image for the introduction of problem 1)

**Note:** Ensure that the HiRISE topography of MARS (mars_map.IMG) is accessible in the program directory for proper functioning.

---

### Conclusion

The Python program provides a versatile framework for simulating and evaluating navigation strategies for Martian exploration missions. By leveraging both informed and uninformed search algorithms, it offers insights into the trade-offs between computational complexity and pathfinding effectiveness in diverse Martian terrain conditions. Continuous refinement and optimization of these algorithms are essential for enabling future robotic missions to explore and uncover the mysteries of the Red Planet.