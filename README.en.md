# Maze Generation and Path Visualization Tool Requirements Document

Iteration Log

<table><tr><td>Date</td><td>Revision</td><td>Modification Description</td><td>Author</td></tr><tr><td>2025.05.19</td><td>1.0</td><td>Defined main features</td><td>Mu Zhishu</td></tr><tr><td>2025.05.20</td><td>2.0</td><td>Defined implementation details (e.g., algorithms and dependencies)</td><td>Mu Zhishu</td></tr></table>

# 1. Project Overview

This project is an interactive maze generation and path visualization tool implemented using Python and the Pygame library. The tool can:

Randomly generate mazes of different sizes
- Automatically find a path from the start to the end point
- Provide multiple visualization options (outline, topological map, centerline, etc.)
- Support saving maze data, images, and navigation videos

# 2. Functional Requirements

# 2.1 Core Functions

# - Maze Generation:

- Use the Randomized Prim's algorithm to generate the maze structure
- Support customizing the number of maze rows and columns (5-50)
- Walls (Gray) and Passages (White)
- Automatically mark the start point (Blue) and end point (Red)

# - Path Finding:

- Automatically find a path from the start to the end point
- Use a backtracking algorithm to ensure a valid path is found
- Visually display the path (Green)

# - Visualization Options:

- Show/Hide maze outline (Blue lines)
- Show/Hide topological structure map (Pink nodes and blue edges)
- Show/Hide passage centerlines (Orange points)
- Show/Hide path (Green fill)

# 2.2 User Interaction

# - Parameter Adjustment:

- Use slider controls to adjust the number of maze rows and columns
- Real-time preview of parameter changes

# Control Buttons:

- Generate Maze button: Regenerate the maze based on current parameters
- Pause/Resume button: Control the maze generation process
- Various Show/Hide toggle buttons

# - Save Functions:

- Save path coordinates to a CSV file
- Save the maze outline as a PNG image
- Save the topological structure map as a PNG image
- Save the path generation process as an MP4 video

# 2.3 Interface Design

- The main window is divided into a control panel (left) and a maze display area (right)
- Responsive design supporting window resizing
- Button hover effects and status feedback
- Real-time display of generation status and operation results
- Prototype Diagram:

!https://cdn-mineru.openxlab.org.cn/result/2025-10-29/645d10b8-d0f5-4bc6-9388-d88b077a4cf8/719922b3fab6bf9706b923cfe8d66ddf002303b208ed853fe2565c350cd43c5b.jpg

# 3. Technical Implementation

# 3.1 Key Algorithms

- Maze Generation Algorithm:
- Based on the Randomized Prim's algorithm
- Maintains a list of candidate nodes and a random selection process

- Path Finding Algorithm:
  - Uses a variant of Depth-First Search (DFS)
  - Supports backtracking mechanism to ensure a path is found

- Topological Analysis:
  - Calculates passage centerline points
  - Builds a graph structure to represent topological relationships

# 3.2 Dependencies

- Pygame: Graphical interface and interaction
- NetworkX: Topological map generation and analysis
- Matplotlib: Topological map plotting
- imageio: Video generation

# Appendix: Color Code Explanation

- White: Passage
- Gray: Wall
- Blue: Start Point
- Red: End Point
- Green: Path
- Orange: Centerline Point
- Pink: Topological Node
- Blue: Topological Edge and Outline
