# Wildfire
Planning path using Hybrid A* and PRM to extinguish fire in a wildfire senario.

[![IMAGE ALT TEXT](http://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](http://www.youtube.com/watch?v=YOUTUBE_VIDEO_ID_HERE "Video Title")


# Environment
The environment consists of a flat square field, 250 meters on a side, filled with obstacles. The obstacles consist of large patches of un-navigable thick brush, trees, and weeds, suspiciously shaped like giant tetrominoes. While the environment is not specifically a grid, the base dimension for each obstacle square unit is 15 meters. Inside this field, a firetruck operates, attempting to extinguish fires that emerge.
Starting at time 0 and at 60 second intervals, an arsonist/wumpus sets a major conflagration at a random obstacle. This sets the obstacle state to burning. After 20 seconds in this state, the obstacle sets all obstacles within a 30 meter radius to the state of burning. Run the simulation for 3600 seconds.
The truck starts at a random point in the map. If the truck stops within 10 meters of a burning obstacle, it sets the state to extinguished. Use a path planner to drive the truck to desired locations and attempt to extinguish as many obstacles as possible.

# Algorithm 
## Search Based Planner: A*
It uses kinematics equation to find the possible neighbor and add them to the open list, if they are not already in it or in the closed list with their cost using the cost function. The node with lowest cost is selected and added to the closed list. It continues till the goal node is reached.
## Sample based Planner: PRM
Probabilistic road map takes n set of random points in the map and discard the ones which are no obstacles. We then connect the k nearest neighbors whose paths are collision free. Using this graph, we apply A* algorithm to get a global planner to reach the goal. Now using the points of the global planner, we find the path to reach from one point to other which is called the local planner. For this simulation I have used the above search-based planner as the local planner. 


# Ackerman Steering Kinematics

<img width="457" alt="ackerman" src="https://user-images.githubusercontent.com/48152038/154722402-ad50a223-5d28-482d-a5d8-7fefe8068779.png">

<img width="322" alt="ackerman_explanation" src="https://user-images.githubusercontent.com/48152038/154723161-b47da45f-4735-4850-a61e-c8e4c3f89011.png">
