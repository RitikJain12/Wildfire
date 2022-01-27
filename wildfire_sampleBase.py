import numpy as np #using numpy arrays
import matplotlib.pyplot as plt #representing the grid in graph
import random
from os import path, startfile
from matplotlib.patches import Rectangle
from numpy import pi, sqrt
import numpy as np
import math
import threading
import time

scaling_factor = 15/4

#parameters for obstacle grid
rows =round(250/scaling_factor)
cols =round(250/scaling_factor)
grid_size = rows*cols
coverage = 20 #coverage percent
obstacle_size = 4*round(15/scaling_factor)
obstacle = np.array([np.array([[1],[1],[1],[1]]),
                    np.array([[1,1,0],[0,1,1]]),
                    np.array([[1,1],[1,1]]),
                    np.array([[1,0],[1,0],[1,1]]),
                    np.array([[1,1,1],[0,1,0]]),
                    np.array([[0,1],[0,1],[1,1]]),
                    np.array([[0,1,1],[1,1,0]])],dtype=object) #defining tetres obstacles
obstacle_state = []
burning_list = []
extinguish = 0
total_cpu = 0

grid = np.zeros((rows+1,cols+1)) #creating an obstacle free grid

#parameters for motion planning
agent_start = [0,0,0,0,0]
agent_goal = [-1,-1]

#kinematic parameters
wheelbase = 3/scaling_factor
steering_angle = 12
vel = 10/scaling_factor
car_height = 2.2/scaling_factor
car_width = 4.9/scaling_factor

burning_radius = 30
estinguish_radius = 10
end_simulation = True

# use to display the car at different points 
agent_bound_T = [[-(car_width-wheelbase)/2,car_width-(car_width-wheelbase)/2,car_width-(car_width-wheelbase)/2,-(car_width-wheelbase)/2],[-car_height/2,-car_height/2,car_height/2,car_height/2],[1,1,1,1]]

total_sample_points = 2000
sample_points=[]
graph = {}

def probabilistic_map():
    
    for i in range(total_sample_points):
        while True:
            # x=round(random.random()*(rows-1),2)
            # y=round(random.random()*(cols-1),2)
            x=random.randint(0,rows-1)
            y=random.randint(0,cols-1)
            if [x,y] not in sample_points and grid[round(x)][round(y)]!=1:
                sample_points.append([x,y])
                break
    for i in range(len(sample_points)):
        min1 = math.inf
        min2 = math.inf
        min3 = math.inf
        min4 = math.inf
        min5 = math.inf
        index1,index2,index3,index4,index5 = -1,-1,-1,-1,-1
        for j in range(len(sample_points)):
            if i!=j:
                if min1>=math.dist(sample_points[i],sample_points[j]) and valid_join(sample_points[i],sample_points[j]):
                    min5=min4
                    min4=min3
                    min3=min2
                    min2=min1
                    min1=math.dist(sample_points[i],sample_points[j])
                    index5=index4
                    index4=index3
                    index3=index2
                    index2=index1
                    index1=j
                elif min2>=math.dist(sample_points[i],sample_points[j]) and valid_join(sample_points[i],sample_points[j]):
                    min5=min4
                    min4=min3
                    min3=min2
                    min2=math.dist(sample_points[i],sample_points[j])
                    index5=index4
                    index4=index3
                    index3=index2
                    index2=j
                elif min3>=math.dist(sample_points[i],sample_points[j]) and valid_join(sample_points[i],sample_points[j]):
                    min5=min4
                    min4=min3
                    min3=math.dist(sample_points[i],sample_points[j])
                    index5=index4
                    index4=index3
                    index3=j
                elif min4>=math.dist(sample_points[i],sample_points[j]) and valid_join(sample_points[i],sample_points[j]):
                    min5=min4
                    min4=math.dist(sample_points[i],sample_points[j])
                    index5=index4
                    index4=j
                elif min5>=math.dist(sample_points[i],sample_points[j]) and valid_join(sample_points[i],sample_points[j]):
                    min5=math.dist(sample_points[i],sample_points[j])
                    index5=j
        graph[str(sample_points[i][0])+','+str(sample_points[i][1])] = [[sample_points[index1][0],sample_points[index1][1]],[sample_points[index2][0],sample_points[index2][1]],[sample_points[index3][0],sample_points[index3][1]],[sample_points[index4][0],sample_points[index4][1]],[sample_points[index5][0],sample_points[index5][1]]]
    return graph

def valid_join(p1,p2):
    slope = math.atan2((p2[1]-p1[1]),(p2[0]-p1[0]))
    d=math.dist(p1,p2)
    step=round(d/1)
    x_=min(p1[0],p2[0])
    y_=min(p1[1],p2[1])
    for i in range(step):
        x_+=1*math.cos(slope)
        y_+=1*math.sin(slope)
        if x_<rows and y_<cols:
            if grid[round(x_)][round(y_)]==1 or grid[round(x_)+1][round(y_)]==1 or grid[round(x_)][round(y_)+1]==1 or grid[round(x_)-1][round(y_)]==1 or grid[round(x_)][round(y_)-1]==1:
                return False
    # for x in range(math.floor(min(p1[0],p2[0])),math.ceil(max(p1[0],p2[0]))+1):
    #     x_+=res*math.cos(slope)
    #     y_+=res*math.sin(slope)
    #     for y in range(math.floor(min(p1[1],p2[1])),math.ceil(max(p1[1],p2[1]))+1): 
    #         if grid[x][y]==1 and math.dist([x,y],[x_,y_])<=1:
    #             return False
    return True


# to check if the position of the car is valid or not
def valid_point(x,y,theta): 
    if x < wheelbase or y < wheelbase or x > rows-wheelbase-2 or y > cols-wheelbase-2:
        return False 
    if grid[round(x)][round(y)]==1:
        return False
    #collision conditions between boundary and different obstacles
    boundary = get_boundary(x,y,theta)
    for x_,y_ in boundary:
        if grid[round(x_)][round(y_)]==1 or grid[round(x_)-1][round(y_)]==1 or grid[round(x_)][round(y_)-1]==1 or grid[round(x_)+1][round(y_)]==1 or grid[round(x_)][round(y_)+1]==1 or grid[round(x_)-1][round(y_)-1]==1 or grid[round(x_)+1][round(y_)-1]==1 or grid[round(x_)+1][round(y_)+1]==1 or grid[round(x_)-1][round(y_)+1]==1:
            return False       
    
    return True

def get_neighbours(x,y):
    neighbour = graph[str(x)+','+str(y)]
    return neighbour

#g(x) function
def cost_function(x1,y1,x2,y2): 
    distance = math.sqrt((pow(x1-x2,2)+pow(y1-y2,2)))
    return distance

#h(x) function
def hurestic_function(x,y): 
    distance = math.sqrt((pow(agent_goal[0]-x,2)+pow(agent_goal[1]-y,2))) # for x,y
    hurestic = distance
    return hurestic

def priority(queue): #find the path with shortest distance that will be selected from the priority queue
    min = math.inf
    index = 0
    for check in range(len(queue)):
        _,value,_,_ = queue[check]
        if value<min:
            min = value
            index = check #index of the shortest path
    return index

# to check for visited nodes for the A* algorithm
def check_visited(current,visited):
    for x,y in visited:
        if current[0]== x and current[1]== y :
            return True
            
    return False

#A* algorithm to find the shortest path from the start orientation to goal orientation
def A_star():
    open_set = []
    visited = []
    start = agent_start
    tcost = 0
    gcost = 0
    path = [start]
    open_set.append((start,tcost,gcost,path))
    while len(open_set)>0:
        index = priority(open_set)
        (shortest,_,gvalue,path) = open_set[index] #select the node with lowest distance
        open_set.pop(index)
        if not (check_visited([(shortest[0]),(shortest[1])],visited)): # check if already visited
            visited.append([(shortest[0]),(shortest[1])])
            if abs((shortest[0])==agent_goal[0]) and ((shortest[1]) == agent_goal[1]): # goal conditions
                return path
            neighbours= get_neighbours(shortest[0],shortest[1]) 
            for neighbour in neighbours:#calculate cost of each neighbor
                temp_gcost = gvalue+(cost_function(shortest[0],shortest[1],neighbour[0],neighbour[1]))
                temp_tcost = temp_gcost+(hurestic_function(neighbour[0],neighbour[1]))
                open_set.append((neighbour,temp_tcost,temp_gcost,path+ [neighbour]))
    print("not working")      
    return path

def connect_node(node):
    min1 = math.inf
    min2 = math.inf
    min3 = math.inf
    index1,index2,index3 = -1,-1,-1
    for j in range(len(sample_points)-1):
        if min1>=math.dist(node,sample_points[j]) and valid_join(node,sample_points[j]):
            min3=min2
            min2=min1
            min1=math.dist(node,sample_points[j])
            index3=index2
            index2=index1
            index1=j
        elif min2>=math.dist(node,sample_points[j]) and valid_join(node,sample_points[j]):
            min3=min2
            min2=math.dist(node,sample_points[j])
            index3=index2
            index2=j
        elif min3>=math.dist(node,sample_points[j]) and valid_join(node,sample_points[j]):
            min3=math.dist(node,sample_points[j])
            index3=j
    graph[str(node[0])+','+str(node[1])] = [[sample_points[index1][0],sample_points[index1][1]],[sample_points[index2][0],sample_points[index2][1]],[sample_points[index3][0],sample_points[index3][1]]]
    graph[str(sample_points[index1][0])+','+str(sample_points[index1][1])].append(node)
    graph[str(sample_points[index2][0])+','+str(sample_points[index2][1])].append(node)
    graph[str(sample_points[index3][0])+','+str(sample_points[index3][1])].append(node)
    return graph


# to get the boundary of the car from the  back axle position using homogeneous transformations 
def get_boundary(x,y,theta):
    tx = x 
    ty = y 
    th = theta
    homogeneous_matrix = [[math.cos(th*(pi/180)),-math.sin(th*(pi/180)),tx],[math.sin(th*(pi/180)),math.cos(th*(pi/180)),ty]]
    mat_mul = np.dot(homogeneous_matrix,agent_bound_T)
    new_boundary = [[mat_mul[0][0],mat_mul[1][0]],[mat_mul[0][1],mat_mul[1][1]],[mat_mul[0][2],mat_mul[1][2]],[mat_mul[0][3],mat_mul[1][3]]]
    return new_boundary



def check_overlap(grid,obstacle,row,col):
    for n in range(row):
        for m in range(col):
            if grid[n][m] == 1 and obstacle[n][m]==1: #to abvoid overlaping 2 obstacles
                return True
    return False

def create_grid(coverage): 
    max_obstacle = round(((coverage/100)*(grid_size))/obstacle_size) #getting max obstacles
    total_obstacle = 0 #check the number obstacles placed
    while total_obstacle!=max_obstacle: #filling obstacles till we reach close coverage percent
        i = random.randint(1,rows-1)
        j = random.randint(1,cols-1)
        index = random.randint(0,6) #choosing a random obstacle
        n,m = obstacle[index].shape
        if i+n>rows or j+m>cols: # check out of bounce condition
            continue 
        elif check_overlap(grid[i:i+n,j:j+m],obstacle[index],n,m): #to abvoid overlaping of 2 obstacles
            continue
        total_obstacle+=1
        for oi in range(n):
            for oj in range(m):
                if obstacle[index][oi][oj] == 1: #place the obstacle
                    grid[i+oi][j+oj]=obstacle[index][oi][oj]
                    obstacle_state.append([i+oi,j+oj,"N"])
    sort_obstacles()
    return 

def sort_obstacles():
    for i in range(len(obstacle_state)):
        for j in range(i+1,len(obstacle_state)-1):
            if obstacle_state[i][0]>obstacle_state[j][0]:
                temp = obstacle_state[i]
                obstacle_state[i]=obstacle_state[j]
                obstacle_state[j]=temp
            elif obstacle_state[i][0]==obstacle_state[j][0]:
                if obstacle_state[i][1]>obstacle_state[j][1]:
                    temp = obstacle_state[i]
                    obstacle_state[i]=obstacle_state[j]
                    obstacle_state[j]=temp
    return

def wumpus_burn():
    burn_index = random.randint(0,len(obstacle_state)-1)
    for i in range(len(obstacle_state)):
        if obstacle_state[burn_index][2]!='B':
            obstacle_state[burn_index][2]='B'
            burning_list.append([obstacle_state[burn_index][0],obstacle_state[burn_index][1]])
            break
        burn_index = (burn_index+1)%len(obstacle_state)
    # print(obstacle_state[burn_index])
    threading.Timer( 61, wumpus_burn).start()
    return

def burn_spread():
    new_burn = []
    for i in range(len(obstacle_state)):
        if obstacle_state[i][2]=='B':
            for j in range(len(obstacle_state)):
                if abs(obstacle_state[j][0]-obstacle_state[i][0])<(burning_radius/scaling_factor) and abs(obstacle_state[j][1]-obstacle_state[i][1])<(burning_radius/scaling_factor) and obstacle_state[j][2]!='B':
                    new_burn.append(j)
    for new in new_burn:
        obstacle_state[new][2]='B'     
        if [obstacle_state[new][0],obstacle_state[new][1]] not in burning_list:
            burning_list.append([obstacle_state[new][0],obstacle_state[new][1]])  
    threading.Timer( 20, burn_spread).start()
    return

def burn_estinguish(x,y):
    global extinguish
    for x_,y_ in burning_list:
        if abs(x-x_)<=round(estinguish_radius/scaling_factor) and abs(y-y_)<=round(estinguish_radius/scaling_factor):
            burning_list.remove([x_,y_])
            extinguish+=1
    change_state()
    return

def change_state():
    for i in range(len(obstacle_state)):
        if obstacle_state[i][2]=='B' and [obstacle_state[i][0],obstacle_state[i][1]] not in burning_list:
            obstacle_state[i][2]='N'
    return

def animate(posx,posy,theta):
    burn_estinguish(round(posx),round(posy))
    plt.figure(coverage) #display 3 windows of diff coverage
    fig = plt.gcf() # represnt the final grid
    ax = fig.add_subplot(1, 1, 1)
    fig.canvas.manager.set_window_title("Search Based Planning")
    track = 0
    # plt.grid()
    for x in range(rows):
        for y in range(cols):
            if grid[x][y] == 1:
                if obstacle_state[track][2]=='N':
                    plt.scatter(x,y,c='green',s=3,marker="s") #plot obstacles
                elif obstacle_state[track][2]=='B':
                    plt.scatter(x,y,c='red',s=6,marker="s") #plot burning obstacles
                track+=1

    plt.xlim([-1,rows])#set graph from 0 to 128 for both x and y 
    plt.ylim([-1,cols])
    boundary = get_boundary(posx,posy,theta)
    X = []
    Y = []
    for x,y in boundary:
        X.append(x)
        Y.append(y)
    
    ax.plot(X,Y,linewidth =0.6)
    return

def get_nearest():
    min = math.inf
    index = 0
    # print(len(burning_list))
    for i in range(len(burning_list)):
        dist = sqrt(pow(agent_start[0]-burning_list[i][0],2)+pow(agent_start[1]-burning_list[i][1],2))
        if dist<min:
            if agent_goal[0]==burning_list[i][0] and agent_goal[1]==burning_list[i][1]:
                continue
            else:
                min = dist
                index = i
    if min==math.inf:
        return True,agent_goal
    else:
        return False,burning_list[index]

def spawn_random():
    while True:
        x=random.randint(0,rows-1)
        y=random.randint(0,cols-1)
        theta =random.randint(0,360)
        if valid_point(x,y,theta):
            break
    return [x,y,theta,0,0]

def simulation_begin():
    global end_simulation
    if end_simulation:
        end_simulation =False
        threading.Timer( 2*60, simulation_begin).start()
    else:
        end_simulation = True
    return
'''
def integrate_rk(x,y,theta,h=12):
    s=vel
    L=wheelbase
    q_list=[]
    phi = theta*(pi/180)
    # if x==0:
    #     phi=math.pi/2
    # else:
    #     phi=math.atan2(y,x)
    q=[x,y,phi]
    for i in range(5):
        k1 = (h*s*math.cos(q[2]),h*s*math.sin(q[2]),h*s*math.tan(phi)/L)
        k2 = (h*s*math.cos(q[2]+0.5*h),h*s*math.sin(q[2]+0.5*h),h*s*math.tan(phi+k1[2]*0.5*h)/L)
        k3 = (h*s*math.cos(q[2]+0.5*h),h*s*math.sin(q[2]+0.5*h),h*s*math.tan(phi+k2[2]*0.5*h)/L)
        k4 = (h*s*math.cos(q[2]+h),h*s*math.sin(q[2]+h),h*s*math.tan(phi+k3[2]*h)/L)

        x_new = q[0] + k1[0]+2*k2[0]+2*k3[0]+k4[0]*h*(1/float(6))
        y_new = q[1] + k1[1]+2*k2[1]+2*k3[1]+k4[1]*h*(1/float(6))
        theta_new = q[2] +k1[2]+2*k2[2]+2*k3[2]+k4[2]*h*(1/float(6))
        # if q[0]>0 and q[0]<rows and q[1]>0 and q[1]<cols:
        q_list.append(q)
        q=[x_new,y_new,theta_new]

    return q_list

def local_planner(p1,p2):
    positions = []
    # print(p2)
    
    dist = sqrt(pow(p1[0]-p2[0],2)+pow(p1[1]-p2[1],2))
    dist_approach=dist
    pre_approach=dist
    x=p1[0]
    y=p1[1]
    theta = p1[2]
    if p2[2]>180:
        theat2_=180-p2[2]
    else:
        theat2_=p2[2]
    if theta>180:
        theta_=180-theta
    else:
        theta_=theta
    if abs(theat2_-theta_)<=90:
        gear =1
    else:
        gear =-1
    while dist_approach<=pre_approach:
        # print(itr)
        if theta>180:
            theta_=180-theta
        else:
            theta_=theta
        if (theat2_-theta_+180)%180<15:
            x+=vel*math.cos(theta*(pi/180))*gear
            y+=vel*math.sin(theta*(pi/180))*gear
            if valid_point(x,y,theta):
                positions.append([x,y,theta])
                pre_approach=dist_approach
                dist_approach=sqrt(pow(x-p2[0],2)+pow(y-p2[1],2))
                # print("forward")
                continue
            else:
                x-=vel*math.cos(theta*(pi/180))*gear
                y-=vel*math.sin(theta*(pi/180))*gear
        if theat2_-theta_>=0:
            x+=vel*math.cos(theta*(pi/180))*gear
            y+=vel*math.sin(theta*(pi/180))*gear
            theta +=(vel*math.tan(steering_angle*(pi/180))/wheelbase)*(180/pi)*gear
            if valid_point(x,y,theta):
                positions.append([x,y,theta])
                pre_approach=dist_approach
                dist_approach=sqrt(pow(x-p2[0],2)+pow(y-p2[1],2))
            # print("left")
                continue
            else:
                x-=vel*math.cos(theta*(pi/180))*gear
                y-=vel*math.sin(theta*(pi/180))*gear
                theta +=(vel*math.tan(steering_angle*(pi/180))/wheelbase)*(180/pi)*gear
        if theat2_-theta_<=0:
            x+=vel*math.cos(theta*(pi/180))*gear
            y+=vel*math.sin(theta*(pi/180))*gear
            theta +=(vel*math.tan(-steering_angle*(pi/180))/wheelbase)*(180/pi)*gear
            if valid_point(x,y,theta):
                positions.append([x,y,theta])
                pre_approach=dist_approach
                dist_approach=sqrt(pow(x-p2[0],2)+pow(y-p2[1],2))
            # print("right")
                continue
            else:
                break
        # plt.cla()
        # animate(x,y,theta)
        # plt.pause(0.1)
    # positions = [[p2[0],p2[1],p2[2]]]
    return positions
'''
def local_get_neighbours(x,y,theta):
    neighbour = []
    for i in range(-steering_angle,steering_angle+1,6):
        x_dot = vel*math.cos(theta*(pi/180))
        y_dot = vel*math.sin(theta*(pi/180))
        theta_dot = (vel*math.tan(i*(pi/180))/wheelbase)*(180/pi)
        if(valid_point(x+x_dot,y+y_dot,theta+theta_dot)): # to check if the neighbour position is a valid one before adding it to the list of neighbour
            neighbour.append([round(x+x_dot,2),round(y+y_dot,2),(round(theta+theta_dot,2))%360,vel,i])
        if(valid_point(x-x_dot,y-y_dot,theta-theta_dot)): # to check if the neighbour position is a valid one before adding it to the list of neighbour
            neighbour.append([round(x-x_dot,2),round(y-y_dot,2),(round(theta-theta_dot,2)+360)%360,-vel,i])
    return neighbour

# g(x) function 
def local_cost_function(x1,y1,x2,y2):
    distance = sqrt((pow(x1-x2,2)+pow(y1-y2,2)))
    return distance

# h(x) function
def local_hurestic_function(x,y,turn,pre_turn,direction,pre_direction,agent_goal):
    turn_cost=0
    change_turn_cost=0
    reverse_cost=0
    gear_change=0
    if turn!=0:
        turn_cost = 1
    if turn != pre_turn:
        change_turn_cost = 5
    if direction<0:
        reverse_cost = 5
    if direction!=pre_direction and pre_direction!=0:
        gear_change = 100
    distance_cost =50*sqrt((pow(agent_goal[0]-x,2)+pow(agent_goal[1]-y,2))) # for x,y # distance of the back axle
    hurestic = distance_cost+turn_cost+change_turn_cost+reverse_cost+gear_change
    return hurestic

def local_check_visited(current,visited):
    for x,y,th in visited:
        if current[0]== x and current[1]== y and current[2]==th :
            return True
    return False

def local_A_star(agent_start,agent_goal):
    open_set = []
    visited = []
    start = agent_start
    tcost = 0
    gcost = 0
    path = [start]
    open_set.append((start,tcost,gcost,path))
    while len(open_set)>0:
        index = priority(open_set)
        (shortest,_,gvalue,path) = open_set[index] #select the node with lowest distance
        open_set.pop(index)
        if not (local_check_visited([round(shortest[0]),round(shortest[1]),round(shortest[2])],visited)): # check if already visited
            visited.append([round(shortest[0]),round(shortest[1]),round(shortest[2])])
            if round(shortest[0]) <= agent_goal[0]+(estinguish_radius/scaling_factor) and round(shortest[0]) >= agent_goal[0]-(estinguish_radius/scaling_factor) and round(shortest[1]) <= agent_goal[1]+(estinguish_radius/scaling_factor) and round(shortest[1]) >= agent_goal[1]-(estinguish_radius/scaling_factor) : #goal condition
                # print("reached")
                return path
            neighbours= local_get_neighbours(shortest[0],shortest[1],shortest[2]) # get valid neighbours using tehe kinematic equation
            for neighbour in neighbours:#calculate cost of each neighbor
                vel = neighbour[3]
                turn = neighbour[4]
                temp_gcost = gvalue+(local_cost_function(shortest[0],shortest[1],neighbour[0],neighbour[1]))
                temp_tcost = temp_gcost+(local_hurestic_function(neighbour[0],neighbour[1],turn,shortest[4],vel,shortest[3],agent_goal))
                if not (local_check_visited([round(neighbour[0]),round(neighbour[1]),round(neighbour[2])],visited)):
                    open_set.append((neighbour,temp_tcost,temp_gcost,path+ [neighbour]))
    print("not working")      
    return path

def get_intact():
    intact=0
    for obs in obstacle_state:
        if obs[2]=='N':
            intact+=1
    return intact

def get_burned():
    burned=0
    for obs in obstacle_state:
        if obs[2]!='N':
            burned+=1
    return burned

def main():
    global agent_goal
    global agent_start
    global end_simulation
    global graph
    global extinguish
    global total_cpu
    create_grid(coverage)
    print("grid ready")
    agent_start = spawn_random()
    # print(agent_start)
    simulation_begin()
    wumpus_burn()
    t1=time.time()
    graph = probabilistic_map()
    
    if [agent_start[0],agent_start[1]] not in sample_points:
        graph = connect_node([agent_start[0],agent_start[1]])
        sample_points.append([agent_start[0],agent_start[1]])
    total_cpu+=(time.time()-t1)
    threading.Timer( 20, burn_spread).start()
    plt.cla()
    animate(agent_start[0],agent_start[1],agent_start[2])
    plt.pause(0.1)
    while not end_simulation:
        # print(burning_list)
        t1=time.time()
        if [agent_start[0],agent_start[1]] not in sample_points:
            graph = connect_node([agent_start[0],agent_start[1]])
            sample_points.append([agent_start[0],agent_start[1]])
        total_cpu+=(time.time()-t1)
        wait_state,agent_goal = get_nearest()
        if wait_state:
            continue
        # print(agent_goal)
        t1=time.time()
        if agent_goal not in sample_points:
            graph = connect_node(agent_goal)
            sample_points.append(agent_goal)
        path = A_star()
        total_cpu+=(time.time()-t1)
        # print(path[-1])
        x = agent_start[0]
        y = agent_start[1]
        theta = agent_start[2]
        for i in range(len(path)-1):
            # if i==0:
            #     local_path=local_planner([agent_start[0],agent_start[1],agent_start[2]],[path[i+1][0],path[i+1][1],(360+(180/pi)*math.atan2(path[i+1][1]-agent_start[1],path[i+1][0]-agent_start[0]))%360])
            # else:
            #     local_path=local_planner([x,y,theta],[path[i+1][0],path[i+1][1],(360+(180/pi)*math.atan2(path[i+1][1]-y,path[i+1][0]-x))%360])
            # print("local_path done")
            # local_path=integrate_rk(path[i][0],path[i][1],theta)
            # local_path= local_planner([x,y,theta],[path[i+1][0],path[i+1][1],(360+(180/pi)*math.atan2(path[i+1][1]-y,path[i+1][0]-x))%360])
            # print(local_path)
            t1=time.time()
            local_path = local_A_star([x,y,theta,0,0],path[i+1])
            total_cpu+=(time.time()-t1)
            for points in local_path:
                plt.cla()
                animate(points[0],points[1],points[2])
                plt.pause(0.1)
                x=points[0]
                y=points[1]
                theta=points[2]
        # print(x,y,theta)
        # print(burning_list)
        agent_start=[x,y,theta]
    threading.Timer( 20, burn_spread).cancel()
    threading.Timer( 61, wumpus_burn).cancel()
    plt.close()
    print("Done")
    print("Intact "+str(get_intact()))
    print("Burned "+str(get_burned()))
    print("Extinguished "+str(extinguish))
    print("Total CPU time "+str(total_cpu))
    # plt.show()

if __name__ == "__main__":
    main()