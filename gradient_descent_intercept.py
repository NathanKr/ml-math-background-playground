import numpy as np
import matplotlib.pyplot as plt
from plot_utils import plot_with_residual

# this is based on https://www.youtube.com/watch?v=sDv4f4s2SB8

# data set
Weight = np.array([0.5 , 2.3 , 2.9]) # x
Height = np.array([1.4 , 1.9 , 3.2]) # y , _observed_height

# Predicted Height = intercept + slope * Weight --> h
def predicted_height(intercept , slope):
    return intercept + slope * Weight

def plot_dataset():
    plt.plot(Weight, Height,'o')
    plt.grid()
    plt.title('data set')
    plt.xlabel("Weight")
    plt.ylabel("Height")
    plt.show()

# _predicted_height is h , _observed_height is y
def sum_of_square_residual(_predicted_height , _observed_height):
    residual = _observed_height - _predicted_height
    return np.dot(residual,residual) # sum the square


def show_ssr_graph_per_intercept(slope):
    vec_intercept = np.arange(0,2,0.1)
    vec_ssr = []
    for intercept in vec_intercept:
        _predicted_height = predicted_height(intercept,slope)
        _observed_height = Height
        ssr = sum_of_square_residual(_predicted_height,_observed_height)
        vec_ssr.append(ssr)

    index_min_ssr = np.argmin(vec_ssr)
    ssr_min = vec_ssr[index_min_ssr]    
    intercept_min = vec_intercept[index_min_ssr]  
    plt.plot(vec_intercept,vec_ssr,'o',intercept_min,ssr_min,'ro')    
    plt.title('cost ssr vs intercept for slope = {}'.format(slope))
    plt.xlabel("intercept")
    plt.ylabel("ssr - sum of squre residual")
    plt.grid()
    plt.show()

def compute_one_ssr(slope):
    intercept = 0 # initial guess
    _predicted_height = predicted_height(intercept,slope)
    _observed_height = Height
    ssr = sum_of_square_residual(_predicted_height,_observed_height)
    print('ssr : ' , ssr)

# ssr is the sum over (y[i])-h[i]))^2
# d_ssr / d_intercept is equal by the chain rule to (d_ssr / d_h) * (d_h / d_intercept)
# d_ssr / d_h -> sum over 2(y[i])-h[i]))*(-1)
# d_h / d_intercept = d(intercept + slope * weight) / d_intercept --> 1 
# so  d_ssr / d_intercept is equal to the sum over 2(y[i])-h[i]))*(-1)  but h = intercept + slope * weight so
# d_ssr / d_intercept is equal to the sum over 2 ( y[i] - (intercept + slope * weight[i]))*(-1) or equivalently
# d_ssr / d_intercept is equal to the sum over 2 ( observed_height[i] - (intercept + slope * weight[i]))(-1)
def d_ssr_to_d_intercept(slope,intercept):
    _observed_height = Height
    _predicted_height = intercept + slope * Weight
    error = _observed_height - _predicted_height
    return 2 * np.sum(error) *(-1)


# ssr is the sum over (y[i])-h[i]))^2
# d_ssr / d_slope is equal by the chain rule to (d_ssr / d_h) * (d_h / d_slope)
# d_ssr / d_h -> sum over 2(y[i])-h[i]))*(-1)
# d_h / d_slope = d(intercept + slope * weight) / d_slope --> weight 
# so  d_ssr / d_intercept is equal to the sum over 2(y[i])-h[i]))*(-1)*weight[i]  but h = intercept + slope * weight so
# d_ssr / d_intercept is equal to the sum over 2 ( y[i] - (intercept + slope * weight[i]))*(-1)*weight[i] or equivalently
# d_ssr / d_intercept is equal to the sum over 2 ( observed_height[i] - (intercept + slope * weight[i]))(-1)*weight[i]
def d_ssr_to_d_slope(slope,intercept):
     _observed_height = Height
     _predicted_height = intercept + slope * Weight
     error = _observed_height - _predicted_height
     # np.dot : mutliply element by element and than sum
     return 2 * np.dot(error , Weight) *(-1)    

def print_d_ssr_to_d_intercept(slope):
    # derivative become smaller
    print("d_ssr/d_intercept @ intercept = 0 : ",d_ssr_to_d_intercept(slope,0))
    print("d_ssr/d_intercept @ intercept = 0.5 : ",d_ssr_to_d_intercept(slope,0.5))
    print("d_ssr/d_intercept @ intercept = 0.8 : ",d_ssr_to_d_intercept(slope,0.8))
    print("d_ssr/d_intercept @ intercept = 0.9 : ",d_ssr_to_d_intercept(slope,0.9))
    print("d_ssr/d_intercept @ intercept = 0.95 : ",d_ssr_to_d_intercept(slope,0.95))
    print("d_ssr/d_intercept @ intercept = 1.05 : ",d_ssr_to_d_intercept(slope,1.05))

def gradient_descent_constant_slope(slope):
    intercept = 0
    learning_rate = 0.1
    step_size = 1 # just a value to enter the loop
    min_step_size = 0.001
    iteration = 0
    ssr_vec = []
    intercept_vec = []

    while abs(step_size) > min_step_size:
        step_size = d_ssr_to_d_intercept(slope,intercept) * learning_rate
        intercept = intercept - step_size # i did not see a proof for this

        # ------------ below this is relevant to plot
        intercept_vec.append(intercept)
        _predicted_height = predicted_height(intercept,slope)
        _observed_height = Height
        iteration += 1
        ssr = sum_of_square_residual(_predicted_height , _observed_height)
        ssr_vec.append(ssr)

        plot1(intercept_vec,ssr_vec,intercept,slope,ssr , iteration,step_size)

def plot1(intercept_vec,ssr_vec,intercept,slope,ssr , iteration,step_size):
    fig, axs = plt.subplots(2)
    fig.suptitle('Part 1')
    plot_with_residual(axs[0],Weight,Height,intercept,slope)
    axs[0].grid()
    axs[0].set_title('data set vs intercpt  + slope * weight and residual\nssr : {:.2f} , intercept : {:.2f} , iteration : {} , step : {:.4f}'.format(ssr , intercept , iteration,step_size))
    axs[0].set_xlabel("Weight")
    axs[0].set_ylabel("Height")
    axs[1].plot(intercept_vec,ssr_vec,'o',intercept,ssr,'ro')
    axs[1].set_title('gradient descent convergence , learn intercept . step size become smaller')
    axs[1].set_xlabel("intercept")
    axs[1].set_ylabel("cost function - ssr")
    axs[1].grid()
    plt.tight_layout()
    plt.show()

def gradient_descent():
    intercept = 0
    slope = 1
    learning_rate = 0.01 # using 0.1 will not due
    step_size = 1 # just a value to enter the loop
    min_step_size = 0.001
    iteration = 0
    ssr_vec = []

    while step_size > min_step_size:
        # ------------ this is the gradient descent engine
        step_size_intercept = d_ssr_to_d_intercept(slope,intercept) * learning_rate
        step_size_slope = d_ssr_to_d_slope(slope,intercept) * learning_rate
        intercept = intercept - step_size_intercept # i did not see a proof for this
        slope = slope - step_size_slope
        step_size = max(abs(step_size_intercept) , abs(step_size_slope))

        # ------------ below this is relevant to plot
        _predicted_height = predicted_height(intercept,slope)
        _observed_height = Height
        iteration += 1
        ssr = sum_of_square_residual(_predicted_height , _observed_height)
        ssr_vec.append(ssr)
        print("intercept : {} , slope : {} , step_size : {} , iteration : {}".format(intercept,slope,step_size,iteration))

    plot2(intercept,slope,step_size,iteration,ssr,ssr_vec)

def plot2(intercept,slope,step_size,iteration,ssr,ssr_vec):
    fig, axs = plt.subplots(2)
    plot_with_residual(axs[0],Weight,Height,intercept,slope)
    axs[0].grid()
    fig.suptitle('Part 2')
    axs[0].set_title('data set vs intercpt  + slope * weight and residual\nssr : {:.2f} , intercept : {:.2f} ,slope : {:.2f} , iteration : {} , step : {:.4f}'.format(ssr , intercept,slope , iteration,step_size))
    axs[0].set_xlabel("Weight")
    axs[0].set_ylabel("Height")
    axs[1].plot(ssr_vec)
    axs[1].set_title('gradient descent convergence , learn intercept and slope')
    axs[1].set_xlabel('iteration')
    axs[1].set_ylabel('cost function - ssr')
    axs[1].grid()
    plt.tight_layout()
    plt.show()

# part 1 assume slope is 0.64 , compute intercept
def part1_learn_slope_is_constant():
    slope = 0.64
    compute_one_ssr(slope)
    show_ssr_graph_per_intercept(slope)
    print_d_ssr_to_d_intercept(slope)
    gradient_descent_constant_slope(0.64)

def part2_learn():
    gradient_descent()

# main
plot_dataset()
part1_learn_slope_is_constant()
part2_learn()
