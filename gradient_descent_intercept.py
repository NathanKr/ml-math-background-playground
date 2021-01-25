import numpy as np
import matplotlib.pyplot as plt


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
    # print("residual : ",residual)
    return np.dot(residual,residual) # sum the square


def show_ssr_graph_per_intercept(slope):
    vec_intercept = np.arange(0,2,0.1)
    vec_ssr = []
    for intercept in vec_intercept:
        _predicted_height = predicted_height(intercept,slope)
        _observed_height = Height
        ssr = sum_of_square_residual(_predicted_height,_observed_height)
        vec_ssr.append(ssr)

    plt.plot(vec_intercept,vec_ssr,'o')    
    plt.title('ssr vs intercept for slope = {}'.format(slope))
    plt.xlabel("intercept")
    plt.ylabel("some of squre residual")
    plt.grid()
    plt.show()

def compute_one_ssr(slope):
    intercept = 0 # initial guess
    _predicted_height = predicted_height(intercept,slope)
    _observed_height = Height
    ssr = sum_of_square_residual(_predicted_height,_observed_height)
    print('ssr : ' , ssr)

# ssr is the sum over (y-h)^2
# d_ssr / d_intercept is equal by the chain rule to (d_ssr / d_h) * (d_h / d_intercept)
# d_ssr / d_h -> sum over 2(y-h)*(-1)
# d_h / d_intercept = d(intercept + slope * weight) / d_intercept --> 1 
# so  d_ssr / d_intercept is equal to the sum over 2(y-h)*(-1)  but h = intercept + slope * weight so
# d_ssr / d_intercept is equal to the sum over 2 ( y - (intercept + slope * weight))*(-1) or equivalently
# d_ssr / d_intercept is equal to the sum over 2 ( observed_height - (intercept + slope * weight))(-1)
def d_ssr_to_d_intercept(slope,intercept):
    _observed_height = Height
    _predicted_height = intercept + slope * Weight
    error = _observed_height - _predicted_height
    return 2 * np.sum(error) *(-1)

def print_d_ssr_to_d_intercept(slope):
    # derivative become smaller
    print("d_ssr/d_intercept @ intercept = 0 : ",d_ssr_to_d_intercept(slope,0))
    print("d_ssr/d_intercept @ intercept = 0.5 : ",d_ssr_to_d_intercept(slope,0.5))
    print("d_ssr/d_intercept @ intercept = 0.8 : ",d_ssr_to_d_intercept(slope,0.8))
    print("d_ssr/d_intercept @ intercept = 0.9 : ",d_ssr_to_d_intercept(slope,0.9))
    print("d_ssr/d_intercept @ intercept = 0.95 : ",d_ssr_to_d_intercept(slope,0.95))
    print("d_ssr/d_intercept @ intercept = 1.05 : ",d_ssr_to_d_intercept(slope,1.05))

def gradient_descent(slope):
    intercept = 0
    learning_rate = 0.1
    step_size = 1 # just a value to enter the loop
    min_step_size = 0.001
    iteration = 0

    while abs(step_size) > min_step_size:
        step_size = d_ssr_to_d_intercept(slope,intercept) * learning_rate
        intercept = intercept - step_size # i did not see a proof for this
        _predicted_height = predicted_height(intercept,slope)
        _observed_height = Height
        iteration += 1
        ssr = sum_of_square_residual(_predicted_height , _observed_height)
        plt.plot(Weight, Height,'o',Weight,_predicted_height)
        plt.grid()
        plt.title('data set vs intercpt  + slope * weight \nssr : {:.2f} , intercept : {:.2f} , iteration : {} , step : {:.4f}'.format(ssr , intercept , iteration,step_size))
        plt.xlabel("Weight")
        plt.ylabel("Height")
        plt.show()


# main
plot_dataset()
# part 1 assume slope is 0.64 , compute intercept
slope = 0.64
compute_one_ssr(slope)
show_ssr_graph_per_intercept(slope)
print_d_ssr_to_d_intercept(slope)
gradient_descent(0.64)
