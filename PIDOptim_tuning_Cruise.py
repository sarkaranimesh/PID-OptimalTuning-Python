import numpy as np
from scipy.optimize import minimize, LinearConstraint, OptimizeResult
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import math


class PID:
    def __init__(self, Kp,Ki,Kd,Kaw,T_C,T,max,min,max_rate):
        self.Kp = Kp # proportional gain
        self.Ki = Ki # integral gain
        self.Kd = Kd # derivative gain
        self.Kaw = Kaw # anti windup gain
        self.T_C = T_C # time constant for derivative filtering
        self.T = T # time step
        self.max = max # max command saturation limit
        self.min = min # min command saturation limit
        self.max_rate = max_rate #max rate of change of command
        self.integral = 0 # integral term
        self.error_prev = 0 # previous error
        self.deriv_prev = 0 # previous derivative
        self.command_sat = 0 # current saturated command
        self.command_sat_prev = 0# previous saturated command
        self.command_prev = 0 # previous command value

    def Step(self,measurement,setpoint):
        """ execute a step of PID controller.
        Inputs:
        measurement : current measurement
        setpoint : desired set point
        """
        #calculate error
        error = setpoint - measurement

        # update integral term with anti-windup
        self.integral += self.Ki*error*self.T + self.Kaw*(self.command_sat_prev - self.command_prev)

        #calculate filtered derivative
        deriv_filt = (err - self.error_prev + self.T_C*self.deriv_prev)/(self.T + self.T_C)
        self.error_prev = error
        self.deriv_prev = deriv_filt

        # calculate command using PID equation
        command = self.Kp*error + self.Kd*deriv_filt + self.Ki*self.integral

        # update previous command
        self.command_prev = command

        # saturate command
        if command > self.max:
            self.command_sat = self.max
        elif command < self.min:
            self.command_sat = self.min
        else:
            self.command_sat = command

        # apply rate limt
        if self.command_sat > self.command_sat_prev + self.max_rate*self.T:
            self.command_sat = self.command_sat_prev + self.max_rate*self.T
        elif self.command_sat < self.command_sat_prev - self.max_rate*self.T:
            self.command_sat = self.command_sat_prev - self.max_rate*self.T

        # store previous saturated command
        self.command_sat_prev = self.command_sat


class vehicle:
    """ This vehicle function represents the dynamics of a car
        Fd = Fg + Fr + Fa
        where Fg = force acting on the car due to gravity
        Fr = rolling frictional force
        Fa = aero-dynamic drag

        """
    def __init__(self,m,Cd,F_max_0,F_max_max, v_max, g, T,Cr):
        self.m = m # mass
        self.b =b  # aerodynamic drag coefficient
        self.v_max = v_max # max speed in mps
        self.F_max_0 = F_max_0 # max force applied to the car by the powertrain at 0 speed
        self.F_max_max = F_max_max # max force applied to the car by the powertrain at v_max speed
        self.T = T # time stesp
        self.v = 0 #speed of the car
        self.g = g #gravity mpss
        self.Cr = Cr # coefficient of rolling friction

    def Step(self, F,theta):
        """ update the speed of the car baaed on the applied force F and the slop angle theta"""
        # max force applied by the powertrain as a function of speed
        v_to_F_max_x_axis = [0,self.v_max]
        F_max_y_axis = [self.F_max_0, self.F_max_max]

        if self.v < v_to_F_max_x_axis[0]:
            F_max = F_max_y_axis[0]
        elif self.v > v_to_F_max_x_axis[-1]:
            F_max = F_max_y_axis[-1]
        else:
            F_max = np.interp(self.v, v_to_F_max_x_axis,F_max_y_axis)


        # saturate input force
        if F > F_max:
            F_sat = F_max
        elif F < 0:
            F_sat = 0
        else:
            F_sat = F

        # calculate derivative dv/dt i.e. acceleration
        Fg = self.m*self.g*math.sin(theta) # gravitational forces
        Fr = self.m*self.g*self.Cr # rolling friction assuming velocity is positive in global coordinate system
        Fa = b*self.v*self.v # aerodynamic drag

        dv_dt = (F_sat - Fg - Fr - Fa)/self.m

        #update the speed by integrating the derivative using the time step T
        self.v += dv_dt*self.T

def Simulation(x,time_step,end_time,m,b,F_max_0,F_max_max, v_max, uphill ):
    """ Simulate the PID control of a car with the given parameters

    Returns:
        (t,stp,v,command,theta): arrays of time,setpoints,positions,commands and slope angle"""
    length = round(end_time/time_step)
    t= np.zeros(length)
    stp = np.zeros(length)
    v =np.zeros(length)
    command = np.zeros(length)
    theta = np.zeros(length)

    # Assuming  a PI controller with anti- windup gain
    [Kp,Ki,Kaw] = x
    Kd = 0
    T_c = 0
    # initialize the PID controller
    pid = PID(Kp,Ki,Kd,Kaw,T_C,time_step,F_max_0,0,300000)

    #initialize the car with given parameters
    car = vehicle(m,b,F_max_0,F_max_max,v_max,9.81,time_step,Cr)

    # iterate through time steps
    for idx in range(0,length):
        t[idx] = idx*time_step
        # set setpoint
        stp[idx] = 42
    # simulating a variable slope scenario
        if t[idx] < end_time/3 or uphill==0:
            theta[idx] = 0
        elif t[idx] <end_time*2/3:
            theta[idx] = 10*math.pi/180
        else:
            theta[idx] = 20*math*pi/180

        # execute the control loop
        v[idx] = car.v
        pid.Step(v[idx],theta[idx])
        command[idx] = pid.command_sat
        car.Step(command[idx],theta[idx])
    return (t,stp,v,command,theta)

def Cost(x, time_step, end_time, m, b, F_max_0, F_max_max, v_max, We, Wu, uphill):

    """ Calculate the cost function for a given set of parameter.
        Inputs:
        x : PID parameters [Kp,Ki,Kd,Kaw,T_C]
        We : weight on control error
        Wu: weight on control effort

        returns:
        cost : scalar
    """






























