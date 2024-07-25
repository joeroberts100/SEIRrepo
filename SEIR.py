# This script models the spread of a disease using a modified SEIR 
# (Susceptible-Exposed-Infectious-Recovered) model, 
# implementing the Euler method for numerical integration.

# This document is based on lectures by Dr Pavel Buividovich 

# The SEIR Model is a system of first-order ODEs, whereby
# dS/dt = (-beta * S * I) + (xi * R)      
# dE/dt = (beta * S * I) - (sigma * E) 
# dI/dt = (sigma * E) - (gamma * I)
# dR/dt = (gamma * I) - (xi * R) 

# recap of Euler Method:
# let dS/dt = F(t, S)  
# then S_i+1 = S_i + dt*F(t_i, S_i)
# where F(t_i, S_i) = (-beta * S_i * I_i) + (xi * R_i)
# therefore S_i+1 = S_i + dt * ( (-beta * S_i * I_i) + (xi * R_i) )
# then changed to python indexing syntax with SEIR arrays we arrive at:
# S[i+1] = S[i] + dt * ( (-beta * S[i] * I[i]) + (xi * R[i]) )


from numpy import *

#Parameters of the system
R0 = 2.5 # basic reproduction number (average number of secondary infections caused by one infected person)
gamma = 0.1  # recovery rate 
beta  = R0*gamma  # transmission rate 
sigma = 1.0  # rate at which the exposed (infected) become infectious 
xi    = 0.0 # rate at which people loose immunity and move from the recovered category back to the susceptible category.
# accounts for second wave of infection

#Parameters of the discretization grid
tmax = 500 # maximum simulation time 
dt = 0.5   # time step for Euler Method
nmax = int(ceil(tmax/dt))+1 #Number of time steps

#We need +1 step to get to tmax rather than tmax - dt
#Initialize the arrays which will contain the solution
S = full(nmax, 0.0)    # individuals who are susceptible
E = full(nmax, 0.0)    # individuals who are exposed (infected)
I = full(nmax, 0.0)    # individuals who are infectious (infected with the disease and are capable of infecting others)
R = full(nmax, 0.0)    # individuals who can be removed from the system (recovered/immune/deceased)

T = full(nmax,0.0)
T[0] = 0.0         # array to store time values 

I[0] = 0.0001  # Initial fraction of the population that is infected.
S[0] = 1.0 - I[0]   # Initial fraction of the population that is susceptible.

#Iterate the Euler method,
#expressing (i+1)'th array elements
#in terms of i'th array elements
for i in range(0,nmax-1) :
    S[i+1] = S[i] + dt*(-beta*S[i]*I[i] + xi*R[i])
    E[i+1] = E[i] + dt*( beta*S[i]*I[i] - sigma*E[i])
    I[i+1] = I[i] + dt*(sigma*E[i] - gamma*I[i])
    R[i+1] = R[i] + dt*( gamma*I[i] - xi*R[i])
    #Also assign values to the array of time values
    T[i+1] = T[i] + dt

for i in range(1,nmax-1) :
    if ((I[i]>I[i+1]) and (I[i]>I[i-1])) :
        print("A peak of height %3.1f at t = %3.1f"%(I[i], T[i]))
        # Checks for local maxima in the I array (infection peaks), 
        # printing the height and time of each peak.

#Plotting the data from all the arrays
from matplotlib import pyplot as plt
plt.xlabel('t')
plt.ylabel('S(t),E(t),I(t),R(t)')
plt.plot(T, S, 'bo', label='S(t)')
plt.plot(T, E, 'mo', label='E(t)')
plt.plot(T, I, 'ro', label='I(t)')
plt.plot(T, R, 'go', label='R(t)')
plt.show()
# blue circles ('bo') - susceptible 
# magenta circles ('mo') - exposed
# red circles ('ro') - infectious 
# green circles ('go') - removed

# During COVID-19, the news often referenced a figure 'R'.
# This represents the effective reproduction number:
# the average number of secondary infections produced by an 
# infectious individual in a population that is not fully susceptible.
# (this is different to our number R from the SEIR system)

# It can be evaluated by:
# R = R0 * S, where R0 is the same R0 as above. 
