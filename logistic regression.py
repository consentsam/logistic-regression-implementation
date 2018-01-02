# Here in this example we will classify data
# into two classes using Logistic Regression
# with the help of Sigmoid Function and Applying Newton Method 
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time 
import matplotlib.pyplot as plt
import pickle
import math
# I have put the start_time as a variable to mark down starting time
start_time =time.time()
#The value of eta (learning rate) is tuned to get the appropriate 
#model  
eta=0.01
#Reading x_data as input 
X=np.genfromtxt('q2x.dat')
# Now we are counting number of rows and columns in dataset X
# We are adding 1 to the number of columns because of the presence
# of theta_0 in the equation 
m=X.shape[0]
n=X.shape[1]+1
# Reading Y data 
Y=np.genfromtxt('q2y.dat')
# Now we are setting a stopping parameter so that when the
# difference between the error values decreases to
# value less than stopping parameter then the iteration will stop.
# We do so because in the start the functions start to decrease
# fastly but after some time the functions start to decrease very
# slowly and so the difference between the error functions seems to 
# decrease very slowly and so if we keep a limit that if the 
# difference between them becomes less than stopping then we 
# will assume this as converged
stopping=0.00032
param=np.zeros((n,1))
param_old=np.ones((n,1))

# Initialising Variables
count=0
number_of_iterations=0

# Defining Sigmoid Function 
# if sigmoid_function returns value less than 0.5
# it will be a Class 1 either Class 2 will do
def sigmoid_function (x):
	return (1/(1+math.exp(-x)))

while(np.linalg.norm(param_old-param)>stopping):
	number_of_iterations=number_of_iterations+1
	gradient=np.zeros((n,1))
	H_matrix=np.zeros((n,n))
	for i in range(m):
		xi = np.array([[1],[X[count][0]],[X[count][1]]])
		# H_matrix is Hessian Matrix
		H_matrix= H_matrix -(sigmoid_function(param.transpose().dot(xi)))*(1-sigmoid_function(param.transpose().dot(xi)))*(xi.dot(xi.transpose()))
		# Using Newton's Method
		gradient = gradient + (Y[count]-sigmoid_function((param.transpose()).dot(xi)))*xi
		count=count+1
	param_old=param
	param=param-np.linalg.inv(H_matrix).dot(gradient)
end_time = time.time()
print("Time Elapsed "+ str(end_time-start_time))
print("Number of Iterations "+str(number_of_iterations))
print(param)

# All for Plotting the Graphs 
# UpperColumn denotes the points above the hypothesis 
# function and LowerColumn denotes the points below 
# Hypothesis Function
# Since every X constittes two column and so there are 
# two X's
uppercolumnx1=[]
uppercolumnx2=[]
lowercolumnx1=[]
lowercolumnx2=[]
value=np.zeros((1,1))
for i in range(m):
	xx=np.array([1,X[i][0],X[i][1]])
	value=param.transpose().dot(xx)
	if ((value>0)):
		uppercolumnx1.append(X[i][0])
		uppercolumnx2.append(X[i][1])
	else:
		lowercolumnx1.append(X[i][0])
		lowercolumnx2.append(X[i][1])
plt.plot(uppercolumnx1,uppercolumnx2,'ro')
plt.plot(lowercolumnx1,lowercolumnx2,'bo')

# Writing Equation of Line 

slope=-1*(param[1]/param[2])
intercept=-1*(param[0]/param[2])
x_zero=np.array([0.0806452,6.53226])
lefty=np.zeros((2,1))
lefty=slope*x_zero+intercept
plt.plot(x_zero,lefty)

plt.show()























































