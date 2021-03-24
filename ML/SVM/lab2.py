import numpy as np
import random as rd
import math 
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.patches as pth
from scipy.optimize.optimize import main

def kernel_function(X1, X2):
    X1t = np.transpose(X1)
    if kernel_params["type"] == "linear":
        return np.dot(X1t, X2)
    elif kernel_params["type"] == "polynomial":
        return np.power(np.dot(X1t, X2), kernel_params["p"])
    elif kernel_params["type"] == "RBF":
        return math.exp(-(np.linalg.norm(X1-X2, 2)**2/(2*kernel_params["sigma"]**2)))

def p_matrix(inputs, targets, N):
    p_matrix = np.zeros([N, N])
    for i in range(0, N):
        for j in range(0, N):
            p_matrix[i][j] = targets[i] * targets[j] * kernel_function(inputs[i], inputs[j])
    return p_matrix

def objective(alpha):
    result = 0
    for i in range(0, N):
        for j in range(0, N):
            result += 0.5 * alpha[i]*alpha[j] * P_matrix[i][j] 
        result -= alpha[i]
    return result

def zerofun(alpha):
    return np.dot(np.transpose(alpha), targets)

def extract_nonzero_alpha(alpha, inputs, targets):
    threshold = 10**-5
    nonzero_inputs = []
    nonzero_targets = []
    nonzero_alpha = []
    for i in range(0, len(alpha)):
        if alpha[i] > threshold:
            nonzero_inputs.append(inputs[i])
            nonzero_targets.append(targets[i])
            nonzero_alpha.append(alpha[i])
    return  nonzero_inputs, nonzero_targets, nonzero_alpha

def calculate_b(alpha, inputs, targets, C):
    si = 0
    for i in range(len(alpha)):
        if alpha[i] < C:
            si = i
            break
    b = 0
    for i in range(len(alpha)):
        b += alpha[i] * targets[i] * kernel_function(inputs[si], inputs[i])
    b -= targets[si]
    return b

def indicator(sv, alpha, inputs, targets, b):
    sm = 0
    for i in range(len(alpha)):
        sm += alpha[i] * targets[i] * kernel_function(sv, inputs[i])
    sm -= b
    return sm


if __name__ == '__main__':

    np.random.seed(100)

    kernel_params = {"type": "linear", "p": 2, "sigma": 0.3}
    classA_params = {"numbers": [30, 30], "center": [[1.5, 0.5],[-1.5, 0.5]], "deviation": [0.2, 0.2]}
    classB_params = {"numbers": 60, "center": [0.0, -0.5], "deviation": 0.2}

    classA = np.concatenate((
        np.random.randn(classA_params["numbers"][0], 2)*classA_params["deviation"][0] + classA_params["center"][0], 
        np.random.randn(classA_params["numbers"][1], 2)*classA_params["deviation"][1] + classA_params["center"][1]))
    classB = np.random.randn(classB_params["numbers"], 2)*classB_params["deviation"] + classB_params["center"]
        
    inputs = np.concatenate((classA, classB))
    targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))

    N = inputs.shape[0]

    permute = list(range(N))
    rd.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]

    P_matrix = p_matrix(inputs, targets, N)
    C = 0.1
    #minimize
    B = [(0, C) for b in range(N)]
    start = np.zeros(N)
    XC = {'type': 'eq', 'fun': zerofun}
    ret = minimize(objective, start, bounds=B, constraints=XC)
    alpha = ret['x']
    print("\n""ret:", ret)

    nonzero_inputs, nonzero_targets, nonzero_alpha = extract_nonzero_alpha(alpha, inputs, targets)
    b = calculate_b(nonzero_alpha, nonzero_inputs, nonzero_targets, C)
    print("\n""SVM with C={}, alpha={}".format(C, nonzero_alpha))

    #plotting 
    plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.', label="Class A")
    plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.', label="Class B")
    Support_Vector_A = [p for p in nonzero_inputs if p in classA]
    Support_Vector_B = [p for p in nonzero_inputs if p in classB]
    plt.plot([p[0] for p in Support_Vector_A], [p[1] for p in Support_Vector_A], 'bx', label="Support Vector A")
    plt.plot([p[0] for p in Support_Vector_B], [p[1] for p in Support_Vector_B], 'rx', label="Support Vector A")
    plt.axis('equal')

    #plotting the title
    if kernel_params["type"] == "linear":
        plt.title("Linear Kernel")
    elif kernel_params["type"] == "polynomial":
        plt.title("Polynomial Kernel(p={})".format(kernel_params["p"]))
    elif kernel_params["type"] == "RBF":
        plt.title("RBF Kernel({})".format(kernel_params["sigma"]))
    
    #plotting the setting
    text = "class A:\n"
    for i in range(0, 2):
        text += "Part {}\n".format(i)
        text += "numbers = {}\n".format(classA_params["numbers"][i])
        text += "center = {}\n".format(classA_params["center"][i])
        text += "deviation = {}\n".format(classA_params["deviation"][i])
    text += "\nclassB\n"
    text += "numbers = {}\n".format(classB_params["numbers"])
    text += "center = {}\n".format(classB_params["center"])
    text += "deviation = {}\n".format(classB_params["deviation"])
    text += "\nBounds\n"
    text += "C = {}".format(C)
    plt.text(4, -4, text, fontsize=8)
    
    #plotting the decision boundary
    xgrid = np.linspace(-5, 5)
    ygrid = np.linspace(-4, 4)
    grid = np.array([[indicator(np.array([x, y]), nonzero_alpha, nonzero_inputs, nonzero_targets, b)
            for x in xgrid]
            for y in ygrid])
    plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))
    
    classA_patch = pth.Patch(color='blue', label='classA')
    classB_patch = pth.Patch(color='red', label='classB')
    decision_boundry_patch = pth.Patch(color='black', label='decision boundry')
    plt.legend(handles=[classA_patch, classB_patch, decision_boundry_patch])

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('decision boundry plt.pdf')
    plt.show()

