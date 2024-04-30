import numpy as np
import matplotlib.pyplot as plt
import math


# 1-2-1 NN
def activation(x):
    # soft plus
    ex = math.e ** x
    ret = math.log(1 + ex)
    return ret
    # return math.log(math.e ** x + 1)
y1 = 0
y2 = 0

def calcTopNN(dosage, weight3):
    global y1
    w1 = 3.34
    w3 = weight3
    b1 = -1.43

    x1 = dosage * w1 + b1
    y1 = activation(x1)
    # print(y1)
    val1 = y1 * w3
    # print(val1)
    return (val1)


def calcBottomNN(dosage, weight4):
    global y2
    w2 = -3.53
    w4 = weight4
    b2 = .57

    x2 = dosage * w2 + b2
    y2 = activation(x2)
    # print(y2)
    val2 = y2 * w4
    # print(val2)
    return (val2)


def calcNN(dosage, bias3, weight3, weight4):
    # b3 = 2.61
    b3 = bias3
    val1 = calcTopNN(dosage, weight3)
    val2 = calcBottomNN(dosage, weight4)
    sum = val1 + val2
    # print(sum)
    # return (sum)
    return (sum + b3)


xpoints = np.array([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
yvals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Here are the observed yvalues.
yvalsObs0 = 0.006334761940589573
yvalsObs5 = 1.000980681806214
yvalsObs10 = -0.00486963403567886


weight3 = 0.36
weight4 = 0.63

# weight3 = .36
# weight4 = .63

notDone = True
while notDone:
    val = np.random.randn()
    if val < 1 and val > -1:
        weight3 = val
        notDone = False
notDone = True
while notDone:
    val = np.random.randn()
    if val < 1 and val > -1:
        weight4 = val
        notDone = False

# Here are the predicted yvalues.
yvalsPred0 = calcNN(xpoints[0], 0, weight3, weight4)
yvalsPred5 = calcNN(xpoints[5], 0, weight3, weight4)
yvalsPred10 = calcNN(xpoints[10], 0, weight3, weight4)


print(weight3)
print(weight4)

learningRate = .1
bias3 = 0
stepSize = -100
while (stepSize < -.005):
    yvals[0] = calcNN(xpoints[0], bias3, weight3, weight4)
    yvals[1] = calcNN(xpoints[1], bias3, weight3, weight4)
    yvals[2] = calcNN(xpoints[2], bias3, weight3, weight4)
    yvals[3] = calcNN(xpoints[3], bias3, weight3, weight4)
    yvals[4] = calcNN(xpoints[4], bias3, weight3, weight4)
    yvals[5] = calcNN(xpoints[5], bias3, weight3, weight4)
    y12 = y1
    y22 = y2
    yvals[6] = calcNN(xpoints[6], bias3, weight3, weight4)
    yvals[7] = calcNN(xpoints[7], bias3, weight3, weight4)
    yvals[8] = calcNN(xpoints[8], bias3, weight3, weight4)
    yvals[9] = calcNN(xpoints[9], bias3, weight3, weight4)
    yvals[10] = calcNN(xpoints[10], bias3, weight3, weight4)
    y13 = y1
    y23 = y2
    # Put Code Here************
    print("pre y1: " + str(y1))
    print("pre y2: " + str(y2))

    SSR = -2 * (yvalsObs0 - (yvalsPred0 + bias3)) * 1
    SSR += -2 * (yvalsObs5 - (yvalsPred5 + bias3)) * 1
    SSR += -2 * (yvalsObs10 - (yvalsPred10 + bias3)) * 1

    stepSize = SSR * learningRate
    bias3 = bias3 - stepSize
    print("stepSize: " + str(stepSize))
    print("SSR: " + str(SSR))
    print("post bias3: " + str(bias3))

    dSSR_dw3 = -2 * (yvalsObs0 - (yvalsPred0 + 0)) * 1
    dSSR_dw3 += -2 * (yvalsObs5 - (yvalsPred5 + y12)) * 1
    dSSR_dw3 += -2 * (yvalsObs10 - (yvalsPred10 + y13)) * 1
    print("dSSR_dw3: " + str(dSSR_dw3))

    # dSSR_dw4 = ???
    # ???

    ypoints = np.array(yvals)
    plt.plot(xpoints, ypoints, marker="o", color="red")


    b3Actual = 2.61
    yvals[0] = calcNN(xpoints[0], b3Actual, -1.22, -2.3)
    yvals[1] = calcNN(xpoints[1], b3Actual, -1.22, -2.3)
    yvals[2] = calcNN(xpoints[2], b3Actual, -1.22, -2.3)
    yvals[3] = calcNN(xpoints[3], b3Actual, -1.22, -2.3)
    yvals[4] = calcNN(xpoints[4], b3Actual, -1.22, -2.3)
    yvals[5] = calcNN(xpoints[5], b3Actual, -1.22, -2.3)
    yvals[6] = calcNN(xpoints[6], b3Actual, -1.22, -2.3)
    yvals[7] = calcNN(xpoints[7], b3Actual, -1.22, -2.3)
    yvals[8] = calcNN(xpoints[8], b3Actual, -1.22, -2.3)
    yvals[9] = calcNN(xpoints[9], b3Actual, -1.22, -2.3)
    yvals[10] = calcNN(xpoints[10], b3Actual, -1.22, -2.3)
    # print (yvals[0])
    # print (yvals[5])
    # print (yvals[10])

    ypoints = np.array(yvals)
    plt.plot(xpoints, ypoints, marker="o", color="black")
    plt.show()
