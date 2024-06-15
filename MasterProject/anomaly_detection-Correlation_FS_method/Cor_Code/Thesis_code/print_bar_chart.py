# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:43:36 2024

@author: eriks
"""

def print_plot(rf, rf1 ,dt, dt1, ann, ann1, knn, knn1, svm, svm1, lr, lr1):
    
    print( f' \\' + 'addplot coordinates ' + '{' + '({},LR) ({},SVM) ({},DT) ({},ANN) ({},kNN)  ({},RF)'.format(lr,svm,dt,ann,knn,rf) + '}' + '; ')
    print( f' \\' + 'addplot coordinates ' + '{' + '({},LR) ({},SVM) ({},DT) ({},ANN) ({},kNN)  ({},RF)'.format(lr1,svm1,dt1,ann1,knn1,rf1) + '}' + '; \n')






# 42  21 acc

print("Acc")

print_plot(100,	100,
100,	99.49,
100,	100,
100,	99.99,
99.99,	99.99,
99.99,	99.99)

print_plot(100,	100,
99.99,	99.99,
100,	99.99,
100	,99.99,
99.98,	99.98,
99.98,	99.98)


print("Precision")

# 42  22 pre

print_plot(100,	100,
100,	99.98,
100,	100,
100,	99.99,
99.99,	99.99,
99.99,	99.99)


print_plot(100,	100,
99.99,	99.99,
100	,100,
100	,99.99,
99.98,	99.98,
99.98,	99.98)


print("Recall")

# 42  22 rec

print_plot(100,	100,
100,	99.50,
100,	100,
100,	99.99,
99.99,	99.99,
99.99,	99.99)


print_plot(100,	100,
99.99	,99.99,
100	,100,
100	,99.99,
99.98,	99.98,
99.99,	100)

print("F1")

# 42  22 f1

print_plot(100,	100,
100,
	99.74,
100,	100,
100,	99.99,
99.99,	99.99,
99.99,	99.99)

print_plot(100,	100,
99.99,	99.99,
100,	100,
100,	99.99,
99.98,	99.98,
99.99,	99.99)
 




