# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:43:36 2024

@author: eriks
"""

RF= "RF"
DT="DT"
ANN="ANN"
kNN="kNN"
SVM="SVM"
LR="LR"


def print_plot(model,org,new, metric, ab, bb , aa, ba):
       
    if metric == "runtime":
        print( f'As for runtime, the feature reduction made the algorithm run from {ab} to {bb} seconds.'.format(aa,bb)+'\n')
        return
   
    if metric == "accuracy":
        print(f"{model} increased the majority of metrics when reducing the features from {org} to {new}.".format(model,org,new))
        if ab==aa and ba<bb:
            print( f'Train {metric} stayed the same at {aa}\%, and test {metric} decreased from {bb}\% to {ba}\%.'.format(metric,aa,ab,metric,ba,bb))
        if ab==aa and ba>bb:
            print( f'Train {metric} stayed the same at {aa}\%, and test {metric} increased from {bb}\% to {ba}\%.'.format(metric,aa,ab,metric,ba,bb))
           
        if ab>aa and ba==bb:
            print( f'The algorithm increased its train {metric} from {aa}\% to {ab}\%, while test {metric} stayed the same at {bb}\%.'.format(metric,aa,ab,metric,ba,bb))
        if ab<aa and ba==bb:
            print( f'The algorithm decreased its train {metric} from {aa}\% to {ab}\%, while test {metric} stayed the same at {bb}\%.'.format(metric,aa,ab,metric,ba,bb))
        if ab==aa and ba==bb:
            print( f'Train {metric} stayed the same at {aa}\%, and test {metric} also stayed the same at {bb}\%.'.format(metric,aa,ab,metric,ba,bb)) 
       
        if ab<aa and ba<bb:
            print( f'The algorithm decreased its train {metric} from {aa}\% to {ab}\%, while test {metric} decreased from {bb}\% to {ba}\%.'.format(metric,aa,ab,metric,ba,bb))
        if ab<aa and bb<ba:
            print( f'The algorithm decreased its train {metric} from {aa}\% to {ab}\%, while test {metric} increased from {bb}\% to {ba}\%.'.format(metric,aa,ab,metric,ba,bb))
        if aa<ab and ba<bb:
            print( f'The algorithm increased its train {metric} from {aa}\% to {ab}\%, while test {metric} decreased from {bb}\% to {ba}\%.'.format(metric,aa,ab,metric,ba,bb))
        if aa<ab and ba>bb:
            print( f'The algorithm increased its train {metric} from {aa}\% to {ab}\%, and test {metric} increased from {bb}\% to {ba}\%.'.format(metric,aa,ab,metric,ba,bb))
    else:
        
        if ab==aa and ba<bb:
            print( f'Train {metric} stayed the same at {aa}\%, while test {metric} decreased from {bb}\% to {ba}\%.'.format(metric,aa,ab,metric,ba,bb))
        if ab==aa and ba>bb:
            print( f'Train {metric} stayed the same at {aa}\%, while test {metric} increased from {bb}\% to {ba}\%.'.format(metric,aa,ab,metric,ba,bb))
           
        if ab>aa and ba==bb:
            print( f'Train {metric} increased from {aa}\% to {ab}\%, while test {metric} stayed the same at {bb}\%.'.format(metric,aa,ab,metric,ba,bb))
        if ab<aa and ba==bb:
            print( f'Train {metric} decreased from {aa}\% to {ab}\%, while test {metric} stayed the same at {bb}\%.'.format(metric,aa,ab,metric,ba,bb))
       
        if ab==aa and ba==bb:
            print( f'Train {metric} stayed the same at {aa}\%, and test {metric} also stayed the same at {bb}\%.'.format(metric,aa,ab,metric,ba,bb))
        
        if ab<aa and ba<bb:
            print(f"Train {metric} decreased from {aa}\% to {ab}\%, and test {metric} decreased from {bb}\% to {ba}\%.")
        if ab<aa and bb<ba:
            print(f"Train {metric} decreased from {aa}\% to {ab}\%, while test {metric} increased from {bb}\% to {ba}\%.")
        if aa<ab and ba<bb:
            print(f"Train {metric} increased from {aa}\% to {ab}\%, while test {metric} decreased from {bb}\% to {ba}\%.")
        if aa<ab and ba>bb:
            print(f"Train {metric} increased from {aa}\% to {ab}\%, and test {metric} increased from {bb}\% to {ba}\%.")

#    print( f' \\' + 'addplot coordinates ' + '{' + '({},LR) ({},SVM) ({},DT) ({},ANN) ({},KNN)  ({},RF)'.format(lr1,svm1,dt1,ann1,knn1,rf1) + '}' + '; \n')

"""
# UNSW_NB15

# Multi

print_plot(RF, 42, 22, "accuracy", 97,	83 ,98.1,	83.3)
print_plot(RF, 42, 22, "precision", 96.7,	77, 98.1	,77)
print_plot(RF, 42, 22, "recall", 98.9,	98.7,99.1,	99.4)
print_plot(RF, 42, 22, "f1-score", 97.8,	83, 98.6,	86.8)
print_plot(RF, 42, 22, "runtime", 18.8,	15.13 , 0,0)

print_plot(DT, 42, 22, "accuracy",94.6,	83.1, 94.7,	85.1)
print_plot(DT, 42, 22, "precision",94.2,	77.2, 94.8	,79.3)
print_plot(DT, 42, 22, "recall", 98.1,	98.2,97.6	,98.6)
print_plot(DT, 42, 22, "f1-score", 96.1,	86.4, 96.2	,87.9)
print_plot(DT, 42, 22, "runtime", 132,	62.3 , 0,0)

print_plot(ANN, 42, 22, "accuracy", 93.75	,81.99 ,94.06	,88.36)
print_plot(ANN, 42, 22, "precision", 86.8	,77.6, 96.1	,83.8)
print_plot(ANN, 42, 22, "recall", 99.2	,98.9,95	,97.7)
print_plot(ANN, 42, 22, "f1-score", 95.5	,85.8 ,95.6	,90.2)
print_plot(ANN, 42, 22, "runtime", 40.3,	33.98 , 0,0)

print_plot(kNN, 42, 22, "accuracy", 95.5	,86.5 ,96.5,	86.7 )
print_plot(kNN, 42, 22, "precision",97.7,	83.2 ,99.9	,84.4)
print_plot(kNN, 42, 22, "recall", 95.6	,94.6,95	,93)
print_plot(kNN, 42, 22, "f1-score", 96.7,	88.5 ,97.4,	88.5)
print_plot(kNN, 42, 22, "runtime", 67.23,	51.2 , 0,0)

print_plot(SVM, 42, 22, "accuracy", 93.5	,81.4, 93.6,	81.4)
print_plot(SVM, 42, 22, "precision", 94,	85.7 ,94,	85.7)
print_plot(SVM, 42, 22, "recall", 93.6,	81.4,93.6,	81.4)
print_plot(SVM, 42, 22, "f1-score", 93.3 ,	80.4 ,93.4,	80.4)
print_plot(SVM, 42, 22, "runtime", 1503.65,	1662.25 , 0,0)

print_plot(LR, 42, 22, "accuracy", 91.5,	76.8 ,93.0	,80.1)
print_plot(LR, 42, 22, "precision", 90.3,	71.6, 91.4,	74.4)
print_plot(LR, 42, 22, "recall", 98	,96,99	,97.4)
print_plot(LR, 42, 22, "f1-score", 94,	82 ,95.1,	84.4)
print_plot(LR, 42, 22, "runtime", 6.4,	5 , 0,0)



# Binary

print_plot(RF, 42, 21, "accuracy", 97,	83 ,98.1,	83.3)
print_plot(RF, 42, 21, "precision", 96.7,	77, 98.1	,77)
print_plot(RF, 42, 21, "recall", 98.9,	98.7,99.1,	99.4)
print_plot(RF, 42, 21, "f1-score", 97.8,	83, 98.6,	86.8)
print_plot(RF, 42, 21, "runtime", 18.8,	15.13 , 0,0)

print_plot(DT, 42, 21, "accuracy",94.6,	83.1, 94.7,	85.1)
print_plot(DT, 42, 21, "precision",94.2,	77.2, 94.8	,79.3)
print_plot(DT, 42, 21, "recall", 98.1,	98.2,97.6	,98.6)
print_plot(DT, 42, 21, "f1-score", 96.1,	86.4, 96.2	,87.9)
print_plot(DT, 42, 21, "runtime", 132,	62.3 , 0,0)

print_plot(ANN, 42, 21, "accuracy", 93.75	,81.99 ,94.06	,88.36)
print_plot(ANN, 42, 21, "precision", 86.8	,77.6, 96.1	,83.8)
print_plot(ANN, 42, 21, "recall", 99.2	,98.9,95	,97.7)
print_plot(ANN, 42, 21, "f1-score", 95.5	,85.8 ,95.6	,90.2)
print_plot(ANN, 42, 21, "runtime", 40.3,	33.98 , 0,0)

print_plot(kNN, 42, 21, "accuracy", 95.5	,86.5 ,96.5,	86.7 )
print_plot(kNN, 42, 21, "precision",97.7,	83.2 ,99.9	,84.4)
print_plot(kNN, 42, 21, "recall", 95.6	,94.6,95	,93)
print_plot(kNN, 42, 21, "f1-score", 96.7,	88.5 ,97.4,	88.5)
print_plot(kNN, 42, 21, "runtime", 67.23,	51.2 , 0,0)

print_plot(SVM, 42, 21, "accuracy", 93.5	,81.4, 93.6,	81.4)
print_plot(SVM, 42, 21, "precision", 94,	85.7 ,94,	85.7)
print_plot(SVM, 42, 21, "recall", 93.6,	81.4,93.6,	81.4)
print_plot(SVM, 42, 21, "f1-score", 93.3 ,	80.4 ,93.4,	80.4)
print_plot(SVM, 42, 21, "runtime", 1503.65,	1662.25 , 0,0)

print_plot(LR, 42, 21, "accuracy", 91.5,	76.8 ,93.0	,80.1)
print_plot(LR, 42, 21, "precision", 90.3,	71.6, 91.4,	74.4)
print_plot(LR, 42, 21, "recall", 98	,96,99	,97.4)
print_plot(LR, 42, 21, "f1-score", 94,	82 ,95.1,	84.4)
print_plot(LR, 42, 21, "runtime", 6.4,	5 , 0,0)


"""
# Ton_IOT
print("--------------------ToN---------------------------")

# Multi

m_ton = 41
mm_ton = 17

print_plot(RF, m_ton, mm_ton, "accuracy", 100 ,	99.9 ,100 ,	99.9)
print_plot(RF, m_ton, mm_ton, "precision", 100,	99.9,100	,99.9 )
print_plot(RF, m_ton, mm_ton, "recall", 100,	99.9,100	,99.9)
print_plot(RF, m_ton, mm_ton, "f1-score", 100	,99.9,100,	99.9 )
print_plot(RF, m_ton, mm_ton, "runtime", 18.663	,18.725 , 0,0)

print_plot(DT, m_ton, mm_ton, "accuracy", 99.95,	97.5 , 99.95,	97.5)
print_plot(DT, m_ton, mm_ton, "precision",99.95,	97.6, 99.95,	97.5 )
print_plot(DT, m_ton, mm_ton, "recall", 99.95,	97.5,99.95,	97.5)
print_plot(DT, m_ton, mm_ton, "f1-score",99.95,	97.2, 99.95,	97.5)
print_plot(DT, m_ton, mm_ton, "runtime", 1.93	,1.383, 0,0)

print_plot(ANN, m_ton, mm_ton, "accuracy", 98.20,	98.19,99.30	,99.28)
print_plot(ANN, m_ton, mm_ton, "precision", 98.24,	98.24, 99.29,	99.28)
print_plot(ANN, m_ton, mm_ton, "recall", 98.19,	98.19,99.29,	99.28)
print_plot(ANN, m_ton, mm_ton, "f1-score", 98.20,	98.19 ,99.29	,99.28)
print_plot(ANN, m_ton, mm_ton, "runtime", 29.07	,42.67, 0,0)

print_plot(kNN, m_ton, mm_ton, "accuracy", 100,	99.8 ,100	,99.6 )
print_plot(kNN, m_ton, mm_ton, "precision",100,	99.8 ,100,	99.6)
print_plot(kNN, m_ton, mm_ton, "recall", 100	,99.8,100,	99.6)
print_plot(kNN, m_ton, mm_ton, "f1-score",100	,99.8 ,100,	99.6)
print_plot(kNN, m_ton, mm_ton, "runtime", 185.674099,	131.692901 , 0,0)

print_plot(SVM, m_ton, mm_ton, "accuracy", 96.7,	96.8, 97.9	,98 )
print_plot(SVM, m_ton, mm_ton, "precision", 96.9	,97 ,97.9	,98)
print_plot(SVM, m_ton, mm_ton, "recall", 96.9	,96.7,97.9,	98)
print_plot(SVM, m_ton, mm_ton, "f1-score", 96.7,	96.9 ,97.9,	98)
print_plot(SVM, m_ton, mm_ton, "runtime", 6984.534249,	6449.844844 , 0,0)

print_plot(LR, m_ton, mm_ton, "accuracy", 85.6,	85.8 , 85.1,	85.3 )
print_plot(LR, m_ton, mm_ton, "precision", 85.2,	85.3, 84.5,	84.7)
print_plot(LR, m_ton, mm_ton, "recall", 85.6	,85.8,	85.1	,85.3)
print_plot(LR, m_ton, mm_ton, "f1-score",85.1,	85.3 ,84.5	,84.7)
print_plot(LR, m_ton, mm_ton, "runtime", 136.65,	98.69, 0,0)

print("-----------------------------------------------------")



# Binary

print_plot(RF, m_ton, mm_ton, "accuracy", 100,	100, 100	,100)
print_plot(RF, m_ton, mm_ton, "precision", 100,	100, 100	,99.9)
print_plot(RF, m_ton, mm_ton, "recall", 100,	99.9, 100,	99.9)
print_plot(RF, m_ton, mm_ton, "f1-score", 100,	99.9, 100,	99.9 )
print_plot(RF, m_ton, mm_ton, "runtime", 14.1,	14.6, 0,0)


print_plot(DT, m_ton, mm_ton, "accuracy", 99.8	,99.2, 99.8,	99.2)
print_plot(DT, m_ton, mm_ton, "precision",99.6	,99.5, 99.6,	99.6 )
print_plot(DT, m_ton, mm_ton, "recall", 99.8	,98.3, 99.8,	98.3)
print_plot(DT, m_ton, mm_ton, "f1-score",99.7	,98.9, 99.7,	98.9)
print_plot(DT, m_ton, mm_ton, "runtime", 1.355,	1.29, 0,0)


print_plot(ANN, m_ton, mm_ton, "accuracy", 98.3	,98.3, 97.6,	97.6)
print_plot(ANN, m_ton, mm_ton, "precision", 96.98,	97.04, 94.10, 94.3)
print_plot(ANN, m_ton, mm_ton, "recall", 98.3	,98.27, 99.35,	99.26)
print_plot(ANN, m_ton, mm_ton, "f1-score", 97.6	,97.6, 96.6,	96.75)
print_plot(ANN, m_ton, mm_ton, "runtime", 45.57,	31.6, 0,0)


print_plot(kNN, m_ton, mm_ton, "accuracy", 100,	99.8, 100,	99.7 )
print_plot(kNN, m_ton, mm_ton, "precision",100,	99.8,100	,99.7)
print_plot(kNN, m_ton, mm_ton, "recall", 100,	99.7, 100,	99.5)
print_plot(kNN, m_ton, mm_ton, "f1-score",100,	99.8, 100,	99.6)
print_plot(kNN, m_ton, mm_ton, "runtime", 185,	152.58, 0,0)


print_plot(SVM, m_ton, mm_ton, "accuracy",94.55,	94.6, 94.3,	94.4 )
print_plot(SVM, m_ton, mm_ton, "precision", 95,	95.1 ,94.3,	94.4)
print_plot(SVM, m_ton, mm_ton, "recall", 94.6	,94.5, 94.3,	94.4)
print_plot(SVM, m_ton, mm_ton, "f1-score", 94.6	,94.6 ,94.4,	94.4)
print_plot(SVM, m_ton, mm_ton, "runtime", 7403,	5589.3, 0,0)

print_plot(LR, m_ton, mm_ton, "accuracy", 88.5,	88.7, 87.7,	87.9)
print_plot(LR, m_ton, mm_ton, "precision", 77.6	,78, 92,	92.2)
print_plot(LR, m_ton, mm_ton, "recall", 94.4,	94.4, 92,	92.2)
print_plot(LR, m_ton, mm_ton, "f1-score",85.2,	85.4, 84	,84.2)
print_plot(LR, m_ton, mm_ton, "runtime", 4.69	,1.50, 0,0)


# BoT_IOT

print("--------------------BOT--------------------------")

# Multi

m_bot = 16
mm_bot = 12

print_plot(RF, m_bot, mm_bot, "accuracy",100,	99.9, 100,	98.6)
print_plot(RF, m_bot, mm_bot, "precision",100,	99.9, 100,	98.8)
print_plot(RF, m_bot, mm_bot, "recall",100,	99.9, 100,	98.6)
print_plot(RF, m_bot, mm_bot, "F1-score",100,	99.9, 100,	98.7)
print_plot(RF, m_bot, mm_bot, "runtime",166.64,	159.75, 0,0)


print_plot(DT, m_bot, mm_bot, "accuracy",100,	99.5, 100,	99.9)
print_plot(DT, m_bot, mm_bot, "precision",100,	99.9, 100,	99.9) 
print_plot(DT, m_bot, mm_bot, "recall",100,	99.5, 100,	99.9)
print_plot(DT, m_bot, mm_bot, "F1-score",100,	99.7, 100,	99.9)
print_plot(DT, m_bot, mm_bot, "runtime",8.01,	6.2  , 0,0)


print_plot(ANN, m_bot, mm_bot, "accuracy",100,	100, 99.99,	99.99 )
print_plot(ANN, m_bot, mm_bot, "precision", 100,	100, 99.99,	99.99)
print_plot(ANN, m_bot, mm_bot, "recall", 100,	100, 99.99,	99.99)
print_plot(ANN, m_bot, mm_bot, "F1-score",100,	100, 99.99,	99.99)
print_plot(ANN, m_bot, mm_bot, "runtime", 277.65,	306.52 , 0,0)


print_plot(kNN,m_bot, mm_bot, "accuracy",  100,	99.9, 100,	99.9 )
print_plot(kNN, m_bot, mm_bot, "precision",100,	99.9, 100,	99.9 )
print_plot(kNN, m_bot, mm_bot, "recall", 100,	99.9, 100,	99.9 )
print_plot(kNN, m_bot, mm_bot, "F1-score",100,	99.9, 100,	99.9 )
print_plot(kNN, m_bot, mm_bot, "runtime",12266,	1106  , 0,0)

print_plot(SVM, m_bot, mm_bot, "accuracy", 99.9,	99.9, 99.9,	99.9)
print_plot(SVM, m_bot, mm_bot, "precision", 99.9,	99.9, 99.9,	99.9)
print_plot(SVM, m_bot, mm_bot, "recall", 99.9,	99.9, 99.9,	99.9)
print_plot(SVM, m_bot, mm_bot, "F1-score", 99.9,	99.9, 99.9,	99.9)
print_plot(SVM, m_bot, mm_bot, "runtime", 594,	586, 0,0)

print_plot(LR, m_bot, mm_bot, "accuracy",99.9,	99.9, 99.9,	99.9)
print_plot(LR, m_bot, mm_bot, "precision", 99.9,	99.9, 99.9,	99.9)
print_plot(LR, m_bot, mm_bot, "recall", 99.9,	99.9, 99.9,	99.9)
print_plot(LR, m_bot, mm_bot, "F1-score",99.9,	99.9, 99.9,	99.9)
print_plot(LR, m_bot, mm_bot, "runtime",163,	152 , 0,0)


print("-----------------------------------------------------")

# Binary

m_bot = 13 
mm_bot = 9

print_plot(RF, m_bot, mm_bot, "accuracy",100,100,100,100)
print_plot(RF, m_bot, mm_bot, "precision",100,100,100,100)
print_plot(RF, m_bot, mm_bot, "recall", 100,100,100,100)
print_plot(RF, m_bot, mm_bot, "F1-score",100,100,100,100)
print_plot(RF, m_bot, mm_bot, "runtime", 77.5	,77.9, 0,0)


print_plot(DT, m_bot, mm_bot, "accuracy",100,	99.4,99.9,	99.93)
print_plot(DT, m_bot, mm_bot, "precision",100	,99.9,99.9,	99.9)
print_plot(DT, m_bot, mm_bot, "recall",100	,99.5, 99.9	,99.9)
print_plot(DT, m_bot, mm_bot, "F1-score",100	,99.7,99.9,	99.9 )
print_plot(DT, m_bot, mm_bot, "runtime", 4.12,	15.99 , 0,0)

print_plot(ANN, m_bot, mm_bot, "accuracy",100,100,100,100 )
print_plot(ANN, m_bot, mm_bot, "precision",100,100,100,100 )
print_plot(ANN, m_bot, mm_bot, "recall", 100,100,100,100)
print_plot(ANN, m_bot, mm_bot, "F1-score",100,100,100,100 )
print_plot(ANN, m_bot, mm_bot, "runtime", 313,	276.5 , 0,0)

print_plot(kNN, m_bot, mm_bot, "accuracy",100,	99.9, 100,	99.9  )
print_plot(kNN, m_bot, mm_bot, "precision",100,	100, 100,	99.9 )
print_plot(kNN, m_bot, mm_bot, "recall", 100,	99.9, 100,	99.9  )
print_plot(kNN, m_bot, mm_bot, "F1-score",100,	99.9, 100,	99.9  )
print_plot(kNN, m_bot, mm_bot, "runtime",1320.5,	1753.69  , 0,0)

print_plot(SVM, m_bot, mm_bot, "accuracy",99.9,	99.9, 99.9,	99.9 )
print_plot(SVM, m_bot, mm_bot, "precision",99.9,	99.9, 99.9,	99.9 )
print_plot(SVM, m_bot, mm_bot, "recall", 99.9,	99.9,99.9	,99.9 )
print_plot(SVM, m_bot, mm_bot, "F1-score",99.9,	99.9, 99.9,	99.9 )
print_plot(SVM, m_bot, mm_bot, "runtime",265.5,	395.97  , 0,0)

print_plot(LR, m_bot, mm_bot, "accuracy",99.9,	99.9, 99.9,	99.9)
print_plot(LR, m_bot, mm_bot, "precision",99.9,	99.9, 99.9,	99.9 )
print_plot(LR, m_bot, mm_bot, "recall", 99.9,	99.9,99.9	,99.9)
print_plot(LR, m_bot, mm_bot, "F1-score",99.9,	99.9, 99.9,	100)
print_plot(LR, m_bot, mm_bot, "runtime",9.3,	8.2  , 0,0)

print("-----------------------------------------------------")
