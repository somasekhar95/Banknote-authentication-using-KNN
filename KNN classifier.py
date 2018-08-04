############# Somasekhar Suryadevara USC ID 3461071540 ########################

import numpy as np
import math
#import numpy.linalg.pinv as psuedo_inv
import matplotlib.pyplot as plt
import pylab
import seaborn as sns
import scipy.spatial.distance
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import validation_curve
import itertools
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as reqd_score


######read data from .txt file. Prepare data, test & train data, labels.##########
path = 'D:\Docs_required\EE559\HW1\Bank_note.txt'
# modify the path above accordingly. Give the path location to the dataset
# which is present on your local disk
data = []
label = []
full_data = []
c1=0
c2=0
train = []
test = []
with open(path) as f:
    for line in f:
        numbers_float = list(map(float, line.split(',')))
        full_data.append(numbers_float)
        data.append(numbers_float[:4])
        label.append(numbers_float[-1])
        if(c1 < 200 and numbers_float[-1] == 0):
            test.append(numbers_float)
            c1 = c1 + 1
        elif(c2 < 200 and numbers_float[-1] == 1):
            test.append(numbers_float)
            c2 = c2 + 1
        else:
            train.append(numbers_float)


#print(data)
train_data_np = np.array(train)
print(train_data_np.shape)
print(train_data_np)
test_data_np = np.array(test)
print(test_data_np.shape)
print(test_data_np)                                                                    

lis = np.reshape(label, (1,1372)).tolist()
print(len(lis))
data_by_features = list(zip(*data))
#print(data_by_features)

colors = ['red' if l == 0 else 'green' for l in label]
####### Create subplots for scatter plot ################
fig = plt.figure()
au = fig.add_subplot(3, 2, 1)
av = fig.add_subplot(3, 2, 2)
aw = fig.add_subplot(3, 2, 3)
ax = fig.add_subplot(3, 2, 4)
ay = fig.add_subplot(3, 2, 5)
az = fig.add_subplot(3, 2, 6)
f_one = []
f_two = []
f_three = []
f_four = []
count = 0 
for data in data_by_features:
    #print(data)
    #print(color)
    #print(group)
    count = count +1
    w, x, y, z = data_by_features
    f_one.append(w)
    f_two.append(x)
    f_three.append(y)
    f_four.append(z)
    if count == 1:
        break

print('count is ',count)
au.scatter(f_one, f_two, alpha=0.8, c=colors)
au.set_title('scatter plot b/w feature one and feature two. red-class 0, green-class 1')
av.scatter(f_one, f_three, alpha=0.8, c=colors)
av.set_title('scatter plot b/w feature one and feature three. red-class 0, green-class 1')
aw.scatter(f_one, f_four, alpha=0.8, c=colors)
aw.set_title('scatter plot b/w feature one and feature four. red-class 0, green-class 1')
ax.scatter(f_two, f_three, alpha=0.8, c=colors)
ax.set_title('scatter plot b/w feature two and feature three. red-class 0, green-class 1')
ay.scatter(f_two, f_four, alpha=0.8, c=colors)
ay.set_title('scatter plot b/w feature two and feature four. red-class 0, green-class 1')
az.scatter(f_three, f_four, alpha=0.8, c=colors)
az.set_title('scatter plot b/w feature three and feature four. red-class 0, green-class 1')

plt.show()

############# For Box Plots ############################

dfull = pd.DataFrame.from_records(full_data, columns = ['variance','skewness','curtosis','entropy','Class'])
print(dfull)

x=dfull[['variance','skewness','curtosis','entropy']]
y=dfull['Class']

data=pd.DataFrame(x,columns=['variance','skewness','curtosis','entropy'])
data['Class']=y
data.boxplot(by='Class')

plt.show()

############## KNN classifier for varying values of K ###############
mse_arr = []
acc = []
train_mse_arr = []
tr_acc = []
k = [i for i in range(1,901,1)]

train_data = train_data_np[:,:4]
train_labels = train_data_np[:, -1]
test_data = test_data_np[:,:4]
test_labels = test_data_np[:,-1]
################# Learning curve ###############################################################################
N = [i for i in range(50,801,50)]
print('Varying size of training set',N)

d0 = []
l0 = []
d1 = []
l1 = []

for i in range(len(train_data)):
    if train_labels[i] == 0:
        d0.append(train_data[i,:])
        l0.append(train_labels[i])
    else:
        d1.append(train_data[i,:])
        l1.append(train_labels[i])

d0 = np.array(d0)
l0 = np.array(l0)
d1 = np.array(d1)
l1 = np.array(l1)

best_err_some_k = []
for tr in N:
    iter_mse_arr = []
    X_tr = np.concatenate((d0[:tr//2,:], d1[:tr//2,:]), axis=0)
    X_la = np.concatenate((l0[:tr//2], l1[:tr//2]), axis=0)
    for neigh in range(1,tr,40):
        iter_mse = 0.0
        #iter_tr_mse = 0.0
        iter_knn = KNeighborsClassifier(n_neighbors=neigh, p=2)
            
        iter_knn.fit(X_tr, X_la)
        predictions = iter_knn.predict(test_data)
        #train_predict = iter_knn.predict(X_tr)

        #mse = ((((float)(predictions - actual)) ** 2).sum()) / (float)(len(predictions))
        iter_mse = 1.0 - accuracy_score(test_labels,predictions)
        iter_mse_arr.append(iter_mse)
        
    best_iter_idx = []  
    for i in range(len(iter_mse_arr)):
        if iter_mse_arr[i] == min(iter_mse_arr):
            best_iter_idx.append(i)
    best_iter_k = k[min(best_iter_idx)]

    best_iter_knn = KNeighborsClassifier(n_neighbors=best_iter_k, p=2)
    best_iter_knn.fit(X_tr, X_la)
    best_iter_train_predict = best_iter_knn.predict(X_tr)  
    best_iter_predictions = best_iter_knn.predict(test_data)
    best_iter_mse = 0.0
    best_iter_mse = (((best_iter_predictions - test_labels) ** 2).sum()) / (float)(len(best_iter_predictions))
    best_err_some_k.append(best_iter_mse)

print('best test error for each N: ', best_err_some_k)
plt.plot(N,best_err_some_k)
plt.xlabel('varying training data length')
plt.ylabel('best error rate')
plt.title('Learning curve')
plt.show()
##################################################################################################################################################################################

inv = []
for val in k:
    mse = 0.0
    tr_mse = 0.0  
    knn = KNeighborsClassifier(n_neighbors=val, p=2)
    knn.fit(train_data, train_labels)
    predictions = knn.predict(test_data)
    train_predict = knn.predict(train_data)

    mse = 1.0 - accuracy_score(test_labels,predictions)
    acc.append(accuracy_score(test_labels,predictions)*100)
    mse_arr.append(mse)

    #tr_mse = ((((float)(train_predict - train_actual)) ** 2).sum()) / (float)(len(train_predict))
    tr_mse = 1.0 - accuracy_score(train_labels,train_predict)
    tr_acc.append(accuracy_score(train_labels,train_predict)*100)
    train_mse_arr.append(tr_mse)
    

    inv.append(1.0/(float)(val))


############### variation of test and train error with 1/K #################
plt.plot(inv, train_mse_arr, 'g')
plt.plot(inv, mse_arr, 'r')
plt.xlabel('inverse k')
plt.ylabel('error')
plt.title('error vs 1/K. green - training error, red - testing error')
plt.show()

############## KNN for optimal K #######################################
best_idx = []  
for i in range(len(mse_arr)):
    if mse_arr[i] == min(mse_arr):
        best_idx.append(i)

print('Best test mse with euclidean metric and majority polling: ', min(mse_arr))
best_k = k[min(best_idx)]
print('Best k = ', best_k)

best_knn = KNeighborsClassifier(n_neighbors=best_k, p=2)
best_knn.fit(train_data, train_labels)
best_train_predict = best_knn.predict(train_data)  
best_predictions = best_knn.predict(test_data)
################ Confusion matrix ######################################
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j],'.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

target_names = ['class 0', 'class 1']
cnf_matrix = (confusion_matrix(test_labels, best_predictions))
np.set_printoptions(precision=2)
plt.figure()
# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=target_names, title='Confusion matrix')

############## precision, F-score, true_positive, true_negative ########################
TP = cnf_matrix[0][0]
FP = cnf_matrix[0][1]
FN = cnf_matrix[1][0]
TN = cnf_matrix[1][1]

TPR = TP/(TP+FN)
TNR = TN/(TN+FP)
F1 = 2*TP/(2*TP+FP+FN)
precision = TP/(TP+FP)
print('True positive rate: ', TPR)
print('True negative rate: ', TNR)
print('Precision: ', precision)
print('F1 score: ', F1)

########################## Various distance metrics #####################
k = [i for i in range(1,901,10)]

mse_arr_mh = []
mse_arr_cb = []
mse_arr_mah = []

#manhattan, chebyshev, mahalanobis
for val in k:
    #manhattan
    mse_mh = 0.0
    knn_mh = KNeighborsClassifier(n_neighbors=val, p=1)
    knn_mh.fit(train_data, train_labels)
    predictions_mh = knn_mh.predict(test_data)
    mse_mh = 1.0 - accuracy_score(test_labels,predictions_mh)
    mse_arr_mh.append(mse_mh)
    #chebyshev
    mse_cb = 0.0
    knn_cb = KNeighborsClassifier(n_neighbors=val, metric = 'chebyshev')
    knn_cb.fit(train_data, train_labels)
    predictions_cb = knn_mh.predict(test_data)
    mse_cb = 1.0 - accuracy_score(test_labels,predictions_cb)
    mse_arr_cb.append(mse_cb)
    #mahalanobis
    mse_mah = 0.0
    knn_mah = KNeighborsClassifier(n_neighbors=val, algorithm='brute', metric = 'mahalanobis', metric_params = {'V': np.cov(train_data)})
    knn_mah.fit(train_data, train_labels)
    predictions_mah = knn_mah.predict(test_data)
    mse_cb = 1.0 - accuracy_score(test_labels,predictions_mah)
    mse_arr_mah.append(mse_mah)


# best manhattan
best_idx_mh = []  
for i in range(len(mse_arr_mh)):
    if mse_arr_mh[i] == min(mse_arr_mh):
        best_idx_mh.append(i)


best_mse_mh = min(mse_arr_mh)
print('best mse manhattan: ', best_mse_mh)
best_k_mh = k[min(best_idx_mh)]
print('Best k = ', best_k_mh)

# best chebyshev
best_idx_cb = []  
for i in range(len(mse_arr_cb)):
    if mse_arr_cb[i] == min(mse_arr_cb):
        best_idx_cb.append(i)

best_mse_cb = min(mse_arr_cb)
print('best mse with chebyshev: ', best_mse_cb)
best_k_cb = k[min(best_idx_cb)]
print('Best k = ', best_k_cb)

# best mahalanobis
best_idx_mah = []  
for i in range(len(mse_arr_mah)):
    if mse_arr_mah[i] == min(mse_arr_mah):
        best_idx_mah.append(i)

best_mse_mah = min(mse_arr_mah)
print('best mse mahalanobis: ', best_mse_mah)
best_k_mah = k[min(best_idx_mah)]
print('Best k = ', best_k_mah)

################ best of log(p)################################
log_p = np.linspace(0,1,11)
p = []
for i in log_p:
    p.append(10**i)

mse_arr_p = []
for val in p:
    mse_p = 0.0
    knn_p = KNeighborsClassifier(n_neighbors= best_k_mh, p=val, metric = "minkowski")    
    knn_p.fit(train_data, train_labels)
    predictions_p = knn_p.predict(test_data)
    
    mse_p = 1.0 - accuracy_score(test_labels,predictions_p)
    mse_arr_p.append(mse_p)

best_idx_p = []  
for i in range(len(mse_arr_p)):
    if mse_arr_p[i] == min(mse_arr_p):
        best_idx_p.append(i)

best_mse_p = min(mse_arr_p)
print('Best error rate: ', best_mse_p)
print('mse arr with varying logp: ', mse_arr_p[:30])
print('Best index ', best_idx_p)
best_p = p[min(best_idx_p)]
print('Best log10(p) = ', math.log10(best_p))

##########weighted voting for metrics Euclidean, Manhattan, chebyshev ##########
k = [i for i in range(1,901,10)]
mse_arr_eu_w = []
mse_arr_mh_w = []
mse_arr_cb_w = []
train_mse_arr_eu_w = []
train_mse_arr_mh_w = []
train_mse_arr_cb_w = []
                                 
                                 
for val in k:
    #euclidean                             
    mse_eu_w = 0.0
    tr_mse_eu_w = 0.0

    knn_eu_w = KNeighborsClassifier(n_neighbors=val, weights = 'distance', p=2)    
    knn_eu_w.fit(train_data, train_labels)
    predictions_eu_w = knn_eu_w.predict(test_data)
    train_predict_eu_w = knn_eu_w.predict(train_data)

    mse_eu_w = 1.0 - accuracy_score(test_labels,predictions_eu_w)
    mse_arr_eu_w.append(mse_eu_w)

    #tr_mse = ((((float)(train_predict - train_actual)) ** 2).sum()) / (float)(len(train_predict))
    tr_mse_eu_w = 1.0 - accuracy_score(train_labels,train_predict_eu_w)
    train_mse_arr_eu_w.append(tr_mse_eu_w)
                                 
    #manhattan                             
    mse_mh_w = 0.0
    tr_mse_mh_w = 0.0

    knn_mh_w = KNeighborsClassifier(n_neighbors=val, weights = 'distance', p=1)    
    knn_mh_w.fit(train_data, train_labels)
    predictions_mh_w = knn_mh_w.predict(test_data)
    train_predict_mh_w = knn_mh_w.predict(train_data)

    mse_mh_w = 1.0 - accuracy_score(test_labels,predictions_mh_w)
    mse_arr_mh_w.append(mse_mh_w)

    #tr_mse = ((((float)(train_predict - train_actual)) ** 2).sum()) / (float)(len(train_predict))
    tr_mse_mh_w = 1.0 - accuracy_score(train_labels,train_predict_mh_w)
    train_mse_arr_mh_w.append(tr_mse_mh_w)
    
    #chebyshev                             
    mse_cb_w = 0.0
    tr_mse_cb_w = 0.0

    knn_cb_w = KNeighborsClassifier(n_neighbors=val, weights = 'distance', metric = 'chebyshev')    
    knn_cb_w.fit(train_data, train_labels)
    predictions_cb_w = knn_cb_w.predict(test_data)
    train_predict_cb_w = knn_cb_w.predict(train_data)

    mse_cb_w = 1.0 - accuracy_score(test_labels,predictions_cb_w)
    mse_arr_cb_w.append(mse_cb_w)

    #tr_mse = ((((float)(train_predict - train_actual)) ** 2).sum()) / (float)(len(train_predict))
    tr_mse_cb_w = 1.0 - accuracy_score(train_labels,train_predict_cb_w)
    train_mse_arr_cb_w.append(tr_mse_cb_w)

# best test errors

print('Best test error using euclidean metric and weighted voting: ', min(mse_arr_eu_w))
print('Best test error using manhattan metric and weighted voting: ', min(mse_arr_mh_w))
print('Best test error using chebyshev metric and weighted voting: ', min(mse_arr_cb_w))

#lowest training error
print('minimum training error: ', min(min(train_mse_arr_eu_w), min(train_mse_arr_mh_w), min(train_mse_arr_cb_w)))
