import csv
import sys
sys.path.insert(0, '../clustering')
sys.path.insert(0, '../classification')



#my classifier - output  testSet_categories.csv
execfile("../classification/MINE.py")
myClassifier = mine()

#k-means - output clustering_KMeans.csv
execfile("../clustering/K_means.py")
kmeans=kMeans()


#output EvaluationMetric_10fold.csv
result = [['Statistic Measure','Naive Bayes','Random Forest','SVM','KNN']]

execfile("../classification/NB.py")
nb = naiveBayes()

execfile("../classification/RF.py")
rf = rf()

execfile("../classification/SVM.py")
svm = support_vector_machine()

execfile("../classification/KNN.py")
knn = kNearest_neighbors()

accuracy = ['Accuracy']
accuracy.append(nb[0])
accuracy.append(rf[0])
accuracy.append(svm[0])
accuracy.append(knn[0])
result.append(accuracy)

precision = ['Precision']
precision.append(nb[1])
precision.append(rf[1])
precision.append(svm[1])
precision.append(knn[1])
result.append(precision)

recall = ['Recall']
recall.append(nb[2])
recall.append(rf[2])
recall.append(svm[2])
recall.append(knn[2])
result.append(recall)

f1_score = ['F-Measure']
f1_score.append(nb[3])
f1_score.append(rf[3])
f1_score.append(svm[3])
f1_score.append(knn[3])
result.append(f1_score)

auc = ['AUC']
auc.append(nb[4])
auc.append(rf[4])
auc.append(svm[4])
auc.append(knn[4])
result.append(auc)

# write the results in csv file
with open('../outputFiles/EvaluationMetric_10fold.csv', 'wb') as fp:
    a = csv.writer(fp, delimiter='\t')
    a.writerows(result)








