#kmeanspp
import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
#loading data
data = pd.read_csv("500_Person_Gender_Height_Weight_Index.csv")

data["Index"].unique()

#visualising the data
plt.scatter(data["Height"], data["Weight"])

plt.xlabel("Height / cm")
plt.ylabel("Weight / Kg")
plt.title("Data Visualisation")

X = data[["Height", "Weight"]]

def get_random_centroids(data, k = 3,):
    
    #return random samples from the dataset
   
    cent = (X.sample(n = k))
    return cent

#initialising centroids
centroids = get_random_centroids(X, k = 5)


plt.figure()

#Lets see where our initial centroids lie
plt.scatter(X["Height"], X["Weight"])
plt.scatter(centroids["Height"],centroids["Weight"] , c = 'red')
plt.legend(["Data", "Centroid"])

def k_means_fit(X,centroids, n = 5):
    #get a copy of the original data
    X_data = X
    
    
    diff = 1
    j=0

    while(diff!=0):

        #creating a copy of the original dataframe
        i=1

        #iterate over each centroid point 
        for index1,row_c in centroids.iterrows():
            ED=[]

            #iterate over each data point
            for index2,row_d in X_data.iterrows():

                #calculate distance between current point and centroid
                d1=(row_c["Height"]-row_d["Height"])**2
                d2=(row_c["Weight"]-row_d["Weight"])**2
                d=np.sqrt(d1+d2)

                #append distance in a list 'ED'
                ED.append(d)

            #append distace for a centroid in original data frame
            X[i]=ED
            i=i+1

        C=[]
        for index,row in X.iterrows():

            #get distance from centroid of current data point
            min_dist=row[1]
            pos=1

            #loop to locate the closest centroid to current point
            for i in range(n):

                #if current distance is greater than that of other centroids
                if row[i+1] < min_dist:

                    #the smaller distanc becomes the minimum distance 
                    min_dist = row[i+1]
                    pos=i+1
            C.append(pos)

        #assigning the closest cluster to each data point
        X["Cluster"]=C

        #grouping each cluster by their mean value to create new centroids
        centroids_new = X.groupby(["Cluster"]).mean()[["Weight","Height"]]
        if j == 0:
            diff=1
            j=j+1

        else:
            #check if there is a difference between old and new centroids
            diff = (centroids_new['Weight'] - centroids['Weight']).sum() + (centroids_new['Height'] - centroids['Height']).sum()
            print(diff.sum())

        centroids = X.groupby(["Cluster"]).mean()[["Weight","Height"]]
        
    return X, centroids

centroids = get_random_centroids(X, k = 4)
clustered, cent = k_means_fit(X,centroids, n= 4)

print(cent)

#setting color values for our 
color=['brown','blue','green','cyan']


def get_kmeans_pp_centroids(X1,k = 5):
    centroids = X1.sample()
    print(centroids)
    i = 1
    dist = []
    while i != k:
        max_dist = [0,0]
        #go through the centroids
        for index, row in centroids.iterrows():
            #calculate distance of every centroid with every other data point 
            d = np.sqrt((X1["Height"] - row["Height"])**2 +(X1["Weight"] - row["Weight"])**2)
            #check which centroid has a max distance with another point
            if max(d) > max(max_dist):
                max_dist = d

        X1 = pd.concat([X1, max_dist], axis = 1)
        idx = X1.iloc[:,i+1].idxmax()
        max_coor = pd.DataFrame(X1.iloc[idx][["Height", "Weight"]]).T
        centroids = pd.concat([centroids,max_coor])
        X1 = X1.drop(idx)
        i+=1
    return centroids

#loading data

centroids = get_kmeans_pp_centroids(X, k = 4)
clustered, cent = k_means_fit(X,centroids, n= 4)


#setting color values for our 
color=['brown','blue','green','cyan']

#plot data
for k in range(len(color)):
    cluster=clustered[clustered["Cluster"]==k+1]
    plt.scatter(cluster["Height"],cluster["Weight"],c=color[k])
    
#plot centroids    
plt.scatter(cent["Height"],cent["Weight"],c='red')
plt.xlabel('Height/ cm')
plt.ylabel('Weight/ kg')

print(cent)
data




