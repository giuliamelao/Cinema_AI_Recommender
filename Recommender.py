import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


#dataset too big, so I'm sampling 100000 rows randomly
ratings = pd.read_csv('Netflix_User_Ratings.csv').sample(n=100000, random_state=42)

#making sure I'm working with numeric data
ratings['CustId'] = ratings['CustId'].astype('int')
ratings['MovieId'] = ratings['MovieId'].astype('int')

#dividing into test and train, this approach of dividing in 2 is best for colaborative filtering
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

#redimensioning my training data to fit a model used for colaborative filtering
ratings_train_matrix = train_data.pivot(index='CustId', columns='MovieId', values='Rating').fillna(0)

#this puts my matrix in a different dimension
sparse_ratings_matrix = csr_matrix(ratings_train_matrix)

#this creates a matrix for each field
U, sigma, Vt = svds(sparse_ratings_matrix, k=min(sparse_ratings_matrix.shape)-1)

#redimensioning again, diagonally
sigma_diag_matrix = np.diag(sigma)

#Here it reconstructs the predicted ratings by multiplying the three matrices obtained from SVD
predicted_ratings = np.dot(np.dot(U, sigma_diag_matrix), Vt)

 #This is the essence of collaborative filtering—predicting
 #user ratings based on the latent features of users and items

