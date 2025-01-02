#####################################################
#   Logistic Regression Example Code                #
#   Author: Grace Stapkowski                        #
#   Date Created: 7/30/24                           #
#####################################################

# imports 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# initialize example training data
# Note: features is resized because it must be 2D 
features = np.arange(10).reshape(-1, 1)
flagged = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# create model
# Note: liblinear solver does not work without regularization
model = LogisticRegression(solver='liblinear', random_state=None)

#fit model
model.fit(features, flagged)

# plot data and logistic function 
plt.scatter(features, flagged)
plt.plot(features, model.predict_proba(features)[:,1], color='red', 
            marker="+")
plt.show()
