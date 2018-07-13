from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#import matplotlib.pyplot as plt
import numpy as np

digits = load_digits()

# Print to show there are 1797 images (8 by 8 images for a dimensionality of 64)
print("Image Data Shape" , digits.data.shape)

# Print to show there are 1797 labels (integers from 0–9)
print("Label Data Shape", digits.target.shape)
#
# plt.figure(figsize=(20,4))
# for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
#  plt.subplot(1, 5, index + 1)
#  plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
#  plt.title('Training: %i\n' % label, fontsize = 20)

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)

# all parameters not specified are set to their defaults
logisticRegr = LogisticRegression()

logisticRegr.fit(x_train, y_train)

predictions = logisticRegr.predict(x_test)
# Use score method to get accuracy of model
score = logisticRegr.score(x_test, y_test)
print(score)

