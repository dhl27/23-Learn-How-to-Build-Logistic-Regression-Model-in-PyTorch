
# ### Model Building: Creating Logistic Regression model in Pytorch
#
#
# #### Logistic Regression is a Linear model so we will use Pytorch's [nn.linear] module which is used for performing linear operations  for making a linear model and then we will pass the data to sigmoid function which separates a binary data in two parts using probability.

# ![Logistic_regress_Image.jpeg](attachment:Logistic_regress_Image.jpeg)

from .LogisticRegression import LogisticRegression
import torch
import torch.nn as nn
import torchvision.transforms as transforms

class TrainModel:

    def __init__(self, n_features, X_train, X_test, y_train, y_test):

        lr = LogisticRegression(n_features)

        # #### Model Compiling: Let us define the number epochs and the learning rate we want our model for training. As the data is binary so we will use Binary Cross Entropy as the loss function that will be used for the optimization of the model using ADAM optimizer.
        # ![Grad_descent_pic.png](attachment:Grad_descent_pic.png)

        num_epochs = 500
        # Traning the model for large number of epochs to see better results
        learning_rate = 0.01
        criterion = nn.BCELoss()
        # We are working on lgistic regression so using Binary Cross Entropy
        optimizer = torch.optim.SGD(lr.parameters(), lr=learning_rate)
        # Using ADAM optimizer to find local minima

        self.train(criterion, lr, num_epochs, optimizer, X_train, y_train)

        self.evaluate(lr, X_test, y_test)

    def evaluate(self, lr, X_test, y_test):
        # #### Model Accuracy: Let us finally see the model accuracy
        with torch.no_grad():
            y_predicted = lr(X_test)
            y_predicted_cls = y_predicted.round()
            acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
            print('------o--------')
            print(f'accuracy is: {acc.item()*100:.4f}%')
            print('-------o--------')
        # #### Looking other Metrics: We can also see the precision, recall, and F1-score using classification report
        from sklearn.metrics import classification_report
        print('--Classification Matrix--')
        print(classification_report(y_test, y_predicted_cls))
        # ####  Visualizing Confusion Matrix
        # ![Confusion-matrix-Exemplified-CM-with-the-formulas-of-precision-PR-recall-RE.png](attachment:Confusion-matrix-Exemplified-CM-with-the-formulas-of-precision-PR-recall-RE.png)
        from sklearn.metrics import confusion_matrix
        print('--confusion Matrix--')
        confusion_matrix = confusion_matrix(y_test, y_predicted_cls)
        print(confusion_matrix)

    def train(self, criterion, lr, num_epochs, optimizer, X_train, y_train):
        # #### Visualizing the training process
        for epoch in range(num_epochs):
            y_pred = lr(X_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if (epoch + 1) % 20 == 0:
                # printing loss values on every 10 epochs to keep track
                print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')
