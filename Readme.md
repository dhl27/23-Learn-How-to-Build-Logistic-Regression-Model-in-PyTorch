# Pytorch

- PyTorch is an open source machine learning library for Python and is completely based on Torch. 
- It is primarily used for applications such as natural language processing
- PyTorch redesigns and implements Torch in Python while sharing the same core C libraries for the backend code.
- PyTorch developers tuned this back-end code to run Python efficiently. They also kept the GPU based hardware acceleration as well as the extensibility features that made Lua-based Torch.

## Logistic Regression

Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable better than Linear Regression.

    Sigmoid function: It’s a mathematical function having a characteristic S shape curve. It classify a data in o class if its probability is less than 0.5 or else it put the data in class 1 and is given by: 

Data is fed to the model using Pytorch nn module where two arguments has been passed:
First argument consists of the Input Features 
Second argument consists of the number of output we want which is 1 in our case either 0 or 1

Finally, we pass the model parameters to the Sigmoid activation function so as to classify them with a threshold of 0.5 in binary class


## Training Process Logistic Regression

A typical training procedure for a Logistic Regression is as follows:
- Define the Logistic Regression that has some learnable parameters (or weights)
- Iterate over a dataset of inputs
- Process input through the network
- Compute the loss (how far is the output from being correct)
- Propagate gradients back into the network’s parameters
- Update the weights of the network, typically using a simple update rule.


## Code Description


    File Name : Engine.py
    File Description : Main class for starting the model training lifecycle


    File Name : LogisticRegression.py
    File Description : Class of Logistic Regression structure
    
    File Name : TrainModel.py
    File Description : Code to train and evaluate the pytorch model


## Steps to Run

There are two ways to execute the end to end flow.

- Modular Code
- IPython

### Modular code

- Create virtualenv
- Install requirements `pip install -r requirements.txt`
- Run Code `python Engine.py`
- Check output for all the visualization

### IPython Google Colab

Follow the instructions in the notebook `LogisticRegression.ipynb`

