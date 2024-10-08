#importing necessary libraries
import pandas as pd
from MLPipeline.TrainModel import TrainModel
from MLPipeline.Preprocessing import Preprocessing

# Reading the data
df = pd.read_csv("Input/data.csv")

#dropping unwanted columns
cols_to_drop=["customer_id", "phone_no", "year"] #put the names of columns to drop
df=Preprocessing(df).drop(cols_to_drop) #comment this line if there are no columns to drop

#applying preprocessing to data
data=Preprocessing(df).apply_preprocessing()

#resampling data for class balance
target_col='churn' #Put target column name here
maj_cls=0 #put majority class here
min_cls=1 #put minority class here
data_new=Preprocessing(data).resample(target_col,maj_cls,min_cls)


# splitting data into train and test
target_col='churn' #Put target column name here
X_train, X_test, y_train, y_test = Preprocessing(data).split_data(data_new,target_col)

#converting data into tensor form
n_features,X_train, X_test, y_train,y_test = Preprocessing(data).convert_to_tensor(X_train,y_train,X_test,y_test)


# # Training the network
TrainModel(n_features, X_train, X_test, y_train, y_test)

