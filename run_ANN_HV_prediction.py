import pandas as pd
import numpy as np 

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #stops display of tensorflow logs

import tensorflow as tf;
from tensorflow.keras import layers;
from tensorflow.keras import Model;
from tensorflow.keras import optimizers;

from sklearn.utils import shuffle
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# Import local functions from 'f_ANN.py' and 'f_extract_input_data.py'
from f_ANN import f_ANN_model;
from f_extract_input_data import f_extract_input;


input_file_data = f_extract_input("Input_ANN.txt"); #dictionary that stores all data in Input txt file

x_feats = input_file_data['x']; # 'x' key value stored as x_feats (model features)
y_prop = input_file_data['y']; # 'y' key value stored as y_prop (phase_code)

print("\nReading database... ");
db_filename = input_file_data['database'][0];
db = pd.read_csv(db_filename, encoding='latin-1'); # database as pandas DataFrame
db = db[['alloy_name', 'phases'] + y_prop + x_feats]; #keeping only required data
print("Done.");

# dropping rows with NaN values in any column(Excluding alloy_name and phases columns)
db = db.dropna(axis='index', subset = y_prop + x_feats);
print('Dataset shape:',db.shape);

#creating directories for storing the results
print("\nCreating project directories ... ");
proj_dir = input_file_data['project_name'][0];
os.mkdir(proj_dir);

model_save_dir = input_file_data['project_name'][0]+'\\model_save\\';
os.mkdir(model_save_dir);

pred_save_dir = input_file_data['project_name'][0]+'\\prediction_save\\';
os.mkdir(pred_save_dir);

perf_save_dir = input_file_data['project_name'][0]+'\\performance_save\\';
os.mkdir(perf_save_dir);

print("Done.");

 
X = db[x_feats];
Y = db[y_prop];

X, Y = shuffle(X, Y, random_state=1);

phases = np.array(db['phases']).astype('str');
labels = db['alloy_name'] + ' (' + phases + ')';


# Dataframes to store loss and accuracy
loss_db = pd.DataFrame(data=np.arange(1,int(input_file_data['iterations'][0])+1,1) ,columns=['Iteration']);
mae_db = pd.DataFrame(data=np.arange(1,int(input_file_data['iterations'][0])+1,1) ,columns=['Iteration']);

K = 5; # value of K for K-fold cross-validation

L = len(Y);
K_L = int(L/K); #size (no. of rows) in each K validation set: here K_L = 40

X = np.array(X);
print('\nX-dataset shape:', X.shape);

Y = np.array(Y);
print('Y-dataset shape:', Y.shape);


# Running K-fold cross-validation
print("\nRunning %d-fold cross validation ... "%int(K));
for Kn in range(0,5):

    
    Kx_train = np.delete(X, np.s_[int(Kn*K_L):int(Kn*K_L+K_L)],0); #delete K_L no. of rows from trainx and store rest as K_trainx
    Kx_test = X[int(Kn*K_L):int(Kn*K_L+K_L)]; #storing rows that were removed from Kx_train into Kx_test

    Ky_train = np.delete(Y, np.s_[int(Kn*K_L):int(Kn*K_L+K_L)],0); #delete K_L no. of rows from trainx and store rest as K_trainx
    Ky_test = Y[int(Kn*K_L):int(Kn*K_L+K_L)]; #storing rows that were removed from Ky_train into Ky_test

    K_labels = labels[int(Kn*K_L):int(Kn*K_L+K_L)]; #labels for validation set instances
    x_axis = np.linspace(1,K_L,K_L); #creating x_axis that may be used for plotting validation predictions


    # Converting train and test set back to pandas dataframe to retain column titles
    Kx_train = pd.DataFrame(data=Kx_train,columns=x_feats);
    Kx_test = pd.DataFrame(data=Kx_test,columns=x_feats);


    # Create empty lists to store predictions for training and test set
    pred_NN_train = [];
    pred_NN_test = [];

    # Create empty lists to store loss for training and validation set
    train_K_loss = [];
    val_K_loss = [];

    # Create empty lists to MAE values for training and validation set
    train_K_mae = [];
    val_K_mae = [];

    threshold = float(input_file_data['check_error'][0]); #error threshold as obtained from INPUT file
    check_err = threshold+2; #initiating check_err as greater than threshold; later it will be updated to actual error obtained


    # Ensuring neural network is not stuck at local minima
    # creating while loop so program does not move forward unless error becomes lower than threshold
    while check_err > threshold:

        ANN_model = f_ANN_model(input_file_data); #assigning model compiled by f_ANN function to ANN_model variable
        
        # Fitting the ANN model and storing all fitting details into ANN_eval
        ANN_eval = ANN_model.fit(Kx_train,
                                 Ky_train,
                                 epochs = int(input_file_data['check_after_iterations'][0]),
                                 validation_data=(Kx_test, Ky_test),
                                 verbose=0);
        
        check_err = ANN_eval.history['mae'][-1]; #from model history, storing last iteration MAE value to check_error


    train_K_loss += ANN_eval.history['loss']; #appending training loss to train_K_loss list
    val_K_loss += ANN_eval.history['val_loss']; #appending validation loss to val_K_loss list

    train_K_mae += ANN_eval.history['mae']; #appending training MAE to train_K_mae list
    val_K_mae += ANN_eval.history['val_mae']; #appending validation MAE to val_K_mae list

    n_stops = int((int(input_file_data['iterations'][0])-int(input_file_data['check_after_iterations'][0]))/
                    int(input_file_data['save_after_iterations'][0])); #no. of stops where results will be saved

    
    # Creating perf DataFrame to store the performance of current validation set after each iteration
    perf = pd.DataFrame(columns=['K-validation set','iteration','MAE_train','MAE_test','RMSE_test','R2_test']);


    # Continuing training and saving results after each stop
    for stop in range(1, n_stops+1):

        ANN_eval = ANN_model.fit(Kx_train,
                                 Ky_train,
                                 epochs = int(input_file_data['save_after_iterations'][0]),
                                 validation_data=(Kx_test, Ky_test),
                                 verbose=0);
        
        train_K_loss += ANN_eval.history['loss'];
        val_K_loss += ANN_eval.history['val_loss'];

        train_K_mae += ANN_eval.history['mae'];
        val_K_mae += ANN_eval.history['val_mae'];

        it_n = int(input_file_data['check_after_iterations'][0])+stop*int(input_file_data['save_after_iterations'][0]); #current iteration
        print(str(input_file_data['project_name'][0])+': K'+str(Kn+1)+'-'+str(it_n)+' iterations');
        
        ANN_model.save(model_save_dir+'stop'+str(it_n)+'_K'+str(Kn+1)+'_model.h5'); #saving trained model

        model = ANN_model;
        #model = tf.keras.models.load_model(model_save_dir+'stop'+str(it_n)+'_K'+str(Kn+1)+'_model.h5');
        
        Kx_train = np.array(Kx_train).reshape(Ky_train.shape[0],1,len(x_feats)); #reshaping Kx_train before feeding it to model for prediction
        Kx_test = np.array(Kx_test).reshape(Ky_test.shape[0],1,len(x_feats)); #reshaping Kx_test before feeding it to model for prediction


        pred_train = np.zeros(Ky_train.shape); #creating all 'zero' array to store training predictions
        
        # Updating each element in pred_train with predicted hardness
        for i in range(0,pred_train.shape[0]):
            pred_train[i] = model.predict(Kx_train[i]); #predicted hardness
        
        
        pred_test = np.zeros(Ky_test.shape); #creating all 'zero' array to store validation predictions
        
        # Updating each element in pred_test with predicted hardness
        for i in range(0,pred_test.shape[0]):
            pred_test[i] = model.predict(Kx_test[i]); #predicted hardness

        pred = pd.DataFrame(data=K_labels,columns=['alloy_name']); #creating DataFrame to store predictions
        
        pred['VHN_actual'] = Ky_test; #adding Ky_test as actual hardness column
        pred['VHN_predicted'] = pred_test; #adding predicted validation results as Predicted Hardness
        pred['Error'] = pred_test - Ky_test; #adding Error column
        pred['Abs Error'] = np.absolute(pred_test - Ky_test); #adding absolute error
        pred['% Error'] = np.absolute((pred_test - Ky_test))*100/Ky_test; #adding % error column
        
        pred.to_csv(pred_save_dir+'stop'+str(it_n)+'_K'+str(Kn+1)+'_pred.csv'); # saving prediction file (.csv) for current iteration
        
        # Calculating statistics using scikit-learn predefined functions
        MAE_train = mean_absolute_error(Ky_train, pred_train); 
        MAE_test = mean_absolute_error(Ky_test, pred_test);
        RMSE_test = mean_squared_error(Ky_test, pred_test)**0.5;
        r2_test = r2_score(Ky_test, pred_test);
        
        # Updating performance database of current K-validation set with current iteration results
        perf.loc[stop-1] = ['K'+str(Kn+1),it_n, MAE_train, MAE_test, RMSE_test, r2_test];
        
        Kx_train = np.array(Kx_train).reshape(Ky_train.shape[0],len(x_feats)); #again reshape back the Kx_train as it will be fed back to ANN
        Kx_test = np.array(Kx_test).reshape(Ky_test.shape[0],len(x_feats)); #again reshape back the Kx_test as it will be fed back to ANN in next loop
        
        
    perf.to_csv(perf_save_dir+'stop'+'_K'+str(Kn+1)+'_perf.csv'); #save performance file for current K-validation set
    
    loss_db['K'+str(Kn+1)+'_train_loss'] = train_K_loss; #add train_K_loss per iteration to main loss_db database
    loss_db['K'+str(Kn+1)+'_val_loss'] = val_K_loss; #add val_K_loss per iteration to main loss_db database
    
    mae_db['K'+str(Kn+1)+'_train_mae'] = train_K_mae; #add train_K_mae per iteration to main mae_db database
    mae_db['K'+str(Kn+1)+'_val_mae'] = val_K_mae; #add val_K_mae per iteration to main mae_db database


loss_db.to_csv(proj_dir+'\\loss.csv'); #save loss_db as .csv file
mae_db.to_csv(proj_dir+'\\mae.csv') #save mae_db as .csv file


print('Finished.');