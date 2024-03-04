import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dense, Dropout, Masking
from tensorflow.keras import Input
from tensorflow.keras import Model

#%% Load Data
df = pd.read_csv('Field_App_Full_S50_2.csv')
df = df.drop('CDWater_BBLPerDAY',axis=1)
# Extract
feature_no = len(list(df))
data_cols = list(list(df)[i] for i in list(range(1,feature_no)))
data = df[data_cols].astype(float).to_numpy()

# qg index
qg_idx = data_cols.index('CDGas_MCFPerDAY')
shut_idx = data_cols.index('ShutIns')
peak_idx = data_cols.index('Max_time')
peakt_idx = data_cols.index('Max_Value')

#%% Data Normalozation
norm_data = data
scaler = MinMaxScaler()
scaler.fit(data)
norm_data = scaler.transform(norm_data)

max_qg = np.max(data[:,qg_idx])
min_qg = np.min(data[:,qg_idx])
print(max_qg)
print(min_qg)
print(norm_data.shape)

t_steps = 36 #reshaped time steps
n_wells = int(len(df)/36)
data_total = norm_data.reshape(n_wells,t_steps,data.shape[1])
data_total_list = data_total.tolist()
random.Random(99).shuffle(data_total_list) # Shuffle the data
data_total = np.array(data_total_list)

#%% split training and testing data
tr_por = 2499
data_train = data_total[:tr_por,:,:]
data_test = data_total[tr_por:,:,:]

#%% Process the data, delay and segmentation
feature_list = list(range(len(list(df))-1))
exampt_list = [i for i in feature_list if i not in [shut_idx]]
exampt_list2 = [i for i in feature_list if i not in [qg_idx]]
exampt_list3 = [i for i in feature_list if i not in [peak_idx,peakt_idx,qg_idx]]

def delay_embedding_MRA(train,m,d,exampt_list,shut_idx,max_len):
    en_trainX = []
    de_trainX = []
    trainY = []
    i=0
    unchanged_val = train[i:i+m*d:d][:,exampt_list]
    shut_val = train[1:i+m*d-d+2, [shut_idx]]
    trainx_val = np.append(unchanged_val,shut_val,axis=1)
    trainx_val_padded = nan_padding(trainx_val,max_len)

    en_trainX.append(trainx_val_padded)
    
    de_trainX.append(train[i+m*d-d+1, exampt_list3])
    
    trainY.append(train[i+m*d-d+1, [qg_idx]]) # colume want to predict

    return np.array(en_trainX), np.array(de_trainX), np.array(trainY)

def nan_padding(sub_list,max_length):
    padded_array = np.empty((max_length,sub_list.shape[1]))
    padded_array[:] = np.nan
    current_t = sub_list.shape[0]
    padded_array[:current_t,:] = sub_list
        
    return padded_array

def zero_padding(sub_list,max_length):
    padded_array = np.zeros((max_length,sub_list.shape[1]))
    current_t = sub_list.shape[0]
    padded_array[:current_t,:] = sub_list
        
    return padded_array

# Training data
d = 1
m_all = list(range(t_steps))
max_len = t_steps
en_trainXX = []
de_trainXX = []
trainYY = []
x = data_train

# 2499*36 = 89964
for i_delay in range(x.shape[0]):
    for m in m_all:
        train_i = x[i_delay,:,:] 
        en_trainX, de_trainX, trainY = delay_embedding_MRA(train_i,m,d,exampt_list,shut_idx,max_len)

        en_trainXX.append(en_trainX.tolist()[0])
        de_trainXX.append(de_trainX.tolist()[0])
        trainYY.append(trainY.tolist()[0])

# re-format
encoder_inputs = np.array(en_trainXX)  
decoder_inputs = np.array(de_trainXX)
decoder_inputs=decoder_inputs.reshape(decoder_inputs.shape[0],1,decoder_inputs.shape[1])
labels = np.array(trainYY)  
print('Encoder Input shape == {}'.format(encoder_inputs.shape))
print('Decoder Input shape == {}'.format(decoder_inputs.shape))
print('Label shape == {}'.format(labels.shape))

# Testing data
d = 1
m_all = list(range(t_steps))
max_len = t_steps
en_trainXX_te = []
de_trainXX_te = []
trainYY_te = []
x = data_test

# 200*36 = 7200
for i_delay in range(x.shape[0]):
    for m in m_all:
        train_i = x[i_delay,:,:] 
        en_trainX, de_trainX, trainY = delay_embedding_MRA(train_i,m,d,exampt_list,shut_idx,max_len)

        en_trainXX_te.append(en_trainX.tolist()[0])
        de_trainXX_te.append(de_trainX.tolist()[0])
        trainYY_te.append(trainY.tolist()[0])

# re-format
encoder_inputs_te = np.array(en_trainXX_te)   
decoder_inputs_te = np.array(de_trainXX_te)  
decoder_inputs_te=decoder_inputs_te.reshape(decoder_inputs_te.shape[0],1,decoder_inputs_te.shape[1])
labels_te = np.array(trainYY_te)  
print('Encoder Input shape == {}'.format(encoder_inputs_te.shape))
print('Decoder Input shape == {}'.format(decoder_inputs_te.shape))
print('Label shape == {}'.format(labels_te.shape))

# Change the padding value
mask_special_val = -1e9
nn_inputs = np.nan_to_num(encoder_inputs, nan=mask_special_val)
nn_inputs_te = np.nan_to_num(encoder_inputs_te, nan=mask_special_val)

#%% Construct the Neural Netowrk
# Encoder
encoder_input = Input(shape=(encoder_inputs.shape[1], encoder_inputs.shape[2]))
mask_input = Masking(mask_value=mask_special_val)(encoder_input)
encoder_gru1 = GRU(64, return_sequences=True)(mask_input)
encoder_gru1 = Dropout(0.1)(encoder_gru1)
encoder_gru2 = GRU(64, return_sequences=False)(encoder_gru1)
encoder_gru2 = Dropout(0.1)(encoder_gru2)

# Decoder
decoder_input = Input(shape=(decoder_inputs.shape[1], decoder_inputs.shape[2]))
decoder_gru1 = GRU(64, return_sequences=True)(decoder_input, initial_state=encoder_gru2)
decoder_gru1 = Dropout(0.1)(decoder_gru1)
decoder_gru2 = GRU(64, return_sequences=True)(decoder_gru1, initial_state=encoder_gru2)
decoder_gru2 = Dropout(0.1)(decoder_gru2)
dense = Dense(64, activation='tanh')(decoder_gru2)
decoder_output = Dense(1, activation='tanh')(dense)

# Define the model
model = Model([encoder_input, decoder_input], decoder_output)

# Compile the model
model.compile(loss='mse', optimizer='adam')
model.summary()

#%% Train the model
start_time = time.time() # record the time training the model
history =  model.fit([nn_inputs,decoder_inputs],
                     labels, epochs=2, batch_size= 200, 
                     verbose=1,validation_split = 0.1)
print("--- %s seconds ---" % (time.time() - start_time))

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#%% Test on Prediction
def MRA_Process(predict_num,prod_period,curr_model):
    filename="MRA_0727_cloud_weight"+str(curr_model+1)
    filepath = filename+".hdf5"
    model.load_weights(filepath)
    
    en_testX = data_test[predict_num,:,:]
    en_testX = en_testX.reshape(1,en_testX.shape[0],en_testX.shape[1])

    en_testX_test = np.empty_like(en_testX)
    en_testX_test[:]=np.nan # like zero padding
    en_testX_test[0,0:prod_period,:] = en_testX[0,0:prod_period,:] # first few value = actual data

    en_testX_test = np.nan_to_num(en_testX_test, nan=mask_special_val)

    for ii in range(data_test.shape[1]-prod_period):
        de_testX_test = en_testX[0,ii+prod_period, exampt_list3]
        de_testX_test = de_testX_test.reshape(1,1,de_testX_test.shape[0])
        org_test = model.predict([en_testX_test,de_testX_test],verbose=0) 
        en_testX_test[:,ii+prod_period,exampt_list2] = en_testX[:,ii+prod_period,exampt_list2]
        en_testX_test[:,ii+prod_period,qg_idx] = org_test

    qg_pred = en_testX_test[:,:,qg_idx].reshape(-1)
    return qg_pred

def local_RMSE(testY_org,testPredict):
    # Local RMSE
    scaler = MinMaxScaler()
    scaler.fit(testY_org.reshape(-1,1))
    norm_y_label = scaler.transform(testY_org.reshape(-1,1))
    norm_y_pred = scaler.transform(testPredict.reshape(-1,1))
    testScore = mean_squared_error(norm_y_label, norm_y_pred,squared=False)
    return testScore

plot_num = 66 # selecet the case you want to plot for visualization
row = 1
col = 4
model_num = 5
fig, ax = plt.subplots(row,col,figsize=(17,2.5))
if isinstance(ax, np.ndarray):
    ax = ax.flatten()

for i, subplot_ax in enumerate(ax):
    if i == 0:
        plot_t = i
    else:
        plot_t = i*2-1
    testScore_all = []
    subplot_ax.plot(list(range(36)), data_test[plot_num,:,qg_idx]*(max_qg - min_qg) + min_qg,'k-',label='Label',alpha=0.1)
    subplot_ax.plot(list(range(36)), encoder_inputs_te[plot_num*36+plot_t,:,qg_idx]*(max_qg - min_qg) + min_qg,'ro-',label='Production Data', alpha=0.3)
    
    for j in range(model_num):      
        qg_pred = MRA_Process(plot_num,plot_t,j)
        label_predict = 'Prediction'+ str(j+1)
        
        plot_qg = qg_pred[plot_t:]*(max_qg - min_qg) + min_qg
        subplot_ax.plot(list(range(plot_t,36)), plot_qg,label=label_predict)
        qg_pred[:plot_t] = data_test[plot_num,:plot_t,qg_idx]
        testScore = local_RMSE(qg_pred,data_test[plot_num,:,qg_idx])
        #testScore = local_R2score(qg_pred,data_test[plot_num,:,qg_idx])
        testScore_all.append(testScore)
        
    if i == 0:
        legend = subplot_ax.legend(loc='upper right',ncol=1,labelspacing=0.1, handlelength=1.0)
        for text in legend.get_texts():
            text.set_fontsize(10)
        ymin = 0
        ymax = np.max(qg_pred*(max_qg - min_qg) + min_qg)*1.6
        subplot_ax.set_ylim([ymin, ymax])
    else:
        ymax = np.max(data_test[plot_num,:,qg_idx]*(max_qg - min_qg) + min_qg)*1.2
        subplot_ax.set_ylim([ymin, ymax])
     
    testScore_ave = sum(testScore_all)/len(testScore_all)
    print(testScore_ave)
    subplot_ax.set_title('RMSE(ave): %.6f\n' % (testScore_ave)+ ' Producing Time(month): %d' %(plot_t))
    #subplot_ax.set_title('$R^2$ (ave): %.6f\n' % (testScore_ave)+ ' Producing Time(month): %d' %(plot_t))
    subplot_ax.set_xlabel('Time (Month)')
    subplot_ax.set_ylabel('Gas Rate (MCF/day)')

plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.95, wspace=0.32, hspace=0.6)

#%% Evaluation
def MRA_Process_group(prod_period,curr_model):
    filename="MRA_0727_cloud_weight"+str(curr_model+1)
    filepath = filename+".hdf5"
    model.load_weights(filepath)
    
    en_testX = data_test[:,:,:]
    en_testX_test = np.empty_like(en_testX)
    en_testX_test[:]=np.nan # like zero padding
    en_testX_test[:,0:prod_period,:] = en_testX[:,0:prod_period,:] # first few value = actual data
    en_testX_test = np.nan_to_num(en_testX_test, nan=mask_special_val)

    for ii in range(data_test.shape[1]-prod_period):
        de_testX_test = en_testX[:,ii+prod_period, exampt_list3]
        de_testX_test = de_testX_test.reshape(de_testX_test.shape[0],1,de_testX_test.shape[1])
        org_test = model.predict([en_testX_test,de_testX_test],verbose=0) 
        en_testX_test[:,ii+prod_period,exampt_list2] = en_testX[:,ii+prod_period,exampt_list2]
        en_testX_test[:,ii+prod_period,qg_idx] = org_test.reshape(-1)

    qg_pred_all = en_testX_test[:,:,qg_idx]
    
    return qg_pred_all

# We evaluate the model on the RMSE
def find_RMSE_MRA(model_num,prod_period):
    testScore_all = []
    for j in range(model_num):
        qg_pred_all = MRA_Process_group(prod_period,j)
        
        testScore_store = []
        for i in range(data_test.shape[0]):
            qg_pred = qg_pred_all[i]
            qg_pred[:prod_period] = data_test[i,:prod_period,qg_idx] # ignore the error form the kown period
            testScore = local_RMSE(qg_pred,data_test[i,:,qg_idx])
            testScore_store.append(testScore)
            
        testScore_arr = np.array(testScore_store)
        testScore_all.append(testScore_arr)
        
    testScore_ave = sum(testScore_all)/len(testScore_all)
    return testScore_ave

def calculate_average_without_outliers(data):
    # Calculate the first quartile (Q1) and third quartile (Q3)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)

    # Calculate the interquartile range (IQR)
    IQR = Q3 - Q1

    # Define the lower and upper bounds to identify outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out the outliers from the data
    data_without_outliers = [x for x in data if lower_bound <= x <= upper_bound]

    # Calculate the average of the data without outliers
    average_without_outliers = np.mean(data_without_outliers)

    return average_without_outliers

qua_idx_store = []
quantile_store = []
maxx = max_qg
minn = min_qg

# Evaluate the model with differnt length of kown production data
for i in [0,1,3,5]:
    prod_period = i
    model_num = 5
    RMSE_qg = find_RMSE_MRA(model_num,prod_period)
    RMSE_qg_ave = np.mean(RMSE_qg)
    #RMSE_qg_ave = calculate_average_without_outliers(RMSE_qg)
    print('RMSE:',RMSE_qg_ave)
    
    quantiles = np.percentile(RMSE_qg,(10,50,90))
    qua_idx = [(np.abs(RMSE_qg - ii)).argmin() for ii in quantiles]
    quantile_store.append(quantiles)
    qua_idx_store.append(qua_idx)
    
    print(str(i),':',quantiles)
    print(str(i),':',qua_idx)