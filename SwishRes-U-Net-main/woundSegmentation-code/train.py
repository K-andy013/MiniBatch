from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope

from models.DUNet import build_model

from utils.learning.metrics import dice_coef, precision, recall
from utils.learning.losses import dice_coef_loss
from utils.io.data import DataGen, save_results, save_history, load_data


# manually set cuda 10.0 path
#os.system('export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}')
#os.system('export PATH=/usr/local/cuda-10.0/bin:/usr/local/cuda-10.0/NsightCompute-1.0${PATH:+:${PATH}}')

# Varibales and data generator
input_dim_x=224
input_dim_y=224
n_filters = 32

# dataset = 'FUSeg'
# data_gen = DataGen('D:/Hafsa/woundSegmentation/data/FUSeg-aug/' + dataset + '/', split_ratio=0.2, x=input_dim_x, y=input_dim_y)
dataset = 'FUSeg'
data_gen = DataGen('C:/Users/kwame/OneDrive/Desktop/UNI STUFF/COMPUTER SCIENCE/L400/Project Work/Test/data/' + dataset + '/', split_ratio=0.2, x=input_dim_x, y=input_dim_y)
save_model_name = "model-DUNet-FUSeg-aug-100ep"
save_model_path = 'SOTA-models/FUSeg/'


########## DUNet Model ###############
model, model_name = build_model(shape=(input_dim_x, input_dim_y, 3))


#### training
batch_size = 8
epochs = 1 
learning_rate = 1e-4
loss = 'binary_crossentropy'

es = EarlyStopping(monitor='val_dice_coef', patience=5, mode='max', restore_best_weights=True)


model.summary()
model.compile(optimizer=Adam(lr=learning_rate), loss=loss, metrics=[dice_coef, precision, recall])

model_history = model.fit_generator(data_gen.generate_data(batch_size=batch_size, train=True),
                                       steps_per_epoch=int(data_gen.get_num_data_points(train=True) / batch_size),
                                       callbacks=[es],
                                       validation_data=data_gen.generate_data(batch_size=batch_size, val=True),
                                       validation_steps=int(data_gen.get_num_data_points(val=True) / batch_size),
                                       epochs=epochs)

### save the model weight file and its training history
save_history(model, model_name, model_history, dataset, n_filters, epochs, learning_rate, loss, color_space='RGB',
             path=save_model_path, temp_name=save_model_name)