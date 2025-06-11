import cv2
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope

from models.DUNet import build_model

from utils.learning.metrics import dice_coef, precision, recall
from utils.BilinearUpSampling import BilinearUpSampling2D
from utils.io.dataVal import load_data, save_results, save_rgb_results, save_history, load_test_images, DataGenValidation


# settings
input_dim_x = 224
input_dim_y = 224
color_space = 'rgb'
# data_path = 'D:/Hafsa/woundSegmentation/data/azh_wound_care_center_dataset_patches/'
data_path = 'C:/Users/kwame/OneDrive/Desktop/UNI STUFF/COMPUTER SCIENCE/L400/Project Work/Mini Batch/data/FuSeg/validation/'

save_model_path = "SOTA-models/FUSeg/"
save_model_name = "model-DUNet-FUSeg-aug-100ep"
weight_file_name = save_model_name+ '.hdf5'


save_result_path = 'C:/Users/kwame/OneDrive/Desktop/UNI STUFF/COMPUTER SCIENCE/L400/Project Work/Mini Batch/results-SOTA/FUSeg/crossCorpora/' + save_model_name + '/model-pred-masks/'
# save_result_path = "D:/Hafsa/woundSegmentation/results/FUSeg/predictions/" + save_model_name  + "/model-pred-masks/"

data_gen = DataGenValidation(data_path, split_ratio=0.0, x=input_dim_x, y=input_dim_y, color_space=color_space)
x_test, test_label_filenames_list = load_test_images(data_path)



## get  DUNet Model
model, model_name = build_model(shape=(input_dim_x, input_dim_y, 3))
model = load_model(save_model_path + weight_file_name
               , custom_objects={'recall':recall,
                                 'precision':precision,
                                 'dice_coef': dice_coef
                                 })


for image_batch, label_batch in data_gen.generate_data(batch_size=len(x_test), test=True):
    prediction = model.predict(image_batch, verbose=1)
    save_results(prediction, 'rgb', save_result_path, test_label_filenames_list)
    break
