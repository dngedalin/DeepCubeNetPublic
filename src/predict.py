from keras.layers import UpSampling3D
from net_blocks import *
from utils import *
import os
import time
import tensorflow as tf
import yaml
with open("config/CS_MUSI.yaml", 'r') as ymlfile:
    config = yaml.load(ymlfile,Loader=yaml.FullLoader)

if config['CPU']:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print('using CPU')
tf_config = tf.ConfigProto(device_count={"CPU": 1})
keras.backend.tensorflow_backend.set_session(tf.Session(config=tf_config))
test_path = config['inputs_dir']
output_path = config['outputs_dir']


#%% Model
input1 = keras.layers.Input(shape=(64, 64, 32))

bp = keras.layers.Conv2D(391, (1, 1), activation=None,trainable=False,bias=False,name='bp1')(input1) # Backprojection

bp_exp=keras.layers.Lambda(expand)(bp)



d_1 = conv3D_block(8,(3,3,11),input = bp_exp)

d_1 = conv3D_block(8,(3,3,11),input = d_1)


d_1_p = keras.layers.MaxPooling3D((1,1,2))(d_1)

d_2 = conv3D_block(16 ,(3,3,9),input = d_1_p)
d_2 = conv3D_block(16 ,(3,3,9),input = d_2)

d_2_p = keras.layers.MaxPooling3D((1,1,2))(d_2)

d_3 = conv3D_block(32,(3,3,7),input = d_2_p)
d_3 = conv3D_block(32,(3,3,7),input = d_3)


d_3_p =  keras.layers.MaxPooling3D((1,1,2))(d_3)


d_4 = conv3D_block(128,(3,3,5),input = d_3_p)

d_4 = conv3D_block(128,(3,3,5),input = d_4)



# Up

u_1 = UpSampling3D((1,1,3))(d_4)

u_1= keras.layers.Cropping3D(cropping=((0,0),(0,0),(0,47)))(u_1)

u_1 = conv3D_block(32,(3,3,7),input = u_1 )
u_1 = conv3D_block(32,(3,3,7),input = u_1 ,concat= d_3)


u_2 = UpSampling3D((1,1,2))(u_1)


d_2_c= keras.layers.Cropping3D(cropping=((0,0),(0,0),(0,1)))(d_2)

u_2 = conv3D_block(16,(3,3,9),input = u_2 )
u_2 = conv3D_block(16,(3,3,9),input = u_2,concat=d_2_c)


u_3 = UpSampling3D((1,1,3))(u_2)

u_3_c= keras.layers.Cropping3D(cropping=((0,0),(0,0),(0,89+102)))(u_3)

u_3 = conv3D_block(8,(3,3,11),input = u_3_c )

u_3 = conv3D_block(8,(3,3,11),input = u_3,concat=d_1 )



# Final Projection

pr =  conv3D_block(1,(1,1,1),input=u_3) # Must be with spectral filter 1

final = keras.layers.Lambda(squeeze)(pr)


model = keras.models.Model(input=input1, output=final)


model.compile(optimizer=keras.optimizers.Adam(0.0001), loss='mse',
          metrics=[psnr,'mse','mae',SSIM])


model_full_path = os.path.join(config['model_path'],config['model_name'])
model.load_weights(model_full_path)

file_list = os.listdir(test_path)

i=0
j=0
count=0
pred_time = []
for file in file_list:
    data = loadmat(os.path.join(test_path,file))
    filename_w_ext = os.path.basename(file)
    filename, file_extension = os.path.splitext(filename_w_ext)
    Rec = np.zeros(np.shape(data['gt']))
    GT = np.zeros(np.shape(data['gt']))
    i=0

    while i+64<=np.shape(data['cs'])[0]:
        j=0
        while j+64<=np.shape(data['cs'])[1]:
            x =data['cs'][i:i+64,j:j+64,:]
            y= np.expand_dims(x,axis=0)
            t = time.time()
            Rec[i:i+64,j:j+64,:]   = model.predict(y)
            print(time.time()-t)
            pred_time.append(time.time() - t)
            GT[i:i + 64, j:j + 64, :] = data['gt'][i:i+64,j:j+64,:]
            count+=1
            j+=64
        i+=64
    savemat(os.path.join(output_path, filename) + '.mat',
            mdict={'gt': GT, 'cs': data['cs'], 'predicted': Rec})
    print(f'last time measured is {pred_time[-1]}')
pred_time = np.array(pred_time)
print(f'avg time for prediction is : {np.mean(pred_time[1:])}')
print(f'STD of time for prediction is : {np.std(pred_time[1:])}')
