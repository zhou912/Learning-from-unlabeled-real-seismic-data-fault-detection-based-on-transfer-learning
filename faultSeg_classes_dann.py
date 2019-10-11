import numpy as np
import keras
from keras.utils import to_categorical

class dann_DataGenerator(keras.utils.Sequence):
  'Generates data for keras'
  def __init__(self,s_dpath,s_fpath,t_dpath,t_fpath,data_IDs, batch_size=1, dim=(128,128,128),
             n_channels=1, shuffle=True):
    'Initialization'
    self.dim   = dim
    self.s_dpath = s_dpath
    self.s_fpath = s_fpath
    self.t_dpath = t_dpath
    self.t_fpath = t_fpath
    self.batch_size = batch_size
    self.data_IDs   = data_IDs
    self.n_channels = n_channels
    self.shuffle    = shuffle
    self.on_epoch_end()

  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(len(self.data_IDs)/self.batch_size))

  def __getitem__(self, index):
    'Generates one batch of data'
    # Generate indexes of the batch
    bsize = self.batch_size
    indexes = self.indexes[index*bsize:(index+1)*bsize]
    print (indexes)
    # Find list of IDs
    data_IDs_temp = [self.data_IDs[k] for k in indexes]

    # Generate data
    X, Y = self.__dann_data_generation(data_IDs_temp)
    return X, Y

  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(len(self.data_IDs))
    if self.shuffle == True:
      np.random.shuffle(self.indexes)

  def __dann_data_generation(self, data_IDs_temp):
    'Generates data containing batch_size samples'
    # Initialization
    X_s = np.zeros((4, *self.dim, self.n_channels),dtype=np.single)
    Y_s = np.zeros((4, *self.dim, self.n_channels),dtype=np.single)
    gx  = np.fromfile(self.s_dpath+str(data_IDs_temp[0])+'.dat',dtype=np.single)
    fx  = np.fromfile(self.s_fpath+str(data_IDs_temp[0])+'.dat',dtype=np.single)
    gx = np.reshape(gx,self.dim)
    fx = np.reshape(fx,self.dim)
    gx = gx-np.min(gx)
    gx = gx/np.max(gx)
    gx = gx*255
    # Generate data
    for i in range(4):
      X_s[i,] = np.reshape(np.rot90(gx,i,(0,1)), (*self.dim,self.n_channels))
      Y_s[i,] = np.reshape(np.rot90(fx,i,(0,1)), (*self.dim,self.n_channels))

    X_t = np.zeros((4, *self.dim, self.n_channels),dtype=np.single)
    Y_t = np.zeros((4, *self.dim, self.n_channels), dtype=np.single)
    gx_t  = np.fromfile(self.t_dpath+str(data_IDs_temp[0])+'.dat',dtype=np.single)
    gx_t = np.reshape(gx_t,self.dim)
    gx_t = gx_t-np.min(gx_t)
    gx_t = gx_t/np.max(gx_t)
    gx_t = gx_t*255
    for i in range(4):
      X_t[i,] = np.reshape(np.rot90(gx_t,i,(0,1)), (*self.dim,self.n_channels))
    X = np.concatenate([X_s,X_t])
    Y = np.concatenate([Y_s,Y_t])
    return X,Y