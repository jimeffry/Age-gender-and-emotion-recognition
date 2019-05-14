#import pandas as pd
import logging
import argparse
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras import backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from keras.optimizers import SGD, Adam
from wide_resnet import WideResNet
from utils import mk_dir, load_data
import numpy as np
from batch_geneter import BatchLoader
from keras.utils import to_categorical
from cnn import mini_XCEPTION
from keras.utils import multi_gpu_model

logging.basicConfig(level=logging.DEBUG)
whole_data = 1
lr_base = 0.01
epochs = 30000
lr_power = 0.9

class Schedule:
    def __init__(self, nb_epochs):
        self.epochs = nb_epochs

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return 0.1
        elif epoch_idx < self.epochs * 0.5:
            return 0.02
        elif epoch_idx < self.epochs * 0.75:
            return 0.004
        return 0.0008


def get_args():
    parser = argparse.ArgumentParser(description="This script trains the CNN model for age and gender estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, default='./data/imdb_train.mat',
                        help="path to input database mat file")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=300,
                        help="number of epochs")
    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network (should be 10, 16, 22, 28, ...)")
    parser.add_argument("--width", type=int, default=2,
                        help="width of network")
    parser.add_argument("--validation_split", type=float, default=0.1,
                        help="validation split ratio")
    parser.add_argument("--img_size",type=int,default=64,
                        help='net input size')
    parser.add_argument("--val_path",type=str,default='./data/imdb_val.mat',
                        help='path to validation data mat file ')
    parser.add_argument("--pretrained_fil",type=str,default=None,
                        help='before pretrained_file for net')
    parser.add_argument("--lr", type=float, default=0.001,
                        help='the net training learningrate')
    parser.add_argument("--gpus", type=int, default=1,
                        help='how many gpus for trainning')
    parser.add_argument("--db",type=str,default="FGNET",\
                        help="training on which dataset")
    args = parser.parse_args()
    return args

def batch_geneter(batchloader):
    imgs, gender,age = batchloader.next_batch()
    y_data_g = to_categorical(gender, 2)
    y_data_a = to_categorical(age, 101)
    imgs = (imgs - 127.5)*0.0078125
    #return imgs,y_data_g,y_data_a
    return imgs,y_data_g

def data_geneter(batchloader):
    while True:
        imgs, gender,age = batchloader.next_batch()
        y_data_g = to_categorical(gender, 2)
        y_data_a = to_categorical(age, 101)
        imgs = (imgs - 127.5)*0.0078125
        #yield imgs,[y_data_g,y_data_a]
        yield imgs,y_data_a

def lr_schedule(epoch_idx):
    base_lr=0.01,
    decay=0.9
    epoch_idx = int(epoch_idx)
    return base_lr * decay**(epoch_idx)


def lr_scheduler(epoch, mode='progressive_drops'):
    '''if lr_dict.has_key(epoch):
        lr = lr_dict[epoch]
        print 'lr: %f' % lr'''

    if mode is 'power_decay':
        # original lr scheduler
        lr = lr_base * ((1 - float(epoch) / epochs) ** lr_power)
    if mode is 'exp_decay':
        # exponential decay
        lr = (float(lr_base) ** float(lr_power)) ** float(epoch + 1)
    # adam default lr
    if mode is 'adam':
        lr = 0.001

    if mode is 'progressive_drops':
        # drops as progression proceeds, good for sgd
        if epoch > 0.9 * epochs:
            lr = 0.00001
        elif epoch > 0.75 * epochs:
            lr = 0.0001
        elif epoch > 0.5 * epochs:
            lr = 0.001
        else:
            lr = 0.01

    print('lr: %f' % lr)
    return lr

def main():
    args = get_args()
    input_path = args.input
    batch_size = args.batch_size*args.gpus
    nb_epochs = args.nb_epochs
    depth = args.depth
    k = args.width
    validation_split = args.validation_split
    img_size = args.img_size
    val_path = args.val_path
    pretrained_fil = args.pretrained_fil
    input_shape = [img_size, img_size, 3]
    patience = 30
    gpu_num = args.gpus
    train_db = args.db

    logging.debug("Loading data...")
    '''
    image, gender, age, _, image_size, _ = load_data(input_path)
    X_data = image
    y_data_g = np_utils.to_categorical(gender, 2)
    y_data_a = np_utils.to_categorical(age, 101)
    '''
    batchdataload = BatchLoader(input_path,batch_size,img_size,train_db)
    valdataload = BatchLoader(val_path,batch_size,img_size)
    model = WideResNet(img_size, depth=depth, k=k)()
    #model = mini_XCEPTION(input_shape,101)
    with open(os.path.join("ag_models", "WRN_{}_{}.json".format(depth, k)), "w") as f:
        f.write(model.to_json())
    if pretrained_fil :
        model.load_weights(pretrained_fil)
    #sgd = SGD(lr=0.001, momentum=0.7, nesterov=True)
    adam = Adam(lr=0.0001,beta_1=0.9,beta_2=0.999,epsilon=1e-5)
    #model.compile(optimizer=sgd, loss=["categorical_crossentropy", "categorical_crossentropy"],
    #              metrics=['accuracy'])
    #if gpu_num >1:
        #model = multi_gpu_model(model,gpu_num)
    model.compile(optimizer=adam, loss=["categorical_crossentropy"],
                  metrics=['accuracy'])
    logging.debug("Model summary...")
    #model.count_params()
    model.summary()
    logging.debug("Saving model...")
    if not os.path.exists("./ag_models"):
        mk_dir("ag_models")

    reduce_lr = ReduceLROnPlateau(monitor="val_loss",factor=0.1,patience=patience*2,verbose=1,min_lr=0.0000001)
    early_stop = EarlyStopping('val_loss', patience=patience)
    modelcheckpoint = ModelCheckpoint("ag_models/weights.{epoch:02d}-{val_loss:.2f}.hdf5",\
                        monitor="val_loss",verbose=1,save_best_only=True,mode="auto",period=1000)
    #mk_dir("checkpoints")
    #reduce_lr = LearningRateScheduler(schedule=reduce_lr)
    #callbacks = [modelcheckpoint,early_stop,reduce_lr]
    callbacks = [modelcheckpoint,reduce_lr]
    logging.debug("Running training...")
    #whole training
    error_min = 0
    if whole_data :
        hist = model.fit_generator(data_geneter(batchdataload), steps_per_epoch=batchdataload.batch_num,
                              epochs=nb_epochs, verbose=1,
                              callbacks=callbacks,
                              validation_data=data_geneter(valdataload),
                              nb_val_samples=valdataload.batch_num,
                              nb_worker=1)
        logging.debug("Saving weights...")
        model.save_weights(os.path.join("ag_models", "WRN_{}_{}.h5".format(depth, k)),overwrite=True)
        #pd.DataFrame(hist.history).to_hdf(os.path.join("ag_models", "history_{}_{}.h5".format(depth, k)), "history")
    else:
        epoch_step = 0
        while epoch_step < nb_epochs:
            step = 0
            while step < batchdataload.batch_num:
                #X_data, y_data_g, y_data_a = batch_geneter(batchdataload)
                X_data, y_data_g = batch_geneter(batchdataload)
                #hist = model.fit(X_data, [y_data_g, y_data_a], batch_size=batch_size, epochs=1, verbose=2)
                hist = model.fit(X_data, y_data_g, batch_size=batch_size, epochs=1, verbose=2)
                step+=1
                if step % 100 ==0:
                    #val_data,val_g,val_a = batch_geneter(valdataload)
                    val_data,val_g = batch_geneter(valdataload)
                    #error_t = model.evaluate(val_data,[val_g,val_a],batch_size=batch_size,verbose=1)
                    error_t = model.evaluate(val_data,val_g,batch_size=batch_size,verbose=1)
                    print ("****** Epoch {} Step {}: ***********".format(str(epoch_step),str(step)) )
                    print (" loss: {}".format(error_t))
                    if epoch_step % 5 ==0:
                        #logging.debug("Saving weights...")
                        #val_data,val_g,val_a = batch_geneter(valdataload)
                        #error_t = model.evaluate(val_data,[val_g,val_a],batch_size=batch_size,verbose=1)
                        if error_t[4] >error_min:
                            logging.debug("Saving weights...")
                            model.save_weights(os.path.join("ag_models", "WRN_{}_{}_epoch{}_step{}.h5".format(depth, k,epoch_step,step)))
                            error_min = error_t[4]
            epoch_step+=1
            if epoch_step % 5 ==0:
                logging.debug("Saving weights...")
                #val_data,val_g,val_a = batch_geneter(valdataload)
                val_data,val_g = batch_geneter(valdataload)
                #error_t = model.evaluate(val_data,[val_g,val_a],batch_size=batch_size,verbose=1)
                error_t = model.evaluate(val_data,val_g,batch_size=batch_size,verbose=1)
                if error_t[1] >error_min:
                    model.save_weights(os.path.join("ag_models", "WRN_{}_{}_epoch{}.h5".format(depth, k,epoch_step)))
                    error_min = error_t[4]
                error_min =0
                #pd.DataFrame(hist.history).to_hdf(os.path.join("ag_models", "history_{}_{}.h5".format(depth, k)), "history")

def set_keras_backend(backend):
    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend

if __name__ == '__main__':
    set_keras_backend('tensorflow')
    main()
