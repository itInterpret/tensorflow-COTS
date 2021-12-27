from functools import partial

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam

from nets.yolo import get_train_model, yolo_body
from nets.yolo_training import get_yolo_loss
from utils.callbacks import (ExponentDecayScheduler, LossHistory,
                             ModelCheckpoint, WarmUpCosineDecayScheduler)
from utils.dataloader import YoloDatasets
from utils.utils import get_classes
from utils.utils_fit import fit_one_epoch

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
if __name__ == "__main__":

    eager = False 
    classes_path    = 'model_data/voc_classes.txt'
    model_path      = 'model_data/yolox_x.h5'
    input_shape     = [1280, 1280]
    phi             = 'x'
    
    mosaic              = True
    Cosine_scheduler    = True

    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 8
    Freeze_lr           = 1e-3
    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 8
    Unfreeze_lr         = 1e-4
    Freeze_Train        = False

    num_workers         = 4

    train_annotation_path   = '2022_train.txt'
    val_annotation_path     = '2022_val.txt'

    class_names, num_classes = get_classes(classes_path)

    model_body  = yolo_body([None, None, 3], num_classes = num_classes, phi = phi)
    if model_path != '':

        print('Load weights {}.'.format(model_path))
        model_body.load_weights(model_path, by_name=True, skip_mismatch=True)

    if not eager:
        model = get_train_model(model_body, input_shape, num_classes)
    else:
        yolo_loss = get_yolo_loss(input_shape, len(model_body.output), num_classes)

    logging         = TensorBoard(log_dir = 'logs/')
    checkpoint      = ModelCheckpoint('logs/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                            monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = 1)
    if Cosine_scheduler:
        reduce_lr   = WarmUpCosineDecayScheduler(T_max = 5, eta_min = 1e-5, verbose = 1)
    else:
        reduce_lr   = ExponentDecayScheduler(decay_rate = 0.94, verbose = 1)
    early_stopping  = EarlyStopping(monitor='val_loss', min_delta = 0, patience = 10, verbose = 1)
    loss_history    = LossHistory('logs/')


    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    if Freeze_Train:
        freeze_layers = {'tiny': 125, 's': 125, 'm': 179, 'l': 234, 'x': 290}[phi]
        for i in range(freeze_layers): model_body.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_body.layers)))
        
    if True:
        batch_size  = Freeze_batch_size
        lr          = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch   = Freeze_Epoch

        epoch_step          = num_train // batch_size
        epoch_step_val      = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('dataset is l...')

        train_dataloader    = YoloDatasets(train_lines, input_shape, batch_size, num_classes, end_epoch - start_epoch, mosaic = mosaic, train = True)
        val_dataloader      = YoloDatasets(val_lines, input_shape, batch_size, num_classes, end_epoch - start_epoch, mosaic = False, train = False)

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        if eager:
            gen     = tf.data.Dataset.from_generator(partial(train_dataloader.generate), (tf.float32, tf.float32))
            gen_val = tf.data.Dataset.from_generator(partial(val_dataloader.generate), (tf.float32, tf.float32))

            gen     = gen.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)
            gen_val = gen_val.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)

            if Cosine_scheduler:
                lr_schedule = tf.keras.experimental.CosineDecayRestarts(
                    initial_learning_rate = lr, first_decay_steps = 5 * epoch_step, t_mul = 1.0, alpha = 1e-2)
            else:
                lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate = lr, decay_steps = epoch_step, decay_rate=0.94, staircase=True)
            
            optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)

            for epoch in range(start_epoch, end_epoch):
                fit_one_epoch(model_body, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, 
                            end_epoch, input_shape, num_classes)

        else:
            model.compile(optimizer=Adam(lr = lr), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

            model.fit_generator(
                generator           = train_dataloader,
                steps_per_epoch     = epoch_step,
                validation_data     = val_dataloader,
                validation_steps    = epoch_step_val,
                epochs              = end_epoch,
                initial_epoch       = start_epoch,
                use_multiprocessing = True if num_workers > 1 else False,
                workers             = num_workers,
                callbacks           = [logging, checkpoint, reduce_lr, early_stopping, loss_history]
            )

    if Freeze_Train:
        for i in range(freeze_layers): model_body.layers[i].trainable = True

    if True:
        batch_size  = Unfreeze_batch_size
        lr          = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch   = UnFreeze_Epoch

        epoch_step          = num_train // batch_size
        epoch_step_val      = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('more dataset')

        train_dataloader    = YoloDatasets(train_lines, input_shape, batch_size, num_classes, end_epoch - start_epoch, mosaic = mosaic, train = True)
        val_dataloader      = YoloDatasets(val_lines, input_shape, batch_size, num_classes, end_epoch - start_epoch, mosaic = False, train = False)

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        if eager:
            gen     = tf.data.Dataset.from_generator(partial(train_dataloader.generate), (tf.float32, tf.float32))
            gen_val = tf.data.Dataset.from_generator(partial(val_dataloader.generate), (tf.float32, tf.float32))

            gen     = gen.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)
            gen_val = gen_val.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)

            if Cosine_scheduler:
                lr_schedule = tf.keras.experimental.CosineDecayRestarts(
                    initial_learning_rate = lr, first_decay_steps = 5 * epoch_step, t_mul = 1.0, alpha = 1e-2)
            else:
                lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate = lr, decay_steps = epoch_step, decay_rate=0.94, staircase=True)
            
            optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)

            for epoch in range(start_epoch, end_epoch):
                fit_one_epoch(model_body, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, 
                            end_epoch, input_shape, num_classes)

        else:
            model.compile(optimizer=Adam(lr = lr), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

            model.fit_generator(
                generator           = train_dataloader,
                steps_per_epoch     = epoch_step,
                validation_data     = val_dataloader,
                validation_steps    = epoch_step_val,
                epochs              = end_epoch,
                initial_epoch       = start_epoch,
                use_multiprocessing = True if num_workers > 1 else False,
                workers             = num_workers,
                callbacks           = [logging, checkpoint, reduce_lr, early_stopping, loss_history]
            )
