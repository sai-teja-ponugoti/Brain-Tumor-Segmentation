from keras import backend as K
import os
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class Train:

    def __init__(self, model, learning_rate, epochs, batch_size, model_dest_path, train_data_generator,
                 len_train_images, len_val_images,
                 train_masks, val_data_generator, val_masks, smooth=100):
        self.model = model
        self.learning_rate = learning_rate
        self.model_dest_path = model_dest_path
        self.train_data_generator = train_data_generator
        self.val_data_generator = val_data_generator
        self.batch_size = batch_size
        self.smooth = smooth
        self.epochs = epochs
        self.len_train_images = len_train_images
        self.len_val_images = len_val_images
        self.callbacks = [
            ModelCheckpoint(os.path.join(model_dest_path, 'unet_brain_mri_seg.hdf5'), verbose=1, save_best_only=True)]

    def plotGraph(self, type_of_graph, path):
        plt.plot(self.history.history[type_of_graph])
        plt.plot(self.history.history['val_' + type_of_graph])
        plt.title('model ' + type_of_graph)
        plt.ylabel(type_of_graph)
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        # plt.show()
        plt.savefig(os.path.join(path, type_of_graph + '_graph.png'))

    def dice_coef(self, y_true, y_pred):
        """
        Dice = (2*|X & Y|)/ (|X|+ |Y|)
            =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
        """
        y_truef = K.flatten(y_true)
        y_predf = K.flatten(y_pred)
        And = K.sum(y_truef * y_predf)
        return (2 * And + self.smooth) / (K.sum(y_truef) + K.sum(y_predf) + self.smooth)

    def dice_coef_loss(self, y_true, y_pred):
        return 1 - self.dice_coef(y_true, y_pred)

    def iou(self, y_true, y_pred):
        intersection = K.sum(y_true * y_pred)
        sum_ = K.sum(y_true + y_pred)
        jac = (intersection + self.smooth) / (sum_ - intersection + self.smooth)
        return jac

    def jac_distance(self, y_true, y_pred):
        """
        Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
                = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

        The jaccard distance loss is usefull for unbalanced datasets. This has been
        shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
        gradient.
        """
        return 1 - self.iou(y_true, y_pred)

    def save_model(self, dest_path):
        self.model.save(dest_path)

    def optimizer(self):
        return Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None,
                    decay=self.learning_rate / self.epochs, amsgrad=False)

    def train(self):
        self.model.compile(optimizer=self.optimizer(), loss=self.dice_coef_loss,
                           metrics=["binary_accuracy", self.iou, self.dice_coef, self.jac_distance])

        self.history = self.model.fit(self.train_data_generator,
                                      steps_per_epoch=self.len_train_images / self.batch_size,
                                      epochs=self.epochs,
                                      callbacks=self.callbacks,
                                      validation_data=self.validation_data_generator,
                                      validation_steps=self.len_val_images / self.batch_size)

        self.model.save(os.path.join(self.model_dest_path, "ctModel"))

        self.plotGraph("binary_accuracy", self.model_dest_path)
        self.plotGraph("dice_coef", self.model_dest_path)
        self.plotGraph("iou", self.model_dest_path)
        self.plotGraph("jac_distance", self.model_dest_path)
