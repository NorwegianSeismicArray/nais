
import keras_tuner as kt
from sklearn.model_selection import KFold
import numpy as np

class CVTuner(kt.engine.tuner.Tuner):
    """
    Add cross validation to keras tuner.
    """

    def run_trial(self, trial, x, y, batch_size=32, epochs=1, cv=KFold(5), callbacks=None, **kwargs):
        """
        batch_size : int
        epochs : int
        cv : cross validation splitter.
            Should have split method that accepts x and y and returns train and test indicies.
        callbacks : function that returns keras.callbacks.Callback instaces (in a list).
            eg. callbacks = lambda : [keras.Callbacks.EarlyStopping('val_loss')]
        """
        val_metrics = []
        oof_p = np.zeros(y.shape)
        for train_indices, test_indices in cv.split(x, y):
            x_train, x_test = x[train_indices], x[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            model = self.hypermodel.build(trial.hyperparameters)

            if callbacks is not None:
                cb = callbacks()
            else:
                cb = None

            model.fit(x_train, y_train,
                      validation_data=(x_test, y_test),
                      batch_size=batch_size,
                      epochs=epochs,
                      callbacks=cb,
                      **kwargs)
            val_metrics.append(model.evaluate(x_test, y_test))
            metrics = model.metrics_names

        val_metrics = np.mean(np.asarray(val_metrics), axis=0)
        res = dict(zip(metrics,val_metrics))

        return res
