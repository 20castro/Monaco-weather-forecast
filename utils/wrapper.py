import tensorflow as tf

class ResidualWrapper(tf.keras.Model):

    def __init__(self, model, residual_index, n_ouput):
        super().__init__()
        self.model = model
        self.residual_index= residual_index
        self.n_output = n_ouput

    def call(self, inputs, *args, **kwargs):
        delta = self.model(inputs, *args, **kwargs)
        for k, index in enumerate(self.residual_index):
            if index is None:
                pass
            else:
                delta[k] += inputs[:, :self.n_output, index]
        return delta
