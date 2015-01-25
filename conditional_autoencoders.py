from lasagne import easy
import lasagne
import lasagne.data
import theano.tensor as T
import theano
import os
import numpy as np
import time



from lasagne import easy
import copy
import lasagne
import lasagne.data
import theano.tensor as T
import theano
import os
import numpy as np
import time

from lasagne.layers.dense import DenseLayer
from lasagne.layers.base import Layer

class ProjectionLayer(Layer):

    def __init__(self, input_layer, indices):
        super(ProjectionLayer, self).__init__(input_layer)
        self.indices = indices

    def get_output_shape_for(self, input_shape):
        return (input_shape[0],  len(self.indices) )  

    def get_output_for(self, input, *args, **kwargs):
        return input[:, self.indices]

class AutoencoderLayer(DenseLayer):
    
    def __init__(self, *args, **kwargs):
        super(AutoencoderLayer, self).__init__(*args, **kwargs)

        tied = kwargs.get("tied", True)
        
        if tied:
            self.W_T = self.create_param(self.W, self.W.shape)
        else:
            self.W_T = self.W.T


        self.b_T = self.create_param(b, (self.num_units,))

    def decode(self, output, *args, **kwargs):
        input = T.dot(self.W_T, output)
        activation = activation + self.b_T.dimshuffle('x', 0)
        return self.nonlinearity(input)


def build_model_conditional_autoencoder(x_dim, y_dim, num_hidden_units_x, num_hidden_units_joint):

    l_x_in = lasagne.layers.InputLayer(
            shape=(None, x_dim),
        )
    l_y_in = lasagne.layers.InputLayer(
            shape=(None, y_dim),
        )
 
    l_hidden_x = lasagne.layers.DenseLayer(
        l_x_in,
        num_units=num_hidden_units_x,
        nonlinearity=lasagne.nonlinearities.rectify,
        )

    l_joint = lasagne.layers.merge.ConcatLayer([l_hidden_x, l_y_in])
    """
    l_y_hidden = lasagne.layers.DenseLayer(
        l_y_in,
        num_units = num_hidden_units_joint,
        nonlinearity=lasagne.nonlinearities.rectify,
    )

    l_x_hidden_prime = lasagne.layers.DenseLayer(
        l_hidden_x,
        num_units = num_hidden_units_joint,
        nonlinearity=lasagne.nonlinearities.rectify,
    )

    l_joint = lasagne.layers.merge.ElemwiseSumLayer([l_y_hidden, l_x_hidden_prime])
    """
    l_x_hat_from_joint = lasagne.layers.DenseLayer(
        l_joint,
        num_units=x_dim,
        nonlinearity=lasagne.nonlinearities.sigmoid,
    )

    l_y_hat_from_joint = lasagne.layers.DenseLayer(
        l_joint,
        num_units=y_dim,
        nonlinearity=lasagne.nonlinearities.sigmoid,
    )

    l_x_hat_from_hidden = lasagne.layers.DenseLayer(
        l_hidden_x,
        num_units=x_dim,
        nonlinearity=lasagne.nonlinearities.sigmoid,
    )
    return l_x_in, l_y_in, l_hidden_x, l_x_hat_from_joint, l_x_hat_from_hidden, l_y_hat_from_joint


def build_model_autoencoder(x_dim, num_hidden):
    l_x_in = lasagne.layers.InputLayer(
            shape=(None, x_dim),
        )
    l_hidden_x = lasagne.layers.DenseLayer(
        l_x_in,
        num_units=num_hidden,
        nonlinearity=lasagne.nonlinearities.rectify,
    )

    l_x_hat = lasagne.layers.DenseLayer(
        l_hidden_x,
        num_units=x_dim,
        nonlinearity=lasagne.nonlinearities.sigmoid,
    )


    return l_hidden_x, l_x_hat


def autoencoder_objective(x, x_hat):
    o =  T.mean( (x  - x_hat) ** 2)
    return o


def conditional_autoencoder_objective(x, y, x_hat_from_hidden, x_hat_from_joint, y_hat_from_joint):
    return T.mean((x - x_hat_from_hidden)**2) + T.mean( (x - x_hat_from_joint)**2 ) + T.mean((y - y_hat_from_joint)**2)



class Conditional(lasagne.easy.Experiment):

    def __init__(self):
        pass

    def load_data(self):
        datasets = lasagne.data.Mnist()
        datasets.load(ratio_valid=0.2)

        for d in datasets.values():
            d.y = theano.shared(lasagne.utils.floatX(lasagne.easy.to_hamming(d.y.get_value(), presence=1, absence=0)))
        return datasets["train"], datasets["valid"], datasets["test"]

    def run(self):
        # Load data
        train, valid, test = self.load_data()

        # Setting hyper-params
        self.learning_rate, self.momentum = 0.6, 0.8

        self.input_dim = train.X.get_value().shape[1]
        self.output_dim  = 10
        self.batch_size = 100
        self.x_hidden_units = 100
        self.y_hidden_units = 10
        self.joint_hidden_units = 50
        self.nb_batches = train.X.get_value().shape[0] // self.batch_size
        self.nb_epochs = 100

        # config
        self.save_each = 10
        self.batch_index, self.X_batch, self.y_batch, self.batch_slice = easy.get_theano_batch_variables(self.batch_size)

        # build model
        (self.l_x_in, self.l_y_in, self.l_hidden_x, self.l_x_hat_from_joint, 
         self.l_x_hat_from_hidden, self.l_y_hat_from_joint) = build_model_conditional_autoencoder(self.input_dim, self.output_dim, 
                                                                                                  self.x_hidden_units, self.joint_hidden_units)
    
        # get the loss
        self.loss = conditional_autoencoder_objective(self.X_batch, 
                                                      self.y_batch,
                                                      self.l_x_hat_from_joint.get_output({self.l_x_in : self.X_batch, self.l_y_in: self.y_batch}), 
                                                      self.l_x_hat_from_hidden.get_output(self.X_batch),
                                                      self.l_y_hat_from_joint.get_output({self.l_x_in : self.X_batch, self.l_y_in: self.y_batch}))
        self.get_loss = theano.function([self.X_batch, self.y_batch], self.loss)
        self.get_reconstruction = theano.function([self.X_batch, self.y_batch], 
                                                  self.l_x_hat_from_joint.get_output({self.l_x_in : self.X_batch, self.l_y_in: self.y_batch}))


        from theano.sandbox.rng_mrg import MRG_RandomStreams
        
        random_stream = MRG_RandomStreams(2014 * 5 + 27)
        X_batch_prime = T.matrix('xprime')

        weights = (random_stream.uniform(low=0, high=1, size=(X_batch_prime.shape[0], self.y_batch.shape[0]) ) * 
                   random_stream.binomial(n=1, p=0.001, size=(X_batch_prime.shape[0], self.y_batch.shape[0])) )
        weights = weights / weights.sum(axis=0).reshape( (1, self.y_batch.shape[0]) )
        
        self.generate = theano.function([X_batch_prime, self.y_batch], 
            self.l_x_hat_from_joint.get_output({self.l_hidden_x : T.dot(self.l_hidden_x.get_output(X_batch_prime).T, weights).T, self.l_y_in: self.y_batch})
        )

        # get the gradient updates
        all_params = lasagne.layers.get_all_params(self.l_x_hat_from_joint)
        #updates = lasagne.updates.nesterov_momentum(loss, all_params, learning_rate, momentum)
        self.updates = lasagne.updates.adadelta(self.loss, all_params, learning_rate=self.learning_rate)
        #updates = lasagne.updates.Adam(loss, all_params, learning_rate=learning_rate)

        # get the iteration update
        iter_update_batch = easy.get_iter_update_supervision(train.X, train.y,
                                                              self.X_batch, self.y_batch,
                                                              self.loss, self.updates,
                                                              self.batch_index, self.batch_slice)

        def iter_update():
            for i in xrange(self.nb_batches):
                iter_update_batch(i)

            loss_train = self.get_loss(train.X.get_value(), train.y.get_value())
            loss_valid = self.get_loss(valid.X.get_value(), valid.y.get_value())
            loss_test = self.get_loss(test.X.get_value(), test.y.get_value())
            return {"train": loss_train, "valid": loss_valid, "test": loss_test}

        def quitter(update_status):
            return False
        
        def monitor(update_status):
            return update_status

        def observer(monitor_output):
            for k, v in monitor_output.items():
                print("%s : %f" % (k, v))
        
        lasagne.easy.main_loop(self.nb_epochs, iter_update, quitter, monitor, observer)
        

        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt

        # reconstruction


        shape = (28, 28)

        x_rec = self.get_reconstruction(train.X.get_value(), train.y.get_value())
        nb = 10
        k = 1
        plt.axis('off')
        for i in xrange(nb):

            ind = i
            x_i = train.X.get_value()[ind].reshape(shape)
            x_rec_i = x_rec[ind].reshape(shape)

            plt.subplot(nb, 2, k)
            plt.imshow(x_i, cmap='gray')
            k += 1
            plt.subplot(nb, 2, k)
            plt.imshow(x_rec_i, cmap='gray')
            k += 1

        plt.savefig("mnist-rec-cond.png")

        plt.clf()
        # generation
        from pylearn2.scripts.plot_weights import grid_plot
        y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        y = lasagne.easy.to_hamming(y, presence=1, absence=0, nb_classes=10)
        y = y.astype(np.float32)

        x_gen = self.generate(train.X.get_value(), y)
        x_gen = x_gen.reshape( (x_gen.shape[0], shape[0], shape[1])  )
        grid_plot(x_gen, imshow_options={"cmap": "gray"})

        plt.savefig("mnist-gen-cond.png")
        
        fd = open("model.pkl", "w")
        self.save(fd)
        fd.close()
         

class Normal(lasagne.easy.Experiment):

    def __init__(self):
        pass

    def load_data(self):
        datasets = lasagne.data.Mnist()
        datasets.load(ratio_valid=0.2)

        for d in datasets.values():
            d.y = theano.shared(lasagne.utils.floatX(lasagne.easy.to_hamming(d.y.get_value(), presence=1, absence=0)))
        return datasets["train"], datasets["valid"], datasets["test"]

    def run(self):
        # Load data
        train, valid, test = self.load_data()

        # Setting hyper-params
        self.learning_rate, self.momentum = 0.6, 0.8

        self.input_dim = train.X.get_value().shape[1]
        self.output_dim  = 10
        self.batch_size = 100
        self.x_hidden_units = 50
        self.nb_batches = train.X.get_value().shape[0] // self.batch_size
        self.nb_epochs = 30

        # config
        self.batch_index, self.X_batch, self.y_batch, self.batch_slice = easy.get_theano_batch_variables(self.batch_size)

        # build model
        self.l_hidden_x, self.l_x_hat = build_model_autoencoder(self.input_dim, self.x_hidden_units)

        # get the loss
        self.loss = autoencoder_objective(self.X_batch, 
                                         self.l_x_hat.get_output(self.X_batch),
                                         self.l_hidden_x.get_output(self.X_batch))
        self.get_loss = theano.function([self.X_batch], self.loss)
        self.get_reconstruction = theano.function([self.X_batch], 
                                                  self.l_x_hat.get_output(self.X_batch))

        from theano.sandbox.rng_mrg import MRG_RandomStreams
        
        random_stream = MRG_RandomStreams(2014 * 5 + 27)
        X_batch_prime = T.matrix('xprime')


        nb = T.iscalar('nb')
        weights = (random_stream.uniform(low=0, high=1, size=(X_batch_prime.shape[0], nb) ) * 
                   random_stream.binomial(n=1, p=0.0001, size=(X_batch_prime.shape[0], nb)) )
        weights = weights / weights.sum(axis=0).reshape( (1, nb) )
        self.generate = theano.function([X_batch_prime, nb], 
            self.l_x_hat.get_output({self.l_hidden_x : T.dot(self.l_hidden_x.get_output(X_batch_prime).T, weights).T})
        )

        # get the gradient updates
        all_params = lasagne.layers.get_all_params(self.l_x_hat)
        #updates = lasagne.updates.nesterov_momentum(loss, all_params, learning_rate, momentum)
        self.updates = lasagne.updates.adadelta(self.loss, all_params, learning_rate=self.learning_rate)
        #updates = lasagne.updates.Adam(loss, all_params, learning_rate=learning_rate)

        # get the iteration update
        iter_update_batch = easy.get_iter_update_nonsupervision(train.X, self.X_batch, self.loss, self.updates,
                                                                self.batch_index, self.batch_slice)

        def iter_update():
            for i in xrange(self.nb_batches):
                iter_update_batch(i)

            loss_train = self.get_loss(train.X.get_value())
            loss_valid = self.get_loss(valid.X.get_value())
            loss_test = self.get_loss(test.X.get_value())
            return {"train": loss_train, "valid": loss_valid, "test": loss_test}

        def quitter(update_status):
            return False
        
        def monitor(update_status):
            return update_status

        def observer(monitor_output):
            for k, v in monitor_output.items():
                print("%s : %f" % (k, v))
        
        lasagne.easy.main_loop(self.nb_epochs, iter_update, quitter, monitor, observer)
        

        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt

        # reconstruction


        shape = (28, 28)

        x_rec = self.get_reconstruction(train.X.get_value())
        nb = 10
        k = 1
        plt.axis('off')
        for i in xrange(nb):

            ind = i
            x_i = train.X.get_value()[ind].reshape(shape)
            x_rec_i = x_rec[ind].reshape(shape)

            plt.subplot(nb, 2, k)
            plt.imshow(x_i, cmap='gray')
            k += 1
            plt.subplot(nb, 2, k)
            plt.imshow(x_rec_i, cmap='gray')
            k += 1

        plt.savefig("mnist-rec.png")

        plt.clf()

        # generation
        from pylearn2.scripts.plot_weights import grid_plot
        x_gen = self.generate(train.X.get_value(), 30)
        x_gen = x_gen.reshape( (x_gen.shape[0], shape[0], shape[1])  )
        grid_plot(x_gen, imshow_options={"cmap": "gray"})

        plt.savefig("mnist-gen.png")
        
        fd = open("model.pkl", "w")
        self.save(fd)
        fd.close()


if __name__ == "__main__":
    n = Normal()
    n.run()
