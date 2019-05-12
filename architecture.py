import tensorflow as tf
import numpy as np


class Actor:

    def __init__(self, num_inputs, h_size, name):
        """Constructor of Actor class.

            Args:
                num_inputs (int): Number of the inputs of the actor
                h_size (int): Size of the LSTM output.
                name (str): name of the context.
        """

        self.num_variables = 10
        self.cell = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)

        # Input
        self.inp = tf.placeholder(shape=[None, num_inputs], dtype=tf.float32)
        self.initializer = tf.contrib.layers.xavier_initializer()
        
        # LSTM
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])
        self.train_length = tf.placeholder(dtype=tf.int32)
        self.rnn_inp = tf.reshape(self.inp, [self.batch_size, self.train_length, num_inputs])
        
        self.state_in = self.cell.zero_state(self.batch_size, tf.float32)
        self.rnn, self.rnn_state = tf.nn.dynamic_rnn(inputs=self.rnn_inp, cell=self.cell,
                                                     dtype=tf.float32, initial_state=self.state_in, scope=name+'_rnn')
        self.rnn = tf.reshape(self.rnn, shape=[-1, h_size])
        
        # MLP
        self.b1 = tf.Variable(self.initializer([1, 1000]))
        self.W1 = tf.Variable(self.initializer([h_size, 1000]))
        self.h1 = tf.nn.relu(tf.matmul(self.rnn, self.W1) + self.b1)

        self.b2 = tf.Variable(self.initializer([1, 100]))
        self.W2 = tf.Variable(self.initializer([1000, 100]))
        self.h2 = tf.nn.relu(tf.matmul(self.h1, self.W2) + self.b2)

        self.b3 = tf.Variable(self.initializer([1, 50]))
        self.W3 = tf.Variable(self.initializer([100, 50]))
        self.h3 = tf.nn.relu(tf.matmul(self.h2, self.W3) + self.b3)

        self.b4 = tf.Variable(self.initializer([1, 1]))
        self.W4 = tf.Variable(self.initializer([50, 1]))
        self.a_unscaled = tf.nn.tanh(tf.matmul(self.h3, self.W4)+self.b4)
        self.a = tf.multiply(self.a_unscaled, .05)

        # Gradients
        self.network_params = tf.trainable_variables()[-self.num_variables:]
        self.critic_gradient = tf.placeholder(tf.float32, [None, 1])
        self.unnormalized_actor_gradients = tf.gradients(self.a, self.network_params, - self.critic_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, 32), self.unnormalized_actor_gradients))
        
        # Optimization
        self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.upd = self.optimizer.apply_gradients(zip(self.actor_gradients, self.network_params))

        self.update_parameters = []

    def create_op_holder(self, params, tau):
        """ Use target network op holder if needed"""
        self.update_parameters = [self.network_params[i].assign(tf.multiply(params[i], tau) +
                                                                tf.multiply(self.network_params[i], 1. - tau))
                                  for i in range(len(self.network_params))]


class Critic:

    def __init__(self, num_inputs, h_size, name):
        """Constructor of Critic class.

            Args:
                num_inputs (int): Number of the inputs of the critic
                h_size (int): Size of the LSTM output.
                name (str): name of the context.
        """

        self.num_variables = 10
        self.cell = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)

        # Input
        self.inp = tf.placeholder(shape=[None, num_inputs], dtype=tf.float32)
        self.initializer = tf.contrib.layers.xavier_initializer()
        
        # LSTM
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])
        self.train_length = tf.placeholder(dtype=tf.int32)
        self.rnn_inp = tf.reshape(self.inp, [self.batch_size, self.train_length, num_inputs])
        
        self.state_in = self.cell.zero_state(self.batch_size, tf.float32)
        self.rnn, self.rnn_state = tf.nn.dynamic_rnn(inputs=self.rnn_inp, cell=self.cell,
                                                     dtype=tf.float32, initial_state=self.state_in, scope=name+'_rnn')
        self.rnn = tf.reshape(self.rnn, shape=[-1, h_size])
        
        # MLP
        self.b1 = tf.Variable(self.initializer([1, 1000]))
        self.W1 = tf.Variable(self.initializer([h_size, 1000]))
        self.h1 = tf.nn.relu(tf.matmul(self.rnn, self.W1) + self.b1)

        self.b2 = tf.Variable(self.initializer([1, 100]))
        self.W2 = tf.Variable(self.initializer([1000, 100]))
        self.h2 = tf.nn.relu(tf.matmul(self.h1, self.W2) + self.b2)

        self.b3 = tf.Variable(self.initializer([1, 50]))
        self.W3 = tf.Variable(self.initializer([100, 50]))
        self.h3 = tf.nn.relu(tf.matmul(self.h2, self.W3) + self.b3)

        self.b4 = tf.Variable(self.initializer([1, 1]))
        self.W4 = tf.Variable(self.initializer([50, 1]))
        self.q = tf.matmul(self.h3, self.W4) + self.b4
        
        # Gradients
        self.network_params = tf.trainable_variables()[-self.num_variables:]
        self.target_q = tf.placeholder(tf.float32, [None, 1])
        
        # Optimization
        self.loss = tf.reduce_mean(tf.square(self.target_q-self.q))
        self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.upd = self.optimizer.minimize(self.loss)
        
        # Gradients
        self.critic_gradients = tf.gradients(self.q, self.inp)

        self.update_parameters = []

    def create_op_holder(self, params, tau):
        """ Use target network op holder if needed"""
        self.update_parameters = [self.network_params[i].assign(tf.multiply(params[i], tau) +
                                                                tf.multiply(self.network_params[i], 1. - tau))
                                  for i in range(len(self.network_params))]


class Agent:

    def __init__(self, a_dof, c_dof, h_size, name, batch_size, train_length, tau):
        """Constructor of Agent class. Each agent is composed of the main actor-critic pair and the target actor-critic
        pair. Target architectures help stabilizing the training of the agents.

                    Args:
                        h_size (int): Size of the LSTM output.
                        name (str): name of the context.
                """

        self.actor = Actor(a_dof, h_size, name+'_actor')
        self.critic = Critic(c_dof, h_size, name+'_critic')
        self.actor_target = Actor(a_dof, h_size, name+'_actor_target')
        self.critic_target = Critic(c_dof, h_size, name+'_critic_target')

        self.batch_size = batch_size
        self.train_length = train_length
        self.state_train = (np.zeros([self.batch_size, h_size]), np.zeros([self.batch_size, h_size]))

        self.actor_target.create_op_holder(self.actor.network_params, tau)
        self.critic_target.create_op_holder(self.critic.network_params, tau)

    def a_actor_operation(self, session, inp, state):

        return session.run([self.actor.a, self.actor.rnn_state],
                           feed_dict={self.actor.inp: inp,
                                      self.actor.state_in: state,
                                      self.actor.batch_size: 1,
                                      self.actor.train_length: 1})

    def a_actor_training(self, session, inp):

        return session.run(self.actor.a, feed_dict={self.actor.inp: inp,
                                                    self.actor.state_in: self.state_train,
                                                    self.actor.batch_size: self.batch_size,
                                                    self.actor.train_length: self.train_length})

    def a_target_actor_training(self, session, inp):

        return session.run(self.actor_target.a, feed_dict={self.actor_target.inp: inp,
                                                           self.actor_target.state_in: self.state_train,
                                                           self.actor_target.batch_size: self.batch_size,
                                                           self.actor_target.train_length: self.train_length})

    def q_target_critic(self, session, inp):

        return session.run(self.critic_target.q, feed_dict={self.critic_target.inp: inp,
                                                            self.critic_target.train_length: self.train_length,
                                                            self.critic_target.batch_size: self.batch_size,
                                                            self.critic_target.state_in: self.state_train})

    def gradients_critic(self, session, inp):

        return session.run(self.critic.critic_gradients, feed_dict={self.critic.inp: inp,
                                                                    self.critic.train_length: self.train_length,
                                                                    self.critic.batch_size: self.batch_size,
                                                                    self.critic.state_in: self.state_train})

    def update_critic(self, session, inp, q):

        session.run(self.critic.upd, feed_dict={self.critic.inp: inp,
                                                self.critic.target_q: q,
                                                self.critic.train_length: self.train_length,
                                                self.critic.batch_size: self.batch_size,
                                                self.critic.state_in: self.state_train})

    def update_actor(self, session, inp, gradients):

        session.run(self.actor.upd, feed_dict={self.actor.inp: inp,
                                               self.actor.state_in: self.state_train,
                                               self.actor.critic_gradient: gradients,
                                               self.actor.batch_size: self.batch_size,
                                               self.actor.train_length: self.train_length})

    def update_targets(self, session):

        session.run(self.actor_target.update_parameters)
        session.run(self.critic_target.update_parameters)

    def initialize_targets(self, session):

        session.run([self.actor_target.network_params[i].assign(self.actor.network_params[i])
                     for i in range(len(self.actor.network_params))])

        session.run([self.critic_target.network_params[i].assign(self.critic.network_params[i])
                     for i in range(len(self.critic.network_params))])

    def importance(self, session, inp_q, inp_q_target, r, gamma, state, new_state):

        act_q = session.run(self.critic_target.q, feed_dict={self.critic_target.inp: inp_q,
                                                             self.critic_target.train_length: 1,
                                                             self.critic_target.batch_size: 1,
                                                             self.critic_target.state_in: state})[0, 0]

        tar_q = session.run(self.critic_target.q, feed_dict={self.critic_target.inp: inp_q_target,
                                                             self.critic_target.train_length: 1,
                                                             self.critic_target.batch_size: 1,
                                                             self.critic_target.state_in: new_state})[0, 0]

        return np.abs(r + gamma*tar_q - act_q)
