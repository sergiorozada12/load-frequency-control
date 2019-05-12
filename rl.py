import numpy as np


class ExperienceBuffer:
    """ Buffer to implement Experience Replay

        Methods:
            add: Add element to the buffer
            sample: Sample a group of elements from the buffer.
    """

    def __init__(self, buffer_size, batch_size, trace_length, n_var):
        """Constructor of Node class.

            Args:
                buffer_size (int): Size of the buffer.
                batch_size (int): Size of the batch to train
                trace_length (int): Size of the trace used to feed the LSTM
                n_var (int): Number of variables per experience
        """

        self.buffer = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.trace_length = trace_length
        self.n_var = n_var
    
    def add(self, experience):

        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1+len(self.buffer))-self.buffer_size] = []

        self.buffer.append(experience)
            
    def sample(self):

        index = np.random.choice(np.arange(len(self.buffer)), self.batch_size)
        sampled_episodes = [self.buffer[i] for i in index]
        sampled_traces = []

        for episode in sampled_episodes:
            point = np.random.randint(0, episode.shape[0]+1-self.trace_length)
            sampled_traces.append(episode[point:point+self.trace_length, :])

        return np.reshape(np.array(sampled_traces), [-1, self.n_var])


def get_new_epsilon(epsilon):
    """ Decay of epsilon over time.
        Args:
            epsilon (float): current epsilon.

        Returns:
            epsilon (float): new epsilon.
    """

    if epsilon < 0.5:
        return epsilon*0.99999

    return epsilon*0.999999

  
def get_reward(delta_f, z1, z2, e_f=.05, e_z=.2):
    """" Get reward from two agents.

        Args:
            delta_f (float): current deviance from network frequency set point.
            z1 (float): current control action of agent 1.
            z2 (float): current control action of agent 2.
            e_f (float): maximum error admitted in frequency dimension.
            e_z (float): maximum error admitted in cost dimension.

        Returns:
            epsilon (float): new epsilon.
    """
    if (z1 < .5) | (z2 < .5) | (z1 > 4.5) | (z2 > 4.5):
        return 0

    if (np.abs(delta_f) < e_f) & (np.abs(z1-(z2/2)) < e_z):
        return 200

    elif (np.abs(delta_f) < e_f) | (np.abs(z1 - (z2 / 2)) < e_z):
        return 100

    return 0
