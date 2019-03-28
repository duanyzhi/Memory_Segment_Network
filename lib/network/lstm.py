import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from lib.config.config import FLAGS as cfg


class lstm:
    def __init__(self):
        self.time_step_size = cfg.Seg
        self.hidden_size = cfg.hidden_size
        self.layer_num = cfg.lstm_layer_num
        self.reuse = False

    def __call__(self, x):  # x: [batch_size, time_step, input_size]
        lstm_cell = rnn.BasicLSTMCell(num_units=self.hidden_size, forget_bias=1.0, state_is_tuple=True, reuse=self.reuse)
        init_state = lstm_cell.zero_state(cfg.batch_size, dtype=tf.float32)
        outputs = list()
        state = init_state
        with tf.variable_scope('Attention_Model'):
            for timestep in range(cfg.Seg):
                if timestep > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = lstm_cell(x[:, timestep, :], state)
                outputs.append(cell_output)
        self.reuse = True
        h_state = outputs[-1]
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="lstm")
        return h_state
