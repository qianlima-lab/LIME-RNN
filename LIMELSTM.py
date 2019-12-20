# -*- coding: utf-8 -*- 
import numpy as np
import tensorflow as tf
import reader
import os

flags = tf.flags
flags.DEFINE_integer("num_layers",1, "LSTM number of layer")
flags.DEFINE_integer("hidden_size",128, "LSTM hidden size")

flags.DEFINE_integer("embedding_size",24, "Dimension of data")
flags.DEFINE_float("missing_flag",-1.0, "missing value is marked with -1.0")

FLAGS = flags.FLAGS

variables_dict = {"W_imp": tf.Variable(tf.zeros([FLAGS.hidden_size,FLAGS.embedding_size]),
        			    name="W_imp"),
    		  "bias": tf.Variable(tf.zeros([FLAGS.embedding_size]), name="bias"),
			  #"W_r":tf.Variable(tf.zeros([FLAGS.hidden_size, FLAGS.hidden_size]), name="W_r"),
		      "W_r":tf.Variable(tf.zeros([FLAGS.hidden_size],tf.float32), name="W_r")  ### Consider diagonal W_r, reducing the number of parameters for small data set (also used in NIPS2016  
																				       ###                         "Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction")
		  }

class LIMELSTM(object):    
	def __init__(self, is_training,config,FLAGS):
		self.batch_size = batch_size = config.batch_size
		self.num_steps = num_steps = config.num_steps
		size = config.hidden_size
		embedding_size = config.embedding_size

		self.input_data = tf.placeholder(tf.float32, [batch_size,num_steps,embedding_size])    # input
		self.targets = tf.placeholder(tf.float32, [batch_size,num_steps,embedding_size])       # output
		lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
		cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers) 
		self.initial_state = cell.zero_state(batch_size, dtype=tf.float32)	

		state = self.initial_state 
		self.F=[]

		with tf.variable_scope("RNN"):	
			for time_step in xrange(num_steps):
				if time_step > 0 : tf.get_variable_scope().reuse_variables() 	
				if time_step ==0 :
					(cell_output, state) = cell(self.input_data[:,time_step,:],state)
					self.F.append(cell_output)
				else:	
					comparison = tf.equal( self.input_data[:,time_step,:], tf.constant( FLAGS.missing_flag ) )
					temp2=tf.matmul(self.F[time_step-1], variables_dict["W_imp"]) + variables_dict["bias"]
					#change the input
					input2=tf.where(comparison, temp2, self.input_data[:,time_step,:])
					(cell_output, state) = cell(input2, state)
					#self.F.append(cell_output+tf.matmul(self.F[time_step-1],variables_dict["W_r"]))
					self.F.append(cell_output+tf.multiply(self.F[time_step-1],variables_dict["W_r"])) ### Consider diagonal W_r for small data set

        #unfolded F into the [batch, hidden_size * num_steps], and then reshape it into [batch * numsteps, hidden_size]
		F_out = tf.reshape(tf.concat(self.F, 1), [-1, size])
		self.prediction = tf.matmul(F_out, variables_dict["W_imp"]) + variables_dict["bias"]
		targets=tf.reshape(self.targets,[-1,embedding_size])

		#change the cost function
		comparison2=tf.equal(targets, tf.constant( FLAGS.missing_flag ) )
		targets=tf.where(comparison2,self.prediction,targets)

		self.cost=cost= tf.reduce_mean(tf.square(targets - self.prediction))/(batch_size)
		self.final_state = state

		if not is_training:  # if not training ,return 
			return

		self._lr = tf.Variable(0.0, trainable=False)
		self.train_op = tf.train.AdamOptimizer(self._lr).minimize(cost)
		
		self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")     
		self._lr_update = tf.assign(self._lr, self._new_lr)     
        
	def assign_lr(self, session, lr_value):
		session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

class TrainConfig(object):	
	"""Train config."""
	init_scale = 0.1
	num_layers = FLAGS.num_layers		
	num_steps = 12 				
	hidden_size = FLAGS.hidden_size		
	batch_size = 4
	embedding_size=FLAGS.embedding_size

class TestConfig(object):	
	"""Train config."""
	init_scale = 0.1
	num_layers = FLAGS.num_layers		
	num_steps = 12 				
	hidden_size = FLAGS.hidden_size		
	batch_size = 1
	embedding_size=FLAGS.embedding_size
	
def run_epoch(session, model, data, eval_op):
	header_pre = []
	pre = []
	for step,(x,y) in enumerate(reader.ptb_iterator(data, model.batch_size,model.num_steps,FLAGS.embedding_size)):
		_, prediction= session.run([eval_op, model.prediction],
                                 {model.input_data: x,
                                  model.targets:    y})
		if step == 0:			
			header_pre.append(prediction[:model.num_steps-1,:])
		for i in range(model.batch_size):
			pre.append(prediction[(i+1)*model.num_steps - 1,:])
	
	return np.concatenate((np.array(header_pre).reshape(-1, FLAGS.embedding_size), np.array(pre)), axis=0)

def get_config(flag):
	if flag == "Train":
		return TrainConfig()
	if flag == "Test":
		return TestConfig()
		
def main(_):

	# get config
  	config = get_config('Train')
	test_config = get_config('Test')

	raw_data, columns_name, norlizer = reader.read_raw_data("data/raw.txt")
	miss_data = reader.read_missing_data("data/miss_data.txt", norlizer, FLAGS.embedding_size)
	
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	gpu_config = tf.ConfigProto()
	gpu_config.gpu_options.allow_growth = True

	with tf.Session(config=gpu_config) as session:
		initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)  

		with tf.variable_scope("LIMELSTM", reuse=None, initializer=initializer):
			model = LIMELSTM(is_training=True, config=config, FLAGS=FLAGS)   	        #train model,is_trainable=True
		with tf.variable_scope("LIMELSTM", reuse=True, initializer=initializer):
			test_model = LIMELSTM(is_training=False, config=test_config, FLAGS=FLAGS)   	        #test model,is_trainable=False

		tf.initialize_all_variables().run()
		model._lr = 0.01
		new_lr = model._lr
		for i in xrange(120):
			print 'Number of iterations:',i
			_ = run_epoch(session, model, miss_data, model.train_op)					
			if i>=10 and i%10==0:
				if new_lr>0.005:new_lr=new_lr-0.003
				else:new_lr=new_lr*0.5
				model.assign_lr(session,new_lr)
		prediction = run_epoch(session, test_model, miss_data, tf.no_op())
		print 'RMSE:', reader.RMSE_Metric(raw_data, miss_data, np.concatenate((miss_data[0,:].reshape(1,-1), np.array(prediction)), axis=0), missing_flag=FLAGS.missing_flag)	
if __name__ == "__main__":
	tf.app.run() 
