# -*- coding: utf-8 -*- 
import numpy as np
import tensorflow as tf
import reader
import time
import os

flags = tf.flags
flags.DEFINE_integer("hidden_size",128, "LSTM hidden size 128")
flags.DEFINE_integer("num_layers",1, "LSTM number of layer")

flags.DEFINE_integer("vocab_size",24, "amount of bias")
flags.DEFINE_float("compar",-1.0, "missing value is marked with -1.0")
flags.DEFINE_integer("embedding_size",24, "Dimension of data")
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
		vocab_size = config.vocab_size

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
					comparison = tf.equal( self.input_data[:,time_step,:], tf.constant( FLAGS.compar ) )
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
		comparison2=tf.equal(targets, tf.constant( FLAGS.compar ) )
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
	keep_prob = 1.0				
	batch_size =4 				
	vocab_size = FLAGS.vocab_size		
	embedding_size=FLAGS.embedding_size	

class TestConfig(object):	
	"""Test config."""
	init_scale = 0.1
	num_layers = FLAGS.num_layers
	num_steps = 12
	hidden_size = FLAGS.hidden_size
	embedding_size=FLAGS.embedding_size
	keep_prob = 1.0
	batch_size = 1
	vocab_size = FLAGS.vocab_size

def run_epoch(session, model,data, eval_op):
	costs = 0.0
	pre=[]
	for step,(x,y) in enumerate(reader.ptb_iterator(data, model.batch_size,model.num_steps,FLAGS.embedding_size)):
		_, prediction= session.run([eval_op,model.prediction],
                                 {model.input_data: x,
                                  model.targets: y
				  })
		for i in xrange(model.batch_size):
			costs+=np.sqrt(np.mean(np.square(prediction[(i+1)*model.num_steps-1,:]-y[i,-1,:])))
		for i in xrange(model.batch_size):
			pre.append(prediction[(i+1)*model.num_steps-1,:])
	return costs,pre

def get_config(flag):
	if flag == "Train":
		return TrainConfig()
	elif flag == "Test":
		return TestConfig()
def main(_):

	# get config
  	config = get_config('Train')
  	eval_config = get_config('Test')
	
	raw = reader.read_data_asmatrix_minmax("data/raw.txt",FLAGS.embedding_size)
	miss_data = reader.read_data_asmatrix("data/miss_data.txt",FLAGS.embedding_size)
	
	fil = test_data = miss_data 
	train_data = miss_data
	
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	gpu_config = tf.ConfigProto()
	gpu_config.gpu_options.allow_growth = True

	with tf.Session(config=gpu_config) as session:
		initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)  

		with tf.variable_scope("model", reuse=None,initializer=initializer):
			m = LIMELSTM(is_training=True,config=config,FLAGS=FLAGS)   	        #train model,is_trainable=True
		with tf.variable_scope("model", reuse=True,initializer=initializer):
			mtest = LIMELSTM(is_training=False,config=eval_config,FLAGS=FLAGS)      #test model,is_trainable=False

		tf.initialize_all_variables().run()   
		m._lr=0.01
		new_lr=m._lr
		start_time = time.time()
		saver = tf.train.Saver(max_to_keep=0)
		for i in xrange(150):
			print 'Number of iterations:',i
			_,pre=run_epoch(session,m,train_data,m.train_op)			
			if i>=10 and i%10==0:
				if new_lr>0.005:new_lr=new_lr-0.003
				else:new_lr=new_lr*0.5
				m.assign_lr(session,new_lr)
		_,prediction=run_epoch(session, mtest,test_data,tf.no_op())
		RMSE=reader.count_RMSE_matrix(raw,fil,np.reshape(np.array(prediction),[-1,FLAGS.embedding_size]),FLAGS.embedding_size,config.num_steps)
	print 'test RMSE:', RMSE
if __name__ == "__main__":
	tf.app.run() 
