"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
PY GAME

https://pygame-learning-environment.readthedocs.io/en/latest/user/games/pong.html

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import os
from os.path import exists

import tensorflow as tf

import ple
from ple import PLE

from ple.games.pong import Pong as pong_game
from ple.games import base
from pygame.constants import K_w, K_s, K_h

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
None
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(physical_devices)
print(config)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Variables
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
actions = { "up": K_w, "none": K_h, "down": K_s }
nb_frames = 100000000000
global reward
reward = 0.0
global step
step = 0
global gamescores
gamescores = 0.0
globalstep = 0
game_global_step = 0
global action
action = K_w

INPUT_DIMS = (1, 1, 30)
learning_rate = 0.000001
batch_size= 10
momentum = 0.1

global gameState
gameState = {'player_y': 96.0, 'player_velocity': 0, 'cpu_y': 96.0, 'ball_x': 128.0, 'ball_y': 96.0, 'ball_velocity_x': 144.0, 'ball_velocity_y': -144.0}

global DATA
DATA = tf.zeros([1, 1, 1, 1, 30], dtype=tf.float32)
global LABEL
LABEL = tf.zeros([1, 1, 1, 1, 1], dtype=tf.float32)

### Mixed of data input
for i in range(15):
	DATA_row = tf.zeros([ 1, 1, 1, 1, 30 ], dtype=tf.float32)	
	DATA = tf.experimental.numpy.vstack([DATA, DATA_row])
	LABEL = tf.experimental.numpy.vstack([LABEL, tf.constant(0, shape=(1, 1, 1, 1, 1))])
	
for i in range(15):
	DATA_row = tf.zeros([ 1, 1, 1, 1, 30 ], dtype=tf.float32)
	DATA = tf.experimental.numpy.vstack([DATA, DATA_row])
	LABEL = tf.experimental.numpy.vstack([LABEL, tf.constant(9, shape=(1, 1, 1, 1, 1))])

checkpoint_path = "F:\\models\\checkpoint\\" + os.path.basename(__file__).split('.')[0] + "\\TF_DataSets_01.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

if not exists(checkpoint_dir) : 
	os.mkdir(checkpoint_dir)
	print("Create directory: " + checkpoint_dir)
	
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Environment
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
game_console = pong_game(width=4 * 64, height=4 * 48, MAX_SCORE=11)
p = PLE(game_console, fps=30, display_screen=True)
p.init()

obs = p.getScreenRGB()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Class / Functions
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def	read_current_sate( string_gamestate ):
	gameState = p.getGameState()

	if string_gamestate in ['player_y', 'player_velocity', 'cpu_y', 'ball_x', 'ball_y', 'ball_velocity_x', 'ball_velocity_y']:
		return gameState[string_gamestate]
	# elif string_gamestate == 'contrast_velocity':
		# return gameState['player_velocity'] - gameState['ball_velocity_y']
	else:
		return None
		
	return None

def update_DATA( action ):
	global reward
	global step
	global gamescores
	global DATA
	global LABEL
	
	player_y_value = read_current_sate('player_y')
	player_velocity_value = read_current_sate('player_velocity')
	cpu_y_value = read_current_sate('cpu_y')
	ball_x_value = read_current_sate('ball_x')
	ball_y_value = read_current_sate('ball_y')
	ball_velocity_x_value = read_current_sate('ball_velocity_x')
	ball_velocity_y_value = read_current_sate('ball_velocity_y')
	
	step = step + 1
	
	print( "step: " + str( int(step) ).zfill(6) + " action: " + str( int(action) ).zfill(6) + " player_y_value: " + str( int(player_y_value) ).zfill(6) + " ball_y_value: " + str( int(ball_y_value) ).zfill(6) )
	
	contrl = reward
	coff_0 = player_y_value
	coff_1 = ball_y_value
	coff_2 = player_y_value - ball_x_value
	coff_3 = 1
	coff_4 = 1
	coff_5 = 1
	coff_6 = 1
	coff_7 = 1
	coff_8 = 1
	
	coeff_row = tf.constant( [ contrl, coff_0, coff_1, coff_2, coff_3, coff_4, coff_5, coff_6, coff_7, coff_8, 
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ], shape=( 30, 1 ), dtype=tf.float32 )
	
	DATA_row = tf.reshape( coeff_row, shape=(1, 1, 1, 1, 30) )
	
	DATA = tf.experimental.numpy.vstack([DATA, DATA_row])
	DATA = DATA[-30:,:,:,:]
	
	LABEL = tf.experimental.numpy.vstack([LABEL, tf.constant(action, shape=(1, 1, 1, 1, 1))])
	LABEL = LABEL[-30:,:,:,:]
	
	DATA = DATA[-30:,:,:,:]
	LABEL = LABEL[-30:,:,:,:]
	
	return DATA, LABEL, step
	
def predict_action( ):

	global DATA
	
	predictions = model.predict(tf.reshape(DATA, shape=( 30, 1, 1, 30 )))
	score = tf.nn.softmax(predictions[0])

	return int(tf.math.argmax(score))
	
def random_action( ): 

	player_y_value = read_current_sate('player_y')
	player_velocity_value = read_current_sate('player_velocity')
	cpu_y_value = read_current_sate('cpu_y')
	ball_x_value = read_current_sate('ball_x')
	ball_y_value = read_current_sate('ball_y')
	ball_velocity_x_value = read_current_sate('ball_velocity_x')
	ball_velocity_y_value = read_current_sate('ball_velocity_y')
	
	# 1 left
	# 2 hold
	# 3 right
	
	player_y_value - ball_y_value
	ball_y_value - player_y_value
	
	coeff_01 = player_y_value - ball_y_value
	coeff_02 = 5
	coeff_03 = ball_y_value - player_y_value
	
	temp = tf.constant( [ coeff_01, coeff_02, coeff_03 ], shape=( 1, 3 ) )
	temp = tf.cast( temp, dtype=tf.int32 )
	
	action = tf.math.argmax( temp[0] ).numpy()

	return action
	
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Callback
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class custom_callback(tf.keras.callbacks.Callback):

	def __init__(self, patience=0):
		self.best_weights = None
		self.best = 999999999999999
		self.patience = patience
	
	def on_train_begin(self, logs={}):
		self.best = 999999999999999
		self.wait = 0
		self.stopped_epoch = 0

	def on_epoch_end(self, epoch, logs={}):
		if(logs['accuracy'] == None) : 
			pass
		
		if logs['loss'] < self.best :
			self.best = logs['loss']
			self.wait = 0
			self.best_weights = self.model.get_weights()
		else :
			self.wait += 1
			if self.wait >= self.patience:
				self.stopped_epoch = epoch
				self.model.stop_training = True
				print("Restoring model weights from the end of the best epoch.")
				self.model.set_weights(self.best_weights)
		
		# if logs['loss'] <= 0.2 and self.wait > self.patience :
		if self.wait > self.patience :
			self.model.stop_training = True

custom_callback = custom_callback(patience=8)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: DataSet
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
dataset = tf.data.Dataset.from_tensor_slices((DATA, LABEL))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Model Initialize
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
model = tf.keras.models.Sequential([
	tf.keras.layers.InputLayer(input_shape=INPUT_DIMS),
	
	tf.keras.layers.Reshape((1, 30)),
	
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, return_state=False)),
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))

])
		
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(192))
model.add(tf.keras.layers.Dense(3))
model.summary()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Optimizer
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
optimizer = tf.keras.optimizers.SGD(
    learning_rate=learning_rate,
    momentum=momentum,
    nesterov=False,
    name='SGD',
)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Loss Fn
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""								
lossfn = tf.keras.losses.MeanSquaredLogarithmicError(reduction=tf.keras.losses.Reduction.AUTO, name='mean_squared_logarithmic_error')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Model Summary
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
model.compile(optimizer=optimizer, loss=lossfn, metrics=['accuracy'])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: FileWriter
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if exists(checkpoint_path) :
	model.load_weights(checkpoint_path)
	print("model load: " + checkpoint_path)
	input("Press Any Key!")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Training
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
history = model.fit(dataset, epochs=1, callbacks=[custom_callback])
model.save_weights(checkpoint_path)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Tasks
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
for i in range(nb_frames):
	game_global_step = game_global_step + 1
	gamescores = gamescores + reward
	reward = 0
	
	DATA, LABEL, step = update_DATA( action )
	
	if p.game_over():
		reward = 0
		done = False
		game_global_step = 0
		gamescores = 0
		step = 0
		
		p.init()
		p.reset_game()

	action = predict_action( )
	reward = p.act(list(actions.values())[action])
	
	if reward < 0 :
		step = 0
	
	if ( reward != 0 and step > -1  ):
		dataset = tf.data.Dataset.from_tensor_slices((DATA, LABEL))
		history = model.fit(dataset, epochs=5, batch_size=batch_size, callbacks=[custom_callback])
		model.save_weights(checkpoint_path)
