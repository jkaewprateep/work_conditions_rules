# work_conditions_rules
Transforms works conditions and rules to AI model training.

## Sample game conditions and rules ##

```
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
	
	print( "step: " + str( int(step) ).zfill(6) + " action: " + str( int(action) ).zfill(6) + 
          " player_y_value: " + str( int(player_y_value) ).zfill(6) + " ball_y_value: " + str( int(ball_y_value) ).zfill(6) )
	
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
```

## Running output ##

```
step: 000044 action: 000000 player_y_value: 000014 ball_y_value: 000046
step: 000045 action: 000000 player_y_value: 000014 ball_y_value: 000045
step: 000046 action: 000000 player_y_value: 000014 ball_y_value: 000044
step: 000047 action: 000000 player_y_value: 000014 ball_y_value: 000043
step: 000048 action: 000000 player_y_value: 000014 ball_y_value: 000041
step: 000049 action: 000000 player_y_value: 000014 ball_y_value: 000040
step: 000050 action: 000000 player_y_value: 000014 ball_y_value: 000039
step: 000051 action: 000000 player_y_value: 000014 ball_y_value: 000038
```

## Random Function ##

```
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
```

## Files and Directory ##

| File name | Description |
| ------------- | ------------- |
| sample.py | sample codes  |
| Ping-Pong.gif | training time  |
| Pong Game.gif | gameplay  |
| Pixel Copter.gif  | Further application  |
| README.md  | readme file |


## Result ##

#### Training AI networks model ####

![Employee data](https://github.com/jkaewprateep/work_conditions_rules/blob/main/Ping-Pong.gif?raw=true "Employee Data title")

#### Ping-Pong game play ####

![Employee data](https://github.com/jkaewprateep/work_conditions_rules/blob/main/Pong%20Game.gif?raw=true "Employee Data title")

#### Pixel Copter game pay ####

![Employee data](https://github.com/jkaewprateep/work_conditions_rules/blob/main/Pixel%20Copter.gif?raw=true "Employee Data title")
