# work_conditions_rules

Transforms works conditions and rules to AI model training, we study and know the limits of the AI networks learning and its optimizers now we select the proper works for them by simple rules and conditions created from the tasks we are committing to it.

## Sample game conditions and rules ##

The simple rule's conditions for Pong game is a sample for many tasks that required curser movements such as drop-catch or distance approaches. The control ```contrl``` rule is the condition telling to do or negative to do, a robot will do it anyway if still cannot find matching rules better than the conditions. Player ```player_y_value ``` and Ball ```ball_y_value``` is required for the robot to learn from game conditions, you may input something else with better results but the objective is for a robot to learn from the actual game conditions. Some rules we added for curser running are important for shortening AI learning process and will neglect when the robot found the pattern ```player_y_value - ball_x_value``` or waits until the robot figures it out from the conditions you input.

### Conditions Table ###
| rule name |    Condition   | Description |
| ------------- | ------------- | ------------- |
| press K_w | move up  | Player curser move up  |
| press K_s | move down  | Player curser move down  |
| press K_h | move none  | Player curser move none  |
| Control  | Negative ack  | Bad or Good  |

### Sample Codes ###

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

Game's inputs using for the simple tasks to perform gameplay as displayed in the result section.

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

There are two conditions as simple rules for ```press K_w``` for run the curser up and ```K_s``` for run the curder down, we spared ```press K_h``` for do ```none```

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

The result from conditions and rules transform to AI input rules.

#### Training AI networks model ####

![Employee data](https://github.com/jkaewprateep/work_conditions_rules/blob/main/Ping-Pong.gif?raw=true "Employee Data title")

#### Ping-Pong game play ####

![Employee data](https://github.com/jkaewprateep/work_conditions_rules/blob/main/Pong%20Game.gif?raw=true "Employee Data title")

#### Pixel Copter game pay ####

![Employee data](https://github.com/jkaewprateep/work_conditions_rules/blob/main/Pixel%20Copter.gif?raw=true "Employee Data title")
