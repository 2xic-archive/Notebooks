import curses
import time
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.models import load_model
from collections import *


class game():
	def __init__(self):
		self.tiles = {
			"player":"-",
			"packet":"#",
			"space":" "
		}
		#	square game_size will be x and y value
		self.game_size = 9

		self.board = np.zeros((self.game_size, self.game_size))
		self.packet_cordinates = []
		self.player_cordinates = []
		self.score = 0
		
		for x in range(3):
			self.player_cordinates.append(x)
			self.board[self.game_size - 1, x] = ord(self.tiles["player"])
		self.drop_packet()

	def get_state(self):
		status = self.packet_cordinates[1] in self.player_cordinates
		if(status):
			status = 0
		else:
			if(self.player_cordinates[0] > self.packet_cordinates[1]):
				status = -1
			elif(self.player_cordinates[-1] < self.packet_cordinates[1]):
				status = 1
			else:
				raise Exception("error in get_state")
		return np.array([status]).reshape(1, 1)

	def draw(self):
		for y in range(self.game_size):
			for x in range(self.game_size):
				if([y, x] == self.packet_cordinates):
					self.board[y, x] = ord(self.tiles["packet"])
				elif(y == self.game_size - 1 and x in self.player_cordinates):
					self.board[y, x] = ord(self.tiles["player"])
				else:
					self.board[y, x] = ord(self.tiles["space"])

	def drop_packet(self):
		if(len(self.packet_cordinates) == 0):
			self.packet_cordinates = [0, random.randint(0, self.game_size - 1)]
			self.draw()
		else:
			self.packet_cordinates[0] += 1
			if(self.packet_cordinates[1] in self.player_cordinates and self.packet_cordinates[0] == self.game_size ):
				self.score += 1
				self.packet_cordinates = [0, random.randint(0, self.game_size - 1)]


	def lost_game(self):
		return (self.packet_cordinates[0] == self.game_size)

	def get_board(self):
		return self.board

	def move(self, left):
		if(left == 0):
			if(self.player_cordinates[-1] == self.game_size-1):
				return
			for i in range(len(self.player_cordinates)):
				self.player_cordinates[i] += 1
		elif(left == 1):
			if(self.player_cordinates[0] == 0):
				return
			for i in range(len(self.player_cordinates)):
				self.player_cordinates[i] -= 1
		else:
			pass

#	just for debugging
def main(screen):
	screen.timeout(0)

	new_game = game()
	while not new_game.lost_game():
		move = screen.getch()
		if move == curses.KEY_RIGHT: 
			new_game.move(0)
		if move == curses.KEY_LEFT: 
			new_game.move(1)

		new_game.drop_packet()
		new_game.draw()
		for y in range(9):
			for game_y in range(new_game.board.shape[0]):
				string_game = ""
				for game_x in range(new_game.board.shape[0]):
					string_game += chr(int(new_game.board[game_y, game_x]))
				screen.addstr(game_y, 0, string_game)			
			screen.refresh()		
		time.sleep(0.3)

		
class q_agent():
	def __init__(self):
		self.state_size = 1 # player position + packet position
		self.action_size = 3 # left or rigth or stay?
		self.memory = deque(maxlen=2000)

		self.gamma = 0.95    
		self.epsilon = 1.0 
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.learning_rate = 0.001

		self.model = self.make_model()

	def do_action(self, state, training=True):
		if np.random.rand() <= self.epsilon and training:
			return np.random.random_integers(0, self.action_size - 1), True

		action = self.model.predict(state)
		return np.argmax(action[0]), False

	def make_model(self):
		model = Sequential()
		model.add(Dense(24, input_dim=self.state_size, activation='relu'))
		model.add(Dense(24, activation='relu'))
		model.add(Dense(self.action_size, activation='linear'))
		model.compile(loss='mse',
					  optimizer=Adam(lr=self.learning_rate))
		return model

	def replay(self, batch_size):
		minibatch = self.memory
		if(len(minibatch) >= batch_size):
			minibatch = random.sample(self.memory, batch_size)

		for state, action, reward, next_state, done in minibatch:
			target = reward
			if not done:
				target = reward + self.gamma * \
					   np.amax(self.model.predict(next_state)[0])
			target_f = self.model.predict(state)
			target_f[0][action] = target
			self.model.fit(state, target_f, epochs=1, verbose=0)
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

def train_model():
	from copy import deepcopy
	agent = q_agent()
	import time
	average_100 = []
	for i in range(50000):
		new_game = game()
		state = deepcopy(new_game.get_state())

		while not new_game.lost_game():
			action, random = agent.do_action(state)

			new_game.move(action)
			new_game.drop_packet()

			reward = new_game.score
			done = new_game.lost_game()
			next_state =  deepcopy(new_game.get_state())
			
			agent.memory.append((state, action, reward, next_state, done))

			if not done:
				new_game.draw()

			state = next_state

			if(reward == 100):
				break

		if(len(average_100) < 100):
			average_100.append(reward)
		else:
			average_100 = average_100[1:] + [reward]

		if(i % 100 == 0):	
			print("Avreage score {} {}".format(sum(average_100)/100, i) )
			print("")

		agent.replay(128)
		if(i % 1000 == 0 and i > 0):
			agent.model.save('my_model_{}'.format(i))

def model_play(screen):
	import time
	from copy import deepcopy

	screen.timeout(0)

	agent = q_agent()
	agent.model = load_model("my_model_49000")
	average_100 = []
	
	new_game = game()
	state = deepcopy(new_game.get_state())
	screen.clear()

	while not new_game.lost_game():
		action, random = agent.do_action(state, training=False)
		new_game.move(action)
		new_game.drop_packet()

		reward = new_game.score
		doen = new_game.lost_game()
		state =  deepcopy(new_game.get_state())

		new_game.draw()
		for y in range(9):
			for game_y in range(new_game.board.shape[0]):
				string_game = ""
				for game_x in range(new_game.board.shape[0]):
					string_game += chr(int(new_game.board[game_y, game_x]))
				screen.addstr(game_y, 0, string_game)
		
		screen.addstr(9 + 1, 0, str(new_game.score))			
		screen.refresh()	
		time.sleep(0.1)
	print(reward)

if __name__=='__main__':
	#curses.wrapper(main)
	#train_model()
	curses.wrapper(model_play)
