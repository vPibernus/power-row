"""
Contient un agent capable d'apprendre à jouer et de faire une partie.
"""
import numpy as np
from keras import Model
from keras.layers import Input, Conv2D, Flatten, Dense
from tqdm import tqdm
from grid import Grid


class Agent:
    """Un agent capable d'apprendre et de jouer au puissance 4."""
    def __init__(self, grid=Grid(), piece='x'):
        self.piece = piece
        self.rows = grid.rows
        self.cols = grid.cols

        # Initialize the model
        self.initialize_model()

    def initialize_model(self):
        # Modèle pour entraîner l'agent
        input_model = Input(shape=(self.rows, self.cols, 3))
        x = Conv2D(
            filters=16, kernel_size=(3, 3), activation="relu")(input_model)
        x = Conv2D(
            filters=16, kernel_size=(3, 3), activation="relu")(x)
        x = Flatten()(x)
        x = Dense(16, activation='relu')(x)
        output_model = Dense(self.cols, activation='linear')(x)

        model = Model(input_model, output_model, name="brain")
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])
        self.model = model

    def encode_grid(self, grid):
        # Encode un array de char en un array de 0 et 1 pour l'entraînement.
        encoded_grid = np.zeros((self.rows, self.cols, 3))
        encoded_grid[grid == ' '] = 1
        encoded_grid[grid == self.piece, 1] = 1
        encoded_grid[~encoded_grid.sum(axis=2).astype(bool), 2] = 1

        return encoded_grid[np.newaxis, :]

    def train_model(self, num_games=100, discount_factor=0.95,
                    eps=0.5, eps_decay_factor=0.999, verbose=False,
                    dict_reward={'-1': -10, '0': 0, '1': 10, '2': 0}):

        for ii in tqdm(range(num_games), desc="Training"):
            grid = Grid()
            state = self.encode_grid(grid.grid)
            eps *= eps_decay_factor
            done = grid.done
            while not done:

                qscore = self.model.predict(state)

                if np.random.random() < eps:
                    col = np.random.choice(np.arange(self.cols))
                else:
                    col = np.argmax(qscore)

                event = grid.make_move(col, self.piece, verbose=verbose)
                reward = dict_reward[str(event)]

                if (event == -1):  # Forbiden move
                    new_state = state
                    new_qscore = qscore
                    reward_op = 0

                elif (event == 1) or (event == 2):  # End of the game
                    new_state = state
                    new_qscore = 0
                    reward_op = 0

                else:  # Not done, plat the best move for opponent
                    col_op = grid.get_available_moves()
                    reward_op = []
                    for c_op in col_op:
                        event_op = grid.make_move(c_op, 'o')
                        reward_op.append(dict_reward[str(event_op)])
                        grid.cancel_last_move()
                    mask_col_op = np.array(reward_op) == np.max(reward_op)
                    col_op = np.random.choice(np.array(col_op)[mask_col_op])
                    event_op = grid.make_move(col_op, 'o', verbose=verbose)
                    reward_op = dict_reward[str(event_op)]

                    new_state = self.encode_grid(grid.grid)
                    new_qscore = self.model.predict(new_state)

                # Définition de la récompense pour ce tour de jeu
                target = reward-reward_op + discount_factor*np.max(new_qscore)
                target_vector = self.model.predict(state)[0]
                target_vector[col] = target
                target_vector = target_vector.reshape(-1, self.cols)

                # Mise à jour du modèle
                self.model.fit(state, target_vector, epochs=1, verbose=0)
                state = new_state
                done = grid.done
            grid.display()
        pass

    def play(self, grid):
        # Encode la grille dans un tableau binaire
        state = self.encode_grid(grid.grid)
        # Calcule le score prédit pour chaque action
        qscore = self.model.predict(state)
        # Jouer l'action qui rapporte le meilleur score
        cols = np.argmax(qscore)
        grid.make_move(cols, self.piece, verbose=1)
        return grid


agent = Agent()
agent.train_model(num_games=1000)

# grid = Grid()
# state = agent.encode_grid(grid.grid)
# agent.model.predict(state)
