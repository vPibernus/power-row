"""
Contient la grille de jeu et fixe les règles pour jouer et gagner.
"""
import numpy as np


class Grid:
    def __init__(self, rows=6, cols=7, n_in_a_row=4):
        self.rows = rows
        self.cols = cols
        self.n_in_a_row = n_in_a_row
        self.count_piece = 0
        self.grid = np.array([[' ' for _ in range(cols)] for _ in range(rows)])
        self.last_move = []
        self.done = False

    def display(self):
        for row in self.grid:
            print('|' + '|'.join(row) + '|')
            print('|' + '-' * (self.cols * 2 - 1) + '|')

    def drop_piece(self, col, piece):
        """
        Rajoute une pièce à la grille
        """
        row = self.get_next_open_row(col)
        self.grid[row, col] = piece
        self.count_piece += 1
        self.last_move.append(col)
        pass

    def is_valid_location(self, col):
        # Vérifier si la colonne spécifiée est valide pour ajouter une pièce
        if ' ' in self.grid[:, col]:
            valid = True
        else:
            valid = False
        return valid

    def get_next_open_row(self, col):
        # Obtenir la prochaine ligne ouverte dans la colonne spécifiée
        try:
            row = np.arange(self.rows)[self.grid[:, col] == ' '].max()
        except ValueError:  # Pas de ligne ouverte dans la colonne
            row = -1
        return row

    def n_consecutive(self, liste, piece):
        # Calcule le nombre de pièce consécutive dans la liste
        n_consec = 0
        n = 0
        for p in liste:
            if p == piece:
                n += 1
                n_consec = max(n_consec, n)
            else:
                n = 0
        return n_consec

    def is_winner(self, piece):
        # Vérifier s'il y a un gagnant
        victory = False

        # Par colonne
        for col in range(self.cols):
            if victory:
                break
            else:
                n_piece = self.n_consecutive(self.grid[:, col], piece)
                if n_piece >= self.n_in_a_row:
                    victory = True

        # Par ligne
        for row in range(self.rows):
            if victory:
                break
            else:
                n_piece = self.n_consecutive(self.grid[row, :], piece)
                if n_piece >= self.n_in_a_row:
                    victory = True

        # Par diagonale 1
        for ii in range(1-self.rows, self.cols):
            if victory:
                break
            else:
                n_piece = self.n_consecutive(
                    np.diagonal(self.grid, offset=ii), piece)
                if n_piece >= self.n_in_a_row:
                    victory = True

        # Par diagonale 2
        for ii in range(1-self.rows, self.cols):
            if victory:
                break
            else:
                n_piece = self.n_consecutive(
                    np.diagonal(self.grid[:, ::-1], offset=ii), piece)
                if n_piece >= self.n_in_a_row:
                    victory = True

        return victory

    def is_full(self):
        # Vérifier si la grille est pleine
        if self.count_piece == self.rows*self.cols:
            full = True
        else:
            full = False
        return full

    def get_available_moves(self):
        # Obtenir les mouvements disponibles
        available_moves = []
        for col in np.arange(self.cols):
            if self.is_valid_location(col):
                available_moves.append(col)
        return available_moves

    def make_move(self, col, piece, verbose=0):
        # Effectuer un mouvement

        if self.is_valid_location(col):
            self.drop_piece(col, piece)
            if self.is_winner(piece):
                if verbose:
                    print("Victoire du joueur " + piece)
                    self.display()
                self.done = True
                return 1
            elif self.is_full():
                if verbose:
                    print("Le tableau est plein, partie terminée")
                    self.display()
                self.done = True
                return 2
            else:
                if verbose:
                    print("Tour {}".format(self.count_piece))
                    self.display()
                return 0
        else:
            if self.is_full():
                if verbose:
                    print("Le tableau est plein, partie terminée")
                self.done = True
                return 2
            else:
                if verbose:
                    print("Mouvement interdit !")
                return -1

    def cancel_last_move(self, verbose=0):
        # Cancel the last move made
        if self.last_move == []:
            print("No move to cancel")
        else:
            col = self.last_move.pop()
            row = self.get_next_open_row(col)+1
            self.grid[row, col] = ' '
            self.count_piece -= 1
        if verbose:
            self.display()

    def save_to_file(self, filename):
        # Sauvegarder la grille dans un fichier
        pass

    @classmethod
    def load_from_file(cls, filename):
        # Charger la grille depuis un fichier
        pass


if __name__ == "__main__":
    # Exemple d'utilisation 1
    grid = Grid()
    grid.display()

    grid.make_move(3, "x")
    grid.make_move(2, "o")
    grid.make_move(2, "x")
    grid.make_move(1, "o")
    grid.make_move(1, "o")
    grid.make_move(1, "x")
    grid.make_move(0, "o")
    grid.make_move(0, "o")
    grid.make_move(0, "o")
    grid.make_move(0, "x")

    # Exemple d'utilisation 2
    grid = Grid()
    grid.display()
    finished = False
    player = 'x'
    while not finished:
        available_moves = grid.get_available_moves()
        move = np.random.choice(available_moves)
        out = grid.make_move(move, player)
        if out == 0:
            if player == 'x':
                player = 'o'
            else:
                player = 'x'
        elif (out == 1) or (out == 2):
            finished = True
        else:
            pass

    # Exemple d'utilisation 3
    grid = Grid(cols=10, rows=14, n_in_a_row=6)
    grid.display()
    finished = False
    player = 'x'
    while not finished:
        available_moves = grid.get_available_moves()
        move = np.random.choice(available_moves)
        out = grid.make_move(move, player)
        if out == 0:
            if player == 'x':
                player = 'o'
            else:
                player = 'x'
        elif (out == 1) or (out == 2):
            finished = True
        else:
            pass
