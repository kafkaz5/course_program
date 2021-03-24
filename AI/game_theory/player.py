#!/usr/bin/env python3
import random
import time
from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR


class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        """
        Function that generates the loop of the game. In each iteration
        the human plays through the keyboard and send
        this to the game through the sender. Then it receives an
        update of the game through receiver, with this it computes the
        next movement.
        :return:
        """

        while True:
            # send message to game that you are ready
            msg = self.receiver()
            if msg["game_over"]:
                return


class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()
        self.initial_value = 1000
        self.alpha = -self.initial_value
        self.beta = self.initial_value

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        :return:
        """

        # Generate game tree object
        first_msg = self.receiver()
        # Initialize your minimax model
        model = self.initialize_model(initial_data=first_msg)

        while True:
            msg = self.receiver()

            # Create the root node of the game tree
            node = Node(message=msg, player=0)

            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move = self.search_best_next_move(
                model=model, initial_tree_node=node)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})

    def initialize_model(self, initial_data):
        """
        Initialize your minimax model
        :param initial_data: Game data for initializing minimax model
        :type initial_data: dict
        :return: Minimax model
        :rtype: object

        Sample initial data:
        { 'fish0': {'score': 11, 'type': 3},
          'fish1': {'score': 2, 'type': 1},
          ...
          'fish5': {'score': -10, 'type': 4},
          'game_over': False }

        Please note that the number of fishes and their types is not fixed between test cases.
        """
        self.state_dict = {}
        fish_num = len(initial_data) - 1
        piece_num = fish_num + 2  # number of fish plus hook
        # maybe some improvement here
        #self.zobristable = [[[random.randint(1, 2 ** 32 - 1) for k in range(piece_num)] for j in range(20)] for i in
        #                    range(20)]
        self.zobristable = [[random.randint(1, 2 ** 32 - 1) for k in range(piece_num)] for j in range(400)]
        # EDIT THIS METHOD TO RETURN A MINIMAX MODEL ###

        return self.zobristable

    def search_best_next_move(self, model, initial_tree_node):
        """
        Use your minimax model to find best possible next move for player 0 (green boat)
        :param model: Minimax model
        :type model: object
        :param initial_tree_node: Initial game tree node
        :type initial_tree_node: game_tree.Node
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """

        # EDIT THIS METHOD TO RETURN BEST NEXT POSSIBLE MODE FROM MINIMAX MODEL ###

        # NOTE: Don't forget to initialize the children of the current node
        #       with its compute_and_get_children() method!
        # max_depth = 4
        self.start_time = time.time()
        self.timeout = False
        # for i in range(5):
        #     for j in range(5):
        #print(self.check_heuristic(initial_tree_node, move))
        depth = 1
        for depth in range(1, 8):
        # here it should be no top limit for depth doing iternative deepening, but for higher score of assignment we did like this.
        # can change to while True and stop with timeout.
            if not self.timeout:
                h, cur_best_node = self.alpha_beta_prunning(initial_tree_node, depth, self.alpha, self.beta)

            if not self.timeout:
                best_node = cur_best_node
        best_next_move = best_node.move
        #self.check_heuristic(initial_tree_node,4)
        #print(self.check_move(initial_tree_node,[3,0,1,1]))
        return ACTION_TO_STR[best_next_move]

    def alpha_beta_prunning(self, tree_node, depth, alpha, beta):
    #def alpha_beta_prunning(self, tree_node, depth):
        if time.time() - self.start_time > 0.06:
            self.timeout = True
        if not self.timeout:
            if tree_node.depth >= depth:
                return self.get_heuristic(tree_node), tree_node

            best_child = None
            if tree_node.depth % 2 == 0:
                # if no children, create children,then change order
                # by doing this, we avoid sorting multiple times
                if not tree_node.children:
                    tree_node.children = sorted(tree_node.compute_and_get_children(), key = lambda n: self.get_heuristic(n), reverse = True)
                value = -self.initial_value
                for child in tree_node.children:
                    #best_child = child
                    heuristic, successor_node = self.alpha_beta_prunning(child, depth, alpha, beta)
                    # print("depth =", child.depth, "h = ",heuristic, "last move=", child.move)
                    value = max(value, heuristic)
                    #self.alpha = max(self.alpha, value)
                    if value > alpha:
                        alpha = value
                        best_child = child
                    if alpha >= beta:
                        break
                # if child.depth == 3 and best_child:
                #     print("move = ", best_child.move, "alpha = ", alpha)
                return alpha, best_child
            else:
                value = self.initial_value
                if not tree_node.children:
                    tree_node.children = sorted(tree_node.compute_and_get_children(), key = lambda n: self.get_heuristic(n))
                for child in tree_node.children:
                    #best_child = child
                    heuristic, successor_node = self.alpha_beta_prunning(child, depth, alpha, beta)
                    value = min(value, heuristic)
                    #beta = min(beta, value)
                    if value < beta:
                        beta = value
                        best_child = child
                    if beta <= alpha:
                        break
                # if child.depth == 2:
                #     print("move = ", best_child.move, "beta = ", beta)
                return beta, best_child
        else:
            return self.get_heuristic(tree_node), tree_node

    def get_heuristic(self, tree_node):
        """
        check dictionary key if success, return value
        if failure, calculate heuristic value
        """

        hash_key = self.zobrist_hashing(tree_node)
        if hash_key in self.state_dict:
            return self.state_dict[hash_key]
        else:
            key_value = self.heuristic_function(tree_node)
            self.state_dict[hash_key] = key_value
            return key_value

    def zobrist_hashing(self, tree_node):
        h = 0
        zobristable = self.zobristable
        tree_node_state = tree_node.state
        hook_positions = tree_node_state.hook_positions
        fish_positions = tree_node_state.fish_positions
        [player0_score, player1_score] = tree_node_state.get_player_scores()
        for i in hook_positions:  # hash of hook position
            pos_x = hook_positions[i][0]
            pos_y = hook_positions[i][1]

            index = pos_x + pos_y * 20
            # from 0-399
            h ^= zobristable[index][i]
        for j in fish_positions:  # hash of fish position
            pos_x = fish_positions[j][0]
            pos_y = fish_positions[j][1]
            index = pos_x + pos_y * 20
            h ^= zobristable[index][j + 2]  # remember to plus 2 because of hooks
        h ^= 10001 * player0_score  # hash of scores
        h ^= 10003 * player1_score
        return h  # hash key

    def heuristic_function(self, tree_node):
        """
        Calculate the approximate heuristic value of the current state
        Naive attempt to compare the score
        """
        tree_node_state = tree_node.state
        hook_positions = tree_node_state.hook_positions
        fish_positions = tree_node_state.fish_positions
        fish_scores = tree_node_state.fish_scores
        player_caught = tree_node_state.player_caught
        heuristic = 0

        for i in fish_positions:
            dis_player_x = abs(fish_positions[i][0] - hook_positions[0][0])
            dis_player_x = min(dis_player_x, 20 - dis_player_x)  # get true min distance
            # it is not correct but performs better than correct one.
            dis_player_y = abs(fish_positions[i][1] - hook_positions[0][1])
            dis_opponent_x = abs(fish_positions[i][0] - hook_positions[1][0])
            dis_opponent_x = min(dis_opponent_x, 20 - dis_opponent_x)  # get true min distance
            dis_opponent_y = abs(fish_positions[i][1] - hook_positions[1][1])
            dis_player = dis_player_x + dis_player_y
            dis_opponent = dis_opponent_x + dis_opponent_y
            # if a fish is caught by opponent, he needs to first catch the fish
            if fish_scores[i] > 0:
                # print('distance of fish ',i ,' is player:', dis_player,' opponent:',dis_opponent)
                # if dis_player <= dis_opponent:
                if dis_opponent != 0:
                    heuristic = heuristic + fish_scores[i] / (dis_player + 1)
                if dis_player != 0:
                    heuristic = heuristic - fish_scores[i] / (dis_opponent + 1)

        # here is a different for loop as correctly calculate distance but perform worse because it is too complex for heuristic.

        # for i in fish_positions:
        #     diff_player_x = abs(fish_positions[i][0] - hook_positions[0][0])
        #     diff_against_x = abs(hook_positions[0][0] - hook_positions[1][0])
        #     # dis_player_x = min(dis_player_x, 20 - dis_player_x)  # get true min distance
        #     dis_player_y = abs(fish_positions[i][1] - hook_positions[0][1])
        #     diff_opponent_x = abs(fish_positions[i][0] - hook_positions[1][0])
        #     # dis_opponent_x = min(dis_opponent_x, 20 - dis_opponent_x) # get true min distance
        #     if diff_opponent_x < diff_player_x and diff_against_x < diff_player_x:
        #         dis_player_x = 20 - diff_player_x
        #     else:
        #         dis_player_x = diff_player_x
        #     if diff_opponent_x > diff_player_x and diff_against_x < diff_player_x:
        #         dis_opponent_x = 20 - diff_player_x
        #     else:
        #         dis_opponent_x = diff_player_x
        #     dis_opponent_y = abs(fish_positions[i][1] - hook_positions[1][1])
        #     dis_player = dis_player_x + dis_player_y
        #     dis_opponent = dis_opponent_x + dis_opponent_y
        #     # if a fish is caught by opponent, he needs to first catch the fish
        #     if fish_scores[i] > 0:
        #         # print('distance of fish ',i ,' is player:', dis_player,' opponent:',dis_opponent)
        #         # if dis_player <= dis_opponent:
        #         heuristic = heuristic + fish_scores[i] / (dis_player + 1) - fish_scores[i] / (dis_opponent + 1)

        [player0_score, player1_score] = tree_node_state.get_player_scores()
        heuristic = player0_score - player1_score + heuristic

        return heuristic

    def check_heuristic(self, tree_node, depth):
        '''
        function to debug:
        output all the heuristic value from the children, grandchildren of the node
        Until the depth is reached
        Just Igore it !
        '''
        if tree_node.depth >= depth:
            return

        children = tree_node.compute_and_get_children()
        for child in children:
            print("depth = ", child.depth, "h = ", self.get_heuristic(child),"lastmove = ", child.move)
            self.check_heuristic(child,depth)
        return None

    def check_move(self, tree_node, move):
        '''
       function to debug:
       output all the heuristic value of node along the path
       Path is defined by the move: e.g.[ 3,1,1]
       Just Igore it !
       '''
        cur_node = tree_node
        for i in move:
            for child in cur_node.children:
                if child.move == i:
                    cur_node = child
            return self.get_heuristic(cur_node)