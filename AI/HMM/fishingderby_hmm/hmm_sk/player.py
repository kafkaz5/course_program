#!/usr/bin/env python3

from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
from math import log
import random


class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def init_parameters(self):
        """
        In this function you should initialize the parameters you will need,
        such as the initialization of models, or fishes, among others.
        """
        self.hmm_model = [] # hidden markov model list
        self.hmm_species = [] #  fish type of model
        self.observation = []
        # create 2d list, list[i] stands for observation list of ith fish
        for i in range(N_FISH):
            self.observation.append([])
        pass

    def guess(self, step, observations):
        """
        This method gets called on every iteration, providing observations.
        Here the player should process and store this information,
        and optionally make a guess by returning a tuple containing the fish index and the guess.
        :param step: iteration number
        :param observations: a list of N_FISH observations, encoded as integers
        :return: None or a tuple (fish_id, fish_type)
        """

        # This code would make a random guess on each step:
        for i in range(N_FISH):
            self.observation[i].append(observations[i])
        if step < 100:
            return None
        else:
            if not self.hmm_model:
                return (step % N_FISH, random.randint(0, N_SPECIES - 1))
            else:
                fish_guess = step % N_FISH
                fish_obs = self.observation[fish_guess]
                prob = []
                for model in self.hmm_model:
                    prob_m = model.test_sequence(fish_obs)
                    prob.append(prob_m)
                seq = prob.index(max(prob))
                type_guess = self.hmm_species[seq]
                return (fish_guess, type_guess)

        # return None

    def reveal(self, correct, fish_id, true_type):
        """
        This methods gets called whenever a guess was made.
        It informs the player about the guess result
        and reveals the correct type of that fish.
        :param correct: tells if the guess was correct
        :param fish_id: fish's index
        :param true_type: the correct type of the fish
        :return:
        """
        obs = self.observation[fish_id]
        if correct:
            if not self.hmm_model:
                new_hmm = HMM(obs, 2)
                new_hmm.iter_mat()
                self.hmm_model.append(new_hmm)
                self.hmm_species.append(true_type)
            else:
                pass
        else:
            new_hmm = HMM(obs,2)
            new_hmm.iter_mat()
            self.hmm_model.append(new_hmm)
            self.hmm_species.append(true_type)


# following the model that was implemented in hmm3
class HMM():
    def __init__(self, obs, mat_dim = 2):
        self.obs = obs
        self.dim_s = mat_dim # dimension of state
        self.dim_o = 8 # dimension of observation
        self.ini_state = [1/self.dim_s for x in range(self.dim_s)]
        self.trans_mat = [[1/self.dim_s for y in range(self.dim_s)]for x in range(self.dim_s)]
        self.emiss_mat = [[1/self.dim_o for y in range(self.dim_o)]for x in range(self.dim_s)]

        # the following function has been moved to self.iter_mat()
        self.alpha, self.norm = self.tranverse_alpha(self.obs)
        self.beta = self.tranverse_beta(self.obs)
        self.gamma, self.gamma_s = self.tranverse_gamma()

    # def separate_product(a, b):
    #     s = len(a)     # vector length
    #     res = [ 0 for x in range(s)]
    #     for i in range(s):
    #         res[i]= a[i] * b[i]
    #     return res


    def get_trans_prob(self, state):
        dim = len(state)   # state dimension
        res = [0.0 for x in range(dim)]
        for i in range(dim):
            c_prob = [state[i] * x for x in self.trans_mat[i]]
            res = [res[j] + c_prob[j] for j in range(dim)]
        return res

    def get_obs_prob(self, state):
        '''
        :param state: current state distribution
        :return: observation distribution
        '''
        res = [0.0 for x in range(self.dim_o)]
        for i in range(self.dim_s):
            obs_prob = [state[i] * x for x in self.emiss_mat[i]]
            res = [res[j] + obs_prob[j] for j in range(self.dim_o)]
        res = [x for x in res]
        return res

    def alpha_forward(self, obs, alpha, arg):
        res = [0.0 for x in range(self.dim_s)] # alpha_t(i), where i differs
        if arg =="ini":
            for i in range(self.dim_s):
                res[i] = self.emiss_mat[i][obs] * alpha[i]
        if arg is None:
            trans_prob = self.get_trans_prob(alpha)
            for i in range(self.dim_s):
                res[i] = self.emiss_mat[i][obs] * trans_prob[i]
        try:
            s = 1/sum(res)  # sum of alpha
            res = [x * s for x in res]  # normalize
        except:
            s = 10000
            pass
        return res,s

    def beta_backforward(self, obs, beta):
        res = [0.0 for x in range(self.dim_s)]  # beta_t(i)
        for i in range(self.dim_s):
            for j in range(self.dim_s):
                res[i] += beta[j] * self.emiss_mat[j][obs] * self.trans_mat[i][j]
        return res

    def tranverse_alpha(self, obs):
        alpha = []
        norm = []
        for k in range(len(obs)):
            if k == 0:
                alpha_new, norm_new = self.alpha_forward(obs[k], self.ini_state, "ini")
                alpha.append(alpha_new)
                norm.append(norm_new)
            else:
                alpha_new, norm_new = self.alpha_forward(obs[k], alpha[k - 1], None)
                alpha.append(alpha_new)
                norm.append(norm_new)

        return alpha,norm

    def tranverse_beta(self, obs):
        T = len(obs)
        beta = [[0.0 for y in range(self.dim_s)]for x in range(T)]
        beta[T-1] = [ 1.0 * self.norm[T-1] for x in range(self.dim_s)]
        #beta.append(last_beta)
        for k in range(len(obs) - 2, -1, -1):
            beta_new = self.beta_backforward(obs[k+1], beta[k+1])
            beta_new = [x * self.norm[k] for x in beta_new]
            beta[k] = beta_new
        return beta


    def tranverse_gamma(self):
        gamma = []
        gamma_s =[]
        for t in range(len(self.obs)-1):
            gamma_t =[[ 0.0 for y in range(self.dim_s) ]for x in range(self.dim_s)]
            for i in range(self.dim_s):
                for j in range(self.dim_s):
                    gamma_t[i][j] = self.alpha[t][i] * self.trans_mat[i][j] * self.emiss_mat[j][self.obs[t + 1]] * \
                                    self.beta[t + 1][j]
            gamma_i = [sum(x) for x in gamma_t]
            gamma.append(gamma_t)
            gamma_s.append(gamma_i)

        gamma_s.append(self.alpha[-1])
        return gamma, gamma_s


    def hidden_state_backforward(self, delta, delta_idx):
        hid_s = []
        max_last_delta = max(delta[-1])
        last_s = (delta[-1]).index(max_last_delta)
        hid_s.append(last_s)
        for i in range(len(delta_idx)):
            s = hid_s[-1]
            hid_s.append(delta_idx[-i-1][s])
        return hid_s[::-1]

    def update_estimate(self):
        t_mat =[[0.0 for y in range(self.dim_s)] for x in range(self.dim_s)]
        e_mat =[[0.0 for y in range(self.dim_o)] for x in range(self.dim_s)]
        for i in range(self.dim_s):
            sum_s = sum([y[i] for y in self.gamma_s[0:-1]])
            for j in range(self.dim_s):
                t_mat[i][j] = sum([x[i][j] for x in self.gamma]) / sum_s

        for i in range(self.dim_s):
            sum_s = sum([y[i] for y in self.gamma_s])
            for k in range(self.dim_o):
                indicator =[1 if self.obs[p] == k else 0 for p in range(len(self.obs)) ]
                gamma_i = [x[i] for x in self.gamma_s]
                e_mat[i][k] = sum([ gamma_i[q] * indicator[q] for q in range(len(self.obs))])/sum_s
        ini_s = self.gamma_s[0]
        #ini_s= self.ini_state
        return t_mat, e_mat, ini_s

    def iter_mat(self, iter = 200):
        max_it = iter
        it = 0
        logprob = sum([-log(x) for x in self.norm])
        oldlogprob = -1000000
        while(it < max_it and logprob > oldlogprob):
            oldlogprob = logprob
            self.trans_mat, self.emiss_mat, self.ini_state = self.update_estimate()

            self.alpha,self.norm= self.tranverse_alpha(self.obs)
            self.beta = self.tranverse_beta(self.obs)
            self.gamma, self.gamma_s = self.tranverse_gamma()
            logprob = sum([-log(x) for x in self.norm])
            it += 1
        #print("iteration time = ", it)
        #self.output_mat()


    # helper functions
    def output_mat(self):
        out_t = str(self.dim_s) + " " + str(self.dim_s) + " " + " ".join(str(round(item,6)) for row in self.trans_mat for item in row)
        out_e = str(self.dim_s) + " " + str(self.dim_o) + " " + " ".join(str(round(item,6)) for row in self.emiss_mat for item in row)
        print(out_t)
        print(out_e)

    def test_sequence(self, obs):
        """
        Given model HMM = (A;B;pi) and a sequence of observations O\
        return P(O|HMM)
        """
        _, norm = self.tranverse_alpha(obs)
        logprob = sum([-log(x) for x in norm])
        return logprob

if __name__ == '__main__':
    obs = [1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4]
    obs2 = [3, 2, 3, 2, 3, 2, 3, 2, 1, 2, 3, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
           2, 2, 2, 2, 2, 2, 6, 2, 2, 2, 2, 4]
    hmm = HMM(obs,8)
    print("logrithm prob =", hmm.test_sequence(obs))
    hmm.iter_mat()
    print("logrithm prob =", hmm.test_sequence(obs))
    print("logrithm prob =", hmm.test_sequence(obs2))