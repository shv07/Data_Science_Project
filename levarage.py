import numpy as np

class Levarage:
    def __init__(self, data, mode='reduced'):
        '''Assuming the datapoints are in rows'''
        data
        self.len = data.shape[0] # number 
        self.pdf = self._find_multinomial_distrib(data, mode) # saving pdf for sampling
    
    def sample_ix(self,  percent=1):
        '''samples percent points and returns thier indexes'''
        num_of_sampes = int(self.len * percent / 100)
        return np.random.choice(range(self.len), num_of_sampes, p=self.pdf)
    
    def _find_multinomial_distrib(self, data, mode):
        q, r = np.linalg.qr(data, mode=mode)
        q_row_norm2 = np.linalg.norm(q, ord=2, axis=1)**2 # assumed the rows are dpts
        q_fnorm = np.linalg.norm(q, ord='fro')**2
        return q_row_norm2 / q_fnorm
    
        