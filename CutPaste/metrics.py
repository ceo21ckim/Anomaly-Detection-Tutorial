from sklearn.covariance import LedoitWolf
from sklearn.neighbors import KernelDensity

import torch 


class Density:
    def fit(self, embeddings):
        raise NotImplementedError
    
    def predict(self, embeddings):
        raise NotImplementedError
        

        
class GaussianDensityTorch:
    def fit(self, embeddings):
        self.mean = torch.mean(embeddings, axis=0)
        self.inv_cov = torch.Tensor(LedoitWolf().fit(embeddings.cpu()).precision_, device='cpu')
        
    def predict(self, embeddings):
        distances = self.mahalanobis_distance(embeddings, self.mean, self.inv_cov)
        return distances
    
    def mahalanobis_distance(values, mean, inv_covariance):
        
        if mean.dim() == 1:
            mean = mean.unsqueeze(0)
            
        x_mu = values - mean
        
        # reference: https://ita9naiwa.github.io/numeric%20calculation/2018/11/10/Einsum.html
        dist = torch.einsum("im,mn,in->i", x_mu, inv_covariance, x_mu) # matrix operation
        return dist.sqrt()
    

class GaussianDensitySklearn:
    def fit(self, embeddings):
        self.kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(embeddings)
        
    def predict(self, embeddings):
        scores = self.kde.score_samples(embedings)
        score = -scores
        return scores 
