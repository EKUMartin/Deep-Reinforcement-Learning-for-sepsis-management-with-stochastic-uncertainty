# sampling 1000 stayids for pretraining
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from tqdm import tqdm
class Sample:
    def __init__(self, data, iterations=1000, threshold=0.01, sample_size=1000,sofa=0):
        # data: DataFrame with columns ['stay_id', 'sofa_score']
        self.data = data
        self.df=sofa
        self.iterations = iterations
        self.threshold = threshold
        self.sample_size = sample_size
        
        # Population
        self.pop_max = []
        self.pop_min = []
        self.pop_range = []
        
        # Best Sample
        self.best_sample_ids = []
        self.best_distance = float('inf')

    def get_sofa_stats(self,df):
        # max, min, range
        stats = df.groupby('stay_id')['sofa_score'].agg(['max', 'min'])
        stats['range'] = stats['max'] - stats['min']
        return stats['max'].values, stats['min'].values, stats['range'].values

    def get_wasserstein_distance(self, pop_dist, sample_dist):
        return wasserstein_distance(pop_dist, sample_dist)

    def get_sample(self):
        # population statistics
        self.pop_max, self.pop_min, self.pop_range = self.get_sofa_stats(self.df)
        unique_stay_ids = self.df['stay_id'].unique()
        
        if len(unique_stay_ids) < self.sample_size:
            raise ValueError("sample size is too large")

        print("sampling starts")
        # Sampling interation starts
        for i in tqdm(range(self.iterations)):
            #Sampling
            sampled_ids = np.random.choice(unique_stay_ids, size=self.sample_size, replace=True)
            sample_df = self.df[self.df['stay_id'].isin(sampled_ids)]
            
            # sample stats
            samp_max, samp_min, samp_range = self.get_sofa_stats(sample_df)
            
            # Wasserstein Distance
            dist_max = self.get_wasserstein_distance(self.pop_max, samp_max)
            dist_min = self.get_wasserstein_distance(self.pop_min, samp_min)
            dist_range = self.get_wasserstein_distance(self.pop_range, samp_range)
            
            total_distance = dist_max + dist_min + dist_range

            #check if best is updated
            if total_distance < self.best_distance:
                self.best_distance = total_distance
                self.best_sample_ids = sampled_ids
                print(f"Iteration {i+1}: New best with {total_distance:.4f}), max: {dist_max:.4f}, min:{dist_min:.4f},range:{dist_range:.4f}")
                if total_distance <= self.threshold: # best is over threshold then stop
                    print(f"({self.threshold}) is met. Early finish.")
                    break
                    
        return self.best_sample_ids

    def main(self):
        print("sampling starts")
        best_ids = self.get_sample()
        final_sample_df = self.data[self.data['stay_id'].isin(best_ids)].copy()
        return final_sample_df