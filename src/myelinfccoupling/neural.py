"""Small, weighted neural residual benchmark with nested participant validation."""
import warnings
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning


class SmallMLP:
    """Training-only scaling and one-hot metadata; 16 tanh units and Adam.

    Fixed epoch budgets intentionally may not converge. No automatic validation
    split or early stopping is used: tuning happens in grouped inner folds.
    """
    def __init__(self,epochs,seed,n_pairs):
        self.epochs=epochs;self.seed=seed;self.n_pairs=n_pairs

    def _features(self,x):
        return np.column_stack([self.scaler.transform(x[:,:3]),np.eye(self.n_pairs)[x[:,3].astype(int)]])

    def fit(self,x,y,sample_weight):
        self.scaler=StandardScaler().fit(x[:,:3],sample_weight=sample_weight)
        self.target_mean=np.average(y,weights=sample_weight)
        self.target_scale=max(np.sqrt(np.average((y-self.target_mean)**2,weights=sample_weight)),1e-8)
        self.model=MLPRegressor(hidden_layer_sizes=(16,),activation='tanh',solver='adam',alpha=.01,batch_size=min(512,len(y)),learning_rate_init=.001,max_iter=self.epochs,early_stopping=False,tol=0,n_iter_no_change=self.epochs+1,random_state=self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',ConvergenceWarning)
            self.model.fit(self._features(x),(y-self.target_mean)/self.target_scale,sample_weight=sample_weight)
        return self

    def predict(self,x):
        return self.target_mean+self.target_scale*self.model.predict(self._features(x))


def run(manifest,lut,out,batch='main',n_outer=5,n_inner=5,seed=20261006,candidates=(0,10,30),cap=1000):
    from .boosting import run as shared_run
    return shared_run(manifest,lut,out,batch,n_outer,n_inner,seed,candidates,cap,model_kind='neural')


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--lut',required=True);p.add_argument('--out',required=True);p.add_argument('--batch',default='main')
    p.add_argument('--outer-folds',type=int,default=5);p.add_argument('--inner-folds',type=int,default=5);p.add_argument('--seed',type=int,default=20261006)
    p.add_argument('--epochs',type=int,nargs='+',default=[0,10,30]);p.add_argument('--max-rows-per-person',type=int,default=1000)
    a=p.parse_args()
    try:run(a.manifest,a.lut,a.out,a.batch,a.outer_folds,a.inner_folds,a.seed,a.epochs,a.max_rows_per_person)
    except (ValueError,FileNotFoundError) as exc:p.error(str(exc))

if __name__=='__main__':main()
