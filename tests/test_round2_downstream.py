import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
from proehn.downstream import RepresentationMatrix,Projection,fit_clusters,fit_pca,select_cluster_number
from proehn.evaluation import BenchmarkFold


def test_projection_is_not_a_clustering_representation_and_pca_fits_training_only():
    with pytest.raises(ValueError,match='t-SNE'):
        fit_clusters(Projection(('p',),np.ones((1,2)),'t-SNE'),None)
    seen=[]
    estimator=SimpleNamespace(fit=lambda x:seen.append(x.copy()),transform=lambda x:x[:,:1])
    train=RepresentationMatrix(('a','b','c'),np.arange(6).reshape(3,2),'vectorized_theta','toy-v1')
    pca=fit_pca(train,estimator)
    pca.transform(RepresentationMatrix(('test',),np.ones((1,2))*100,'vectorized_theta','toy-v1'))
    assert len(seen)==1 and seen[0].max()==5
    assert select_cluster_number(train,[2],lambda x,k:0.)==2


