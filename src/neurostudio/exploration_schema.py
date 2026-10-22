from typing import Literal
from pydantic import Field
from .core.schema import Strict

class Exploration(Strict):
    dataset_id:str
    target:str='target'
    features:list[str]=Field(default_factory=list,max_length=200)
    method:Literal['kmeans','dbscan','agglomerative']='kmeans'
    projection:Literal['pca','tsne','umap']='pca'
    clusters:int=Field(default=3,ge=2,le=30)
    eps:float=Field(default=.5,gt=0,le=100)
    min_samples:int=Field(default=5,ge=2,le=100)
    neighbors:int=Field(default=15,ge=2,le=100)
    perplexity:float=Field(default=30,ge=2,le=100)
    sample_limit:int=Field(default=1000,ge=10,le=5000)
    seed:int=Field(default=42,ge=0,le=2147483647)
