"""Portable CPU clustering and projections. Exploration is not supervised evaluation."""
import json
from pathlib import Path
import time


def frame_data(descriptor,target='target'):
    import pandas as pd
    if descriptor['kind'] in ['iris','diabetes']:
        from sklearn.datasets import load_iris,load_diabetes
        return (load_iris if descriptor['kind']=='iris' else load_diabetes)(as_frame=True).data
    frame=pd.read_csv(descriptor['path'])
    if target in frame:frame=frame.drop(columns=target)
    return frame


def run_exploration(config,descriptor,out,control=lambda:None):
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler,OneHotEncoder
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering
    from sklearn.metrics import silhouette_score,davies_bouldin_score
    from joblib import dump
    began=time.monotonic();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    frame=frame_data(descriptor,config.get('target','target'))
    if config.get('features'):frame=frame[config['features']]
    if not 2<=len(frame)<=250000 or not 1<=len(frame.columns)<=200:raise ValueError('EXPLORATION_SIZE: Cần 2–250.000 dòng và 1–200 đặc trưng.')
    indices=np.random.default_rng(config['seed']).choice(len(frame),min(config['sample_limit'],len(frame)),replace=False)
    frame=frame.iloc[indices];control()
    numeric=frame.select_dtypes(include='number').columns.tolist();categorical=[c for c in frame if c not in numeric]
    transformations=[]
    if numeric:transformations.append(('numeric',make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler()),numeric))
    if categorical:transformations.append(('category',make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore',sparse_output=False,max_categories=32)),categorical))
    prep=ColumnTransformer(transformations);matrix=prep.fit_transform(frame)
    if matrix.shape[1]>512 or not np.isfinite(matrix).all():raise ValueError('EXPLORATION_FEATURES: Tối đa 512 đặc trưng hữu hạn sau encoding.')
    method=config['method']
    if method=='kmeans':
        if config['clusters']>=len(matrix):raise ValueError('CLUSTERS: Số cụm phải nhỏ hơn số mẫu.')
        model=KMeans(n_clusters=config['clusters'],random_state=config['seed'],n_init=10)
    elif method=='dbscan':model=DBSCAN(eps=config['eps'],min_samples=config['min_samples'],n_jobs=1)
    else:
        if config['clusters']>=len(matrix):raise ValueError('CLUSTERS: Số cụm phải nhỏ hơn số mẫu.')
        model=AgglomerativeClustering(n_clusters=config['clusters'])
    labels=model.fit_predict(matrix);control()
    projection=config['projection'];note=''
    if projection=='pca':
        dims=min(2,*matrix.shape);reducer=PCA(n_components=dims,random_state=config['seed']);points=reducer.fit_transform(matrix)
        if dims==1:points=np.column_stack([points,np.zeros(len(matrix))])
        note='Explained variance: '+str(reducer.explained_variance_ratio_.tolist())
    elif projection=='tsne':
        from sklearn.manifold import TSNE
        if config['perplexity']>=len(matrix):raise ValueError('PERPLEXITY: Perplexity phải nhỏ hơn số mẫu.')
        reducer=TSNE(n_components=2,perplexity=config['perplexity'],random_state=config['seed'],init='random',learning_rate='auto',max_iter=500,n_jobs=1)
        points=reducer.fit_transform(matrix);note='t-SNE is transductive; distances between distant groups are not calibrated.'
    else:
        try:from umap import UMAP
        except ImportError as e:raise ValueError('UMAP_REQUIRED: Cài môi trường đầy đủ trong Máy & cài đặt.') from e
        reducer=UMAP(n_components=2,n_neighbors=min(config['neighbors'],len(matrix)-1),random_state=config['seed'],n_jobs=1)
        points=reducer.fit_transform(matrix);note='UMAP is stochastic; the seed and preprocessing are recorded.'
    selected=labels>=0;k=len(set(labels[selected]));metrics={'clusters':k,'noise_samples':int((labels<0).sum()),'samples':len(frame),'features':matrix.shape[1]}
    if 1<k<int(selected.sum()):
        metrics.update(silhouette=float(silhouette_score(matrix[selected],labels[selected],sample_size=min(1000,int(selected.sum())),random_state=config['seed'])),davies_bouldin=float(davies_bouldin_score(matrix[selected],labels[selected])))
    if method=='kmeans':metrics['inertia']=float(model.inertia_)
    result=dict(metrics=metrics,points=[dict(x=float(p[0]),y=float(p[1]),cluster=int(label),row=int(index)) for p,label,index in zip(points,labels,indices)],projection=projection,method=method,note=note,supervised_score=False,elapsed_seconds=time.monotonic()-began,features=prep.get_feature_names_out().tolist())
    dump(dict(preprocessor=prep,model=model,reducer=reducer,method=method),out/'pipeline.joblib')
    for name,value in [('config.json',config),('result.json',result),('dataset-source.json',descriptor)]:
        (out/name).write_text(json.dumps(value,allow_nan=False,indent=2)+'\n', encoding='utf-8')
    pd.DataFrame(result['points']).to_csv(out/'clusters.csv',index=False)
    return result


def predict_clusters(directory,rows):
    import pandas as pd
    from joblib import load
    state=load(Path(directory)/'pipeline.joblib')
    if state['method']!='kmeans':raise ValueError('NO_PREDICT: DBSCAN/Agglomerative chỉ gán cụm cho dữ liệu đã fit; dùng clusters.csv.')
    return state['model'].predict(state['preprocessor'].transform(pd.DataFrame(rows))).tolist()
