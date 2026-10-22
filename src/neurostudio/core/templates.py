from .schema import Project


def chain(kind, specs):
    nodes = [dict(id=f"n{i}", type=t, params=p, position={"x": i * 210, "y": 100}) for i, (t, p) in enumerate(specs)]
    return dict(kind=kind, nodes=nodes, edges=[dict(source=nodes[i]["id"], target=nodes[i+1]["id"]) for i in range(len(nodes)-1)])


def template(name="iris"):
    data = chain("data", [("Dataset", {}), ("Split", {}), ("Imputer", {}), ("StandardScaler", {}), ("OneHotEncoder", {})])
    exp = chain("experiment", [("Train", {}), ("Evaluate", {}), ("Export", {})])
    demos={
        'iris_rf':('Iris · Random forest','RandomForest',{'n_estimators':100,'max_depth':5}),
        'iris_svm':('Iris · Support vector machine','SVM',{'C':1.,'kernel':'rbf'}),
        'iris_pca':('Iris · PCA + logistic regression','LogisticRegression',{'C':1.,'max_iter':500}),
        'diabetes_forest':('Diabetes · Random forest regression','RandomForestRegressor',{'n_estimators':100,'max_depth':8}),
    }
    if name in demos:
        title,model,params=demos[name]
        regression=name=='diabetes_forest'
        if name=='iris_pca':data=chain('data',[(n['type'],n['params']) for n in data['nodes']]+[('PCA',{'n_components':2})])
        return Project(name=title,task='regression' if regression else 'classification',data={'dataset_id':'diabetes' if regression else 'iris','target':'target','stratify':not regression},graphs=[data,chain('model',[(model,params)]),exp])
    if name == "iris":
        return Project(name="Iris • Bông hoa đầu tiên", graphs=[data, chain("model", [("LogisticRegression", {"C": 1.0, "max_iter": 500})]), exp])
    if name=='regression':
        return Project(name='Hồi quy · Giá trị liên tục',task='regression',data={'dataset_id':'diabetes','target':'target','stratify':False},graphs=[data,chain('model',[('Ridge',{'alpha':1.})]),exp])
    if name != "cifar10":
        raise ValueError("UNKNOWN_TEMPLATE")
    return Project(name="CIFAR-10 • CNN của tôi", backend="torch", classes=10,
        data={"dataset_id": "cifar10", "target": "target"},
        graphs=[chain("data", [("Dataset", {}), ("Split", {})]), chain("model", [
            ("Input", {"shape": ["B", 3, 32, 32]}),
            ("Conv2D", {"in_channels": 3, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1, "dilation": 1, "bias": True}),
            ("ReLU", {}), ("MaxPool2D", {"kernel_size": 2, "stride": 2, "padding": 0}),
            ("AdaptiveAvgPool2D", {"output_size": 1}), ("Flatten", {}),
            ("Linear", {"in_features": 16, "out_features": 10, "bias": True}), ("Output", {})]), exp])
