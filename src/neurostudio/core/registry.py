"""Versioned block contracts consumed by Python validation and browser forms."""

def field(default, minimum=None, maximum=None, description="", options=None):
    return dict(default=default, minimum=minimum, maximum=maximum, description=description,options=options,type=type(default).__name__)


REGISTRY = {}


def block(name, kind, description, params=None, inputs=1):
    REGISTRY[name] = dict(type=name, version=1, kind=kind, description=description,
                          params=params or {}, inputs=inputs, implementation_id=f"python.{name}.v1")


block("Input", "model", "Tensor đầu vào: B là batch; ảnh dùng [B,C,H,W].", {"shape": field(["B", 3, 32, 32])}, 0)
block("Output", "model", "Đầu ra logits cho phân loại; không thêm Softmax trước CrossEntropyLoss.")
block("Linear", "model", "Áp dụng phép biến đổi tuyến tính trên chiều cuối.", {"in_features": field(32, 1, 1000000), "out_features": field(10, 1, 100000), "bias": field(True)})
block("Conv2D", "model", "Học bộ lọc không gian trên ảnh.", {"in_channels": field(3, 1, 4096), "out_channels": field(16, 1, 4096), "kernel_size": field(3, 1, 15), "stride": field(1, 1, 8), "padding": field(1, 0, 15), "dilation": field(1, 1, 8), "bias": field(True)})
for name in ["MaxPool2D", "AvgPool2D"]:
    block(name, "model", "Giảm kích thước không gian bằng pooling.", {"kernel_size": field(2, 1, 16), "stride": field(2, 1, 16), "padding": field(0, 0, 8)})
block("AdaptiveAvgPool2D", "model", "Lấy trung bình về kích thước không gian cố định.", {"output_size": field(1, 1, 32)})
block("Flatten", "model", "Gộp các chiều sau batch thành vector đặc trưng.")
block("Reshape", "model", "Đổi shape sau batch, giữ nguyên số phần tử.", {"shape": field([16, 8, 8])})
block("Dropout", "model", "Bỏ ngẫu nhiên phần tử chỉ khi train.", {"p": field(0.2, 0, 0.99)})
block("BatchNorm1D", "model", "Chuẩn hóa đặc trưng với running statistics.", {"num_features": field(32, 1, 100000)})
for name in ["ReLU", "Sigmoid", "Tanh", "Identity"]:
    block(name, "model", "Hàm kích hoạt/biến đổi phần tử.")
block("Softmax", "model", "Đổi logits thành phân phối; không dùng trước CrossEntropyLoss.", {"dim": field(1, 1, 4)})
block("Add", "model", "Cộng các tensor cùng shape (Alpha không tự broadcast).", inputs=2)
block("Concatenate", "model", "Ghép tensor theo chiều đã chọn, giữ nguyên các chiều khác.", {"dim": field(1, 1, 4)}, 2)
block("Dataset", "data", "Nguồn dữ liệu đã đăng ký.", inputs=0)
block("Split", "data", "Chia train/validation/test theo indices và seed.")
block("Imputer", "data", "Học median từ train cho số; mode cho category.")
block("StandardScaler", "data", "Học trung bình và độ lệch chuẩn chỉ trên train.")
block("MinMaxScaler", "data", "Học min/max chỉ trên train.")
block("OneHotEncoder", "data", "Mã hóa category, unseen categories bỏ qua theo policy.")
block("LogisticRegression", "model", "Bộ phân loại tuyến tính scikit-learn.", {"C": field(1.0, 0.00001, 10000), "max_iter": field(500, 10, 10000)}, 0)
block("KNN", "model", "Phân loại theo k láng giềng.", {"n_neighbors": field(5, 1, 100)}, 0)
block("Train", "experiment", "Chạy dataset + model với config đã lưu.", inputs=0)
block("Evaluate", "experiment", "Đánh giá validation; test chạy riêng theo yêu cầu.")
block("Export", "experiment", "Đóng gói Python + fitted state và sample.")

# Explicit backend metadata drives the palette and parameter forms.
CLASSIFIERS={"LogisticRegression","KNN","SVM","DecisionTree","RandomForest","ExtraTrees","GradientBoosting","AdaBoost"}
REGISTRY['LogisticRegression']['params'].update(solver=field('lbfgs',options=['lbfgs','liblinear','newton-cg','sag','saga']),tol=field(.0001,.00000001,1),fit_intercept=field(True),class_weight=field('none',options=['none','balanced']))
REGISTRY['KNN']['params'].update(weights=field('uniform',options=['uniform','distance']),algorithm=field('auto',options=['auto','ball_tree','kd_tree','brute']),p=field(2,1,5),leaf_size=field(30,1,200))
block('SVM','model','Support Vector Machine với xác suất được hiệu chỉnh trên train.',{'C':field(1.,.00001,10000),'kernel':field('rbf',options=['linear','rbf','poly','sigmoid']),'gamma':field('scale',options=['scale','auto']),'degree':field(3,1,10),'tol':field(.001,.00000001,1),'class_weight':field('none',options=['none','balanced'])},0)
for name in ['DecisionTree','RandomForest','ExtraTrees']:
    params={'criterion':field('gini',options=['gini','entropy','log_loss']),'max_depth':field(0,0,100,description='0 = không giới hạn'),'min_samples_split':field(2,2,10000),'min_samples_leaf':field(1,1,10000),'class_weight':field('none',options=['none','balanced'])}
    if name!='DecisionTree':params.update(n_estimators=field(100,1,2000),max_features=field('sqrt',options=['sqrt','log2','all']),bootstrap=field(name=='RandomForest'))
    block(name,'model','Cây quyết định / ensemble; số cây và độ sâu quyết định tốc độ, bộ nhớ.',params,0)
block('GradientBoosting','model','Boosting cây tuần tự scikit-learn.',{'n_estimators':field(100,1,2000),'learning_rate':field(.1,.00001,10),'max_depth':field(3,1,100),'subsample':field(1.,.05,1),'min_samples_leaf':field(1,1,10000)},0)
block('AdaBoost','model','AdaBoost với decision stump.',{'n_estimators':field(50,1,2000),'learning_rate':field(1.,.00001,10)},0)
REGISTRY['Imputer']['params']={'strategy':field('median',options=['median','mean','most_frequent'])}
REGISTRY['StandardScaler']['params']={'with_mean':field(True),'with_std':field(True)}
REGISTRY['MinMaxScaler']['params']={'clip':field(False)}
REGISTRY['OneHotEncoder']['params']={'handle_unknown':field('ignore',options=['ignore','error','infrequent_if_exist']),'min_frequency':field(1,1,10000)}
for name,spec in REGISTRY.items():
    spec['backend']='sklearn' if name in CLASSIFIERS else 'torch' if spec['kind']=='model' else 'shared'

REGRESSORS={'LinearRegression','Ridge','Lasso','KNNRegressor','SVR','DecisionTreeRegressor','RandomForestRegressor','GradientBoostingRegressor'}
for name in ['LinearRegression','Ridge','Lasso']:
    params={'fit_intercept':field(True)}
    if name!='LinearRegression':params['alpha']=field(1.,.000001,100000)
    if name=='Lasso':params.update(max_iter=field(1000,10,100000),tol=field(.0001,.00000001,1))
    block(name,'model','Dự đoán giá trị số liên tục bằng hồi quy.',params,0)
block('KNNRegressor','model','Hồi quy theo láng giềng.',{'n_neighbors':field(5,1,100),'weights':field('uniform',options=['uniform','distance']),'p':field(2,1,5)},0)
block('SVR','model','Support Vector Regression.',{'C':field(1.,.00001,10000),'epsilon':field(.1,0,1000),'kernel':field('rbf',options=['linear','rbf','poly']),'gamma':field('scale',options=['scale','auto'])},0)
for name in ['DecisionTreeRegressor','RandomForestRegressor','GradientBoostingRegressor']:
    params={'max_depth':field(3 if name=='GradientBoostingRegressor' else 0,0,100),'min_samples_leaf':field(1,1,10000)}
    if name!='DecisionTreeRegressor':params['n_estimators']=field(100,1,2000)
    if name=='GradientBoostingRegressor':params['learning_rate']=field(.1,.00001,10)
    block(name,'model','Hồi quy cây / ensemble.',params,0)
ESTIMATORS=CLASSIFIERS|REGRESSORS
for name in ESTIMATORS:
    REGISTRY[name]['backend']='sklearn'
    REGISTRY[name]['task']='regression' if name in REGRESSORS else 'classification'

block("PCA","data","Giảm chiều sau encoding. n_components phải nhỏ hơn số đặc trưng và số mẫu train.",{"n_components":field(2,1,499),"whiten":field(False)})
REGISTRY["PCA"]["backend"]="shared"


# Tensor blocks share one portable contract with runtime and code generation.
from ..engine.layers import LAYER_SPECS,NN_NAMES
for name, spec in LAYER_SPECS.items():
    block(name,"model",spec["description"],spec["params"],spec["inputs"])
    REGISTRY[name].update(backend="torch",category=spec["category"])
    if name in NN_NAMES:REGISTRY[name]["documentation_url"]=f"https://docs.pytorch.org/docs/2.7/generated/torch.nn.{NN_NAMES[name]}.html"
