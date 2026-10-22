from __future__ import annotations
import math
from .schema import Project
from .registry import REGISTRY,ESTIMATORS
from ..engine.layers import LAYER_SPECS,validate_parameter,infer_layer


class GraphError(ValueError):
    pass


def validate(project: Project):
    errors, warnings, inferred, counts, orders = [], [], {}, {}, {}
    def err(code, message, node=None, kind=None):
        errors.append(dict(code=code, message=message, node_id=node, graph=kind))
    if project.task=='regression' and project.backend!='sklearn':err('TASK_BACKEND','Hồi quy hiện dùng scikit-learn; chọn project dữ liệu bảng.')
    kinds = [g.kind for g in project.graphs]
    if sorted(kinds) != ["data", "experiment", "model"]:
        err("GRAPH_KINDS", "Cần đúng một DataGraph, ModelGraph và ExperimentGraph.")
    for graph in project.graphs:
        nodes = {n.id: n for n in graph.nodes}
        if len(nodes) != len(graph.nodes):
            err("DUPLICATE_ID", "ID block bị trùng.", kind=graph.kind)
        incoming = {n: [] for n in nodes}
        outgoing = {n: [] for n in nodes}
        seen_edges = set()
        for edge in graph.edges:
            if edge.source not in nodes or edge.target not in nodes:
                err("ORPHAN_EDGE", "Kết nối tới block không còn tồn tại.", edge.target, graph.kind)
                continue
            if edge.source_port != "out" or edge.target_port != "in":
                err("PORT", "Tên cổng không hợp lệ: cần out → in.", edge.target, graph.kind)
            key = (edge.source, edge.target, edge.order)
            if key in seen_edges:
                err("DUPLICATE_EDGE", "Kết nối bị lặp.", edge.target, graph.kind)
            seen_edges.add(key)
            incoming[edge.target].append(edge)
            outgoing[edge.source].append(edge.target)
        degree = {k: len(v) for k, v in incoming.items()}
        queue = [k for k, v in degree.items() if v == 0]
        order = []
        while queue:
            key = queue.pop(0); order.append(key)
            for child in outgoing[key]:
                degree[child] -= 1
                if degree[child] == 0: queue.append(child)
        if len(order) != len(nodes):
            err("CYCLE", "Graph có vòng lặp. Hãy xóa một kết nối trong vòng.", kind=graph.kind)
        orders[graph.kind] = order
        for node in graph.nodes:
            spec = REGISTRY.get(node.type)
            if not spec or node.version != 1:
                err("BLOCK_VERSION", "Block hoặc phiên bản chưa được hỗ trợ.", node.id, graph.kind); continue
            if spec["kind"] != graph.kind:
                err("PORT_TYPE", "Block không thuộc loại graph này.", node.id, graph.kind)
            if node.type in ESTIMATORS and project.backend != "sklearn":
                err("BACKEND", "Estimator này cần backend scikit-learn.", node.id, graph.kind)
            if graph.kind == "model" and project.backend == "sklearn" and node.type not in ESTIMATORS:
                err("BACKEND", "Tensor blocks cần backend PyTorch.", node.id, graph.kind)
            if spec.get('task') and spec['task']!=project.task:err('TASK_ESTIMATOR','Chọn estimator phù hợp với loại bài toán.',node.id,graph.kind)
            for p in node.params:
                if p not in spec["params"]:
                    err("UNKNOWN_PARAMETER", f"Tham số {p} chưa được hỗ trợ.", node.id, graph.kind)
            for key, rule in spec["params"].items():
                value = node.params.get(key, rule["default"])
                default = rule["default"]
                valid = type(value) is type(default) if not isinstance(default, float) else isinstance(value, (int, float)) and not isinstance(value, bool)
                if isinstance(value,float) and not math.isfinite(value):valid=False
                if rule.get("options") and value not in rule["options"]:valid=False
                if node.type in LAYER_SPECS:
                    valid=validate_parameter(value,rule)
                else:
                    valid=valid and (rule['minimum'] is None or value>=rule['minimum']) and (rule['maximum'] is None or value<=rule['maximum'])
                if not valid:
                    err("PARAMETER", f"{key}: giá trị ngoài phạm vi hoặc sai kiểu.", node.id, graph.kind)
            arity = spec["inputs"]
            if len(incoming[node.id]) != arity:
                err("ARITY", f"{node.type} cần {arity} đầu vào, nhận {len(incoming[node.id])}.", node.id, graph.kind)
        if graph.kind == "data":
            roots = [n for n in nodes.values() if n.type == "Dataset"]
            if len(roots) != 1 or sum(n.type == "Split" for n in nodes.values()) != 1:
                err("DATA_ROOT", "Cần một nguồn dữ liệu và một bước Split.", kind="data")
            if any(len(v) > 1 for v in outgoing.values()):
                err("DATA_BRANCH", "DataGraph Alpha cần một pipeline tuần tự.", kind="data")
            split_seen = False
            for key in order:
                t = nodes[key].type
                if t == "Split": split_seen = True
                if t in ("Imputer", "StandardScaler", "MinMaxScaler", "OneHotEncoder", "PCA") and not split_seen:
                    err("LEAKAGE", "Đặt Split trước bước fit. Chỉ được fit trên train.", key, "data")
            types=[nodes[k].type for k in order if nodes[k].type not in ("Dataset","Split")]
            if len(types)!=len(set(types)):err("DUPLICATE_TRANSFORM", "Mỗi preprocessing block chỉ xuất hiện một lần trong Alpha.", kind="data")
            rank={"Imputer":0,"StandardScaler":1,"MinMaxScaler":1,"OneHotEncoder":2,"PCA":3}
            if [rank.get(t,99) for t in types] != sorted(rank.get(t,99) for t in types):
                err("TRANSFORM_ORDER", "Alpha dùng thứ tự Imputer → Scaler → OneHotEncoder → PCA; hãy nối lại theo thứ tự này.", kind="data")
            scalers = [n for n in nodes.values() if n.type in ("StandardScaler", "MinMaxScaler")]
            if len(scalers) > 1: err("SCALER", "Chọn một scaler duy nhất.", kind="data")
            if project.backend == "torch" and any(n.type not in ("Dataset", "Split") for n in nodes.values()):
                err("MODALITY", "Ảnh dùng augmentation/normalize trong cấu hình dữ liệu.", kind="data")
        if graph.kind == "experiment":
            if [nodes[k].type for k in order] != ["Train", "Evaluate", "Export"]:
                err("EXPERIMENT", "Pipeline thí nghiệm cần Train → Evaluate → Export.", kind="experiment")
        if graph.kind != "model": continue
        if project.backend == "sklearn":
            if len(nodes) != 1: err("ESTIMATOR", "Chọn đúng một estimator.", kind="model")
            continue
        if sum(n.type == "Input" for n in nodes.values()) != 1 or sum(n.type == "Output" for n in nodes.values()) != 1:
            err("MODEL_BOUNDARY", "Model cần đúng một Input và Output.", kind="model")
        for node in nodes.values():
            if not outgoing[node.id] and node.type != "Output":
                err("UNUSED_NODE", "Block không dẫn tới Output.", node.id, "model")
        shapes = {}
        for key in order:
            n = nodes[key]; t = n.type
            if t not in REGISTRY: continue
            p = {k: n.params.get(k, v["default"]) for k, v in REGISTRY[t]["params"].items()}
            try:
                sources = sorted(incoming[key], key=lambda e: (e.order, e.source))
                ins = [shapes[e.source] for e in sources]
                s = list(ins[0]) if ins else []
                params = 0
                if t == "Input":
                    s = p["shape"]
                    if len(s) < 2 or s[0] != "B" or any(type(x) is not int or x < 1 for x in s[1:]): raise GraphError("Shape cần [B, kích thước dương...].")
                    if s != ["B", 3, project.data.image_size, project.data.image_size]: raise GraphError("Input cần khớp ảnh RGB và image_size trong Data Lab.")
                elif t in LAYER_SPECS:
                    s,params=infer_layer(t,p,ins)
                    if t.startswith('InstanceNorm') and math.prod(ins[0][2:])<=1:raise GraphError('InstanceNorm training needs more than one spatial element per channel.')
                    if t.startswith('BatchNorm') and project.training.batch_size*math.prod(ins[0][2:])<=1:raise GraphError('BatchNorm training needs more than 1 value per channel. Increase batch size or normalize before global pooling.')
                    if t in ['Softmax','LogSoftmax','Softmin'] and any(nodes[child].type=='Output' for child in outgoing[key]):
                        err('LOSS_HEAD','Use logits at Output; remove probability/log-probability normalization before the classification loss.',key,'model')
                elif t == "Output":
                    if s != ["B", project.classes]: raise GraphError(f"Classification cần [B,{project.classes}], nhận {s}.")
                if any(isinstance(x,int) and x < 1 for x in s): raise GraphError("Layer tạo kích thước tensor không dương.")
                shapes[key] = s; inferred[key] = dict(shape=s, dtype="float32"); counts[key] = params
            except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as e:
                err("SHAPE", str(e), key, "model")
    if sum(counts.values()) > 20000000:
        err("MODEL_LIMIT", "Alpha giới hạn 20 triệu tham số để bảo vệ bộ nhớ máy.")
    if project.backend == "sklearn" and project.training.precision != "fp32":
        warnings.append(dict(code="CPU_ONLY", message="scikit-learn chạy CPU; precision AMP chỉ áp dụng PyTorch."))
    activation_bytes=sum(math.prod(v['shape'][1:])*project.training.batch_size*4 for v in inferred.values())
    memory_estimate=dict(weights_mb=sum(counts.values())*4/1024**2,layer_outputs_mb=activation_bytes/1024**2,
        batch_size=project.training.batch_size,note="FP32 sum of layer outputs; excludes backward workspaces, optimizer state, caches and prefetch. Not a VRAM guarantee.")
    return dict(memory_estimate=memory_estimate,ok=not errors, errors=errors, warnings=warnings, inferred=inferred, parameters=sum(counts.values()), parameter_counts=counts, orders=orders)


def resolve(project):
    result = validate(project)
    if not result["ok"]: raise GraphError("; ".join(x["message"] for x in result["errors"]))
    data = project.model_dump()
    for g in data["graphs"]:
        for n in g["nodes"]:
            n["params"] = {k: n["params"].get(k,v["default"]) for k,v in REGISTRY[n["type"]]["params"].items()}
    data["execution_order"] = result["orders"]
    data["resource_estimate"] = result["memory_estimate"]
    from neurostudio import __version__
    data["codegen_version"] = __version__
    return data
