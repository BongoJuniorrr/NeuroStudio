from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Node(Strict):
    id: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_-]{0,63}$")
    type: str
    version: int = 1
    params: dict = Field(default_factory=dict)
    position: dict = Field(default_factory=lambda: {"x": 0, "y": 0})


class Edge(Strict):
    source: str
    target: str
    source_port: str = "out"
    target_port: str = "in"
    order: int = 0


class Graph(Strict):
    kind: Literal["data", "model", "experiment"]
    nodes: list[Node] = Field(max_length=250)
    edges: list[Edge] = Field(max_length=1000)


class DataConfig(Strict):
    dataset_id: str = "iris"
    target: str = "target"
    features: list[str] = Field(default_factory=list)
    train: float = Field(default=0.7, gt=0, lt=1)
    validation: float = Field(default=0.15, gt=0, lt=1)
    test: float = Field(default=0.15, gt=0, lt=1)
    stratify: bool = True
    image_size: int = Field(default=32, ge=8, le=256)
    augment_flip: bool = True
    augment_crop: bool = False
    normalize: bool = True
    sample_limit: int | None = Field(default=None, ge=30, le=60000)
    @model_validator(mode="after")
    def fractions(self):
        if abs(self.train + self.validation + self.test - 1) > 1e-6:
            raise ValueError("Tỷ lệ train/validation/test phải có tổng bằng 1.")
        return self


class MetricSettings(Strict):
    metrics:list[str]=Field(default_factory=list,max_length=30)
    primary_metric:str|None=Field(default=None,max_length=64)
    primary_mode:Literal['auto','min','max']='auto'
    metric_formulas:dict[str,str]=Field(default_factory=dict)
    metric_code:str=Field(default='',max_length=12000)
    loss_code:str=Field(default='',max_length=12000)
    focal_gamma:float=Field(default=2,ge=0,le=10)
    @model_validator(mode='after')
    def metric_contract(self):
        import re
        if self.primary_metric and self.metrics and self.primary_metric not in self.metrics:raise ValueError('PRIMARY_METRIC: Primary metric must be selected for monitoring.')
        if len(self.metric_formulas)>10 or any(not re.fullmatch(r'custom_[A-Za-z0-9_]{1,48}',k) or len(v)>400 for k,v in self.metric_formulas.items()):raise ValueError('METRIC_FORMULA: Use up to 10 custom_ names and formulas <=400 characters.')
        return self


class Training(MetricSettings):
    epochs: int = Field(default=5, ge=1, le=10000)
    batch_size: int = Field(default=64, ge=1, le=4096)
    learning_rate: float = Field(default=0.001, gt=0, le=10)
    optimizer: Literal["SGD", "Adam", "AdamW", "RMSprop", "Adagrad"] = "AdamW"
    weight_decay: float = Field(default=0.0001, ge=0, le=1)
    momentum: float = Field(default=0.9, ge=0, lt=1)
    adam_beta1: float = Field(default=0.9, ge=0, lt=1)
    adam_beta2: float = Field(default=0.999, ge=0, lt=1)
    optimizer_eps: float = Field(default=1e-8, gt=0, le=1)
    amsgrad: bool = False
    dampening: float = Field(default=0, ge=0, le=1)
    nesterov: bool = False
    rmsprop_alpha: float = Field(default=0.99, ge=0, lt=1)
    centered: bool = False
    label_smoothing: float = Field(default=0, ge=0, lt=1)
    min_lr: float = Field(default=0, ge=0, le=10)
    scheduler_patience: int = Field(default=3, ge=0, le=1000)
    scheduler: Literal["none", "StepLR", "CosineAnnealingLR", "ExponentialLR", "ReduceLROnPlateau"] = "none"
    step_size: int = Field(default=5, ge=1)
    gamma: float = Field(default=0.5, gt=0, le=1)
    patience: int = Field(default=5, ge=1, le=1000)
    seed: int = Field(default=42, ge=0, le=2147483647)
    device: str = Field(default="auto", pattern=r"^(auto|cpu|cuda:[0-9]+)$")
    gpu_memory_fraction: float = Field(default=0.2, ge=0.05, le=0.9)
    precision: Literal["fp32", "amp"] = "fp32"
    deterministic: bool = True
    loader_workers: int = Field(default=0, ge=0, le=8)
    drop_last: bool = False
    cpu_threads: int = Field(default=2, ge=1, le=64)
    timeout_seconds: int = Field(default=3600, ge=10, le=604800)
    grad_clip: float = Field(default=1, gt=0, le=100)
    loss: Literal['CrossEntropyLoss','auto','cross_entropy','nll','focal','bce_logits','multi_margin','kl_div','custom']='CrossEntropyLoss'

    @model_validator(mode="after")
    def optimizer_options(self):
        if self.optimizer=='SGD' and self.nesterov and (self.momentum<=0 or self.dampening!=0):
            raise ValueError('Nesterov cần momentum > 0 và dampening = 0.')
        if self.min_lr>self.learning_rate:raise ValueError('min_lr không được lớn hơn learning_rate.')
        if self.scheduler=='ReduceLROnPlateau' and self.gamma>=1:raise ValueError('ReduceLROnPlateau cần gamma < 1.')
        return self


class Project(Strict):
    schema_version: Literal["1.0"] = "1.0"
    name: str = Field(default="Dự án mới", min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)
    task: Literal["classification", "regression"] = "classification"
    backend: Literal["sklearn", "torch"] = "sklearn"
    classes: int = Field(default=3, ge=2, le=1000)
    data: DataConfig = Field(default_factory=DataConfig)
    training: Training = Field(default_factory=Training)
    graphs: list[Graph]

