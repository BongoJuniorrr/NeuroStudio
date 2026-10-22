"""Validated contracts for supervised task adapters; independent of graph IR."""
from typing import Literal
from pydantic import Field,model_validator
from .core.schema import Strict,MetricSettings

TASK_MODELS={
    'time_series':['gru_forecaster','linear_forecaster'],
    'image_classification':['resnet18','resnet34','resnet50','resnet101','efficientnet_b0','efficientnet_b1','efficientnet_v2_s','mobilenet_v2','mobilenet_v3_small','mobilenet_v3_large','densenet121','vgg16','convnext_tiny','vit_b_16','swin_t'],
    'image_reconstruction':['conv_autoencoder','vae'],
    'text_generation':['causal_gru','seq2seq_gru','gpt2','distilgpt2','hf_causal'],
    'segmentation':['unet'],
    'detection':['fasterrcnn_mobilenet'],
    'text_classification':['bigru','transformer','bert_tiny','bert_tiny_lora','bert_base','distilbert','deberta','hf_classifier'],
    'audio_classification':['spectrogram_cnn'],
}

class TaskTraining(MetricSettings):
    pretrained_path:str=Field(default='',max_length=2048)
    hf_model:str=Field(default='',max_length=512)
    hf_revision:str=Field(default='main',max_length=128)
    loss:str='auto'
    name:str=Field(default='Thử nghiệm mới',min_length=1,max_length=120)
    dataset_id:str=Field(min_length=1,max_length=128)
    task:Literal['image_classification','segmentation','detection','text_classification','audio_classification','image_reconstruction','text_generation','time_series']
    model:str=Field(max_length=100)
    epochs:int=Field(default=3,ge=1,le=1000)
    batch_size:int=Field(default=4,ge=1,le=128)
    learning_rate:float=Field(default=.001,gt=0,le=1)
    weight_decay:float=Field(default=.0001,ge=0,le=1)
    optimizer:Literal['AdamW','Adam','SGD']='AdamW'
    momentum:float=Field(default=.9,ge=0,lt=1)
    device:str=Field(default='auto',pattern=r'^(auto|cpu|cuda:[0-9]+)$')
    memory_fraction:float=Field(default=.2,ge=.05,le=.9)
    image_size:int=Field(default=128,ge=32,le=512)
    max_length:int=Field(default=64,ge=8,le=256)
    max_vocab:int=Field(default=10000,ge=100,le=100000)
    hidden_size:int=Field(default=64,ge=16,le=256,multiple_of=8)
    segmentation_mode:Literal['binary','multiclass']='binary'
    segmentation_loss:Literal['bce','bce_dice']='bce_dice'
    dice_weight:float=Field(default=.5,ge=0,le=1)
    latent_size:int=Field(default=32,ge=4,le=256)
    kl_weight:float=Field(default=.001,ge=0,le=1)
    lora_rank:int=Field(default=8,ge=1,le=64)
    freeze_backbone:bool=False
    pretrained:bool=False
    seed:int=Field(default=42,ge=0,le=2147483647)
    patience:int=Field(default=5,ge=1,le=100)
    grad_clip:float=Field(default=1,gt=0,le=100)
    timeout_seconds:int=Field(default=3600,ge=10,le=86400)
    train_limit:int|None=Field(default=None,ge=4,le=1000000)
    validation_limit:int|None=Field(default=None,ge=2,le=100000)
    @model_validator(mode='after')
    def compatible(self):
        from .engine.objectives import METRICS,LOSSES,DEFAULTS
        if self.loss not in LOSSES.get(self.task,['auto']):raise ValueError('LOSS_TASK: Loss is incompatible with task.')
        if not self.metrics:self.metrics=[DEFAULTS[self.task]]
        if not self.primary_metric:self.primary_metric=DEFAULTS[self.task]
        allowed=set(METRICS[self.task])|set(self.metric_formulas)
        if any(m not in allowed and not(m.startswith('custom_') and self.metric_code) for m in self.metrics) or self.primary_metric not in self.metrics:raise ValueError('METRIC_TASK: Select compatible metrics and one primary metric.')
        if self.loss=='custom' and not self.loss_code.strip():raise ValueError('CUSTOM_LOSS: Define custom_loss(prediction, target).')
        if self.model=='vit_b_16' and self.image_size!=224:raise ValueError('MODEL_IMAGE_SIZE: ViT-B/16 needs image size 224.')
        if self.model not in TASK_MODELS[self.task]:raise ValueError('Mô hình không phù hợp với bài toán đã chọn.')
        if self.task=='detection' and self.batch_size>4:raise ValueError('Detection giới hạn batch size ≤ 4 để giảm áp lực bộ nhớ.')
        if self.freeze_backbone and self.task=='image_classification' and not self.pretrained and not self.pretrained_path:raise ValueError('Đóng băng backbone cần bật pretrained weights.')
        return self


class TaskStudy(Strict):
    candidates:list[TaskTraining]=Field(min_length=2,max_length=8)
    @model_validator(mode='after')
    def same_problem(self):
        first=self.candidates[0]
        for config in self.candidates:
            if (config.primary_metric,config.primary_mode,config.metric_formulas,config.metric_code)!=(first.primary_metric,first.primary_mode,first.metric_formulas,first.metric_code):raise ValueError('STUDY_OBJECTIVE: All candidates must use the same metric objective.')
            if (config.task,config.dataset_id,config.model,config.seed,config.train_limit,config.validation_limit)!=(first.task,first.dataset_id,first.model,first.seed,first.train_limit,first.validation_limit):raise ValueError('STUDY_DATA: Candidates must use the same task, model, seed and data limits.')
        return self
