from medet import Trainer
from medet.config import Config

# Step 1: 加载配置
cfg = Config()

# 配置数据路径
cfg.dataset.train = '/path/to/train/data'
cfg.dataset.val = '/path/to/val/data'

# 配置模型参数
cfg.model.name = 'resnet50'
cfg.model.num_classes = 10

# 配置训练参数
cfg.train.batch_size = 16
cfg.train.num_epochs = 50
cfg.train.lr = 0.001

# Step 2: 创建训练器
trainer = Trainer(cfg)

# Step 3: 开始训练
trainer.train()

# from mmdet.apis import inference_detector, init_detector, show_result_pyplot
# config = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# checkpoint = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
# model = init_detector(config, checkpoint, device='cuda:0')
# img = 'demo/demo.jpg'
# result = inference_detector(model, img)
# show_result_pyplot(model, img, result, score_thr=0.3)