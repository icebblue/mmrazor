_base_ = [
    'mmrazor::_base_/datasets/mmcls/cifar100_bs128.py',
    'mmrazor::_base_/vanilla_models/resnet50_cifar100.py',
    'mmrazor::_base_/schedules/mmcls/cifar100_bs256.py',
    'mmrazor::_base_/default_runtime.py',
]
model_path = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth'
test_evaluator = dict(topk=(1, 5))