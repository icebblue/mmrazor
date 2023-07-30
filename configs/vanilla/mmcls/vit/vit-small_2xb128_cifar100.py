_base_ = [
    'mmrazor::_base_/datasets/mmcls/cifar100_bs128.py',
    'mmrazor::_base_/vanilla_models/vit-small_cifar100.py',
    'mmrazor::_base_/schedules/mmcls/cifar100_bs256.py',
    'mmrazor::_base_/default_runtime.py',
]
test_evaluator = dict(topk=(1, 5))