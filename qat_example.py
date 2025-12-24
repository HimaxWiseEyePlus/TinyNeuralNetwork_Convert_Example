import argparse
import os
import sys

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

sys.path.insert(1, os.path.join(CURRENT_PATH, '../'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tinynn.converter import TFLiteConverter
from tinynn.graph.quantization.quantizer import QATQuantizer
from tinynn.graph.tracer import model_tracer
from tinynn.prune import OneShotChannelPruner
from tinynn.util.cifar10 import get_dataloader, train_one_epoch, validate
from tinynn.util.train_util import DLContext, get_device, train
from examples.models.cifar10 import mobilenet


def main_worker(args):
    with model_tracer():
        print("###### TinyNeuralNetwork quick start for beginner ######")

        model = mobilenet.Mobilenet()
        model.load_state_dict(torch.load(mobilenet.DEFAULT_STATE_DICT, weights_only=False))

        device = get_device()
        model.to(device=device)

        # Provide a viable input for the model
        dummy_input = torch.rand((1, 3, 224, 224))

        context = DLContext()
        context.device = device
        context.train_loader, context.val_loader = get_dataloader(args.data_path, 224, args.batch_size, args.workers,download=True)

        print("\n###### Start preparing the model for quantization ######")
        # We provides a QATQuantizer class that may rewrite the graph for and perform model fusion for quantization
        # The model returned by the `quantize` function is ready for QAT training
        quantizer = QATQuantizer(model, dummy_input, work_dir='out_qat')
        qat_model = quantizer.quantize()
    # print(qat_model)
    print("\n###### Start quantization-aware training ######")
    # Move model to the appropriate device
    qat_model.to(device=device)

    context = DLContext()
    context.device = device
    context.train_loader, context.val_loader = get_dataloader(
        args.data_path, 224, args.batch_size * 2 // 3, args.workers
    )
    context.max_epoch = 3
    context.criterion = nn.BCEWithLogitsLoss()
    context.optimizer = torch.optim.SGD(qat_model.parameters(), 0.01, momentum=0.9, weight_decay=5e-4)
    context.scheduler = optim.lr_scheduler.CosineAnnealingLR(context.optimizer, T_max=context.max_epoch + 1, eta_min=0)

    # Quantization-aware training
    train(qat_model, context, train_one_epoch, validate, qat=True, work_dir='out_qat')

    print("\n###### Start converting the model to TFLite ######")
    with torch.no_grad():
        qat_model.eval()
        qat_model.cpu()

        # The step below converts the model to an actual quantized model, which uses the quantized kernels.
        qat_model = quantizer.convert(qat_model)

        # When converting quantized models to TFLite, please ensure the quantization backend is QNNPACK.
        torch.backends.quantized.engine = 'qnnpack'

        # The code section below is used to convert the model to the TFLite format
        converter = TFLiteConverter(qat_model, dummy_input, tflite_path='out_qat/qat_model.tflite',quantize_input_output_type="int8",fuse_quant_dequant=True,quantize_target_type="int8")
        converter.convert()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', metavar='DIR', default="/data/datasets/cifar10", help='path to cifar10 dataset')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=256)

    args = parser.parse_args()
    main_worker(args)
