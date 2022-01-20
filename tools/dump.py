"""
dump model for inference
"""
import os
from statistics import mode
import sys
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import time
import argparse
import megengine as mge
import megengine.functional as F
from megengine import jit
import numpy as np
import megenginelite as mgelite

from edit.utils import Config
from edit.models import build_model
from edit.core.runner import EpochBasedRunner


def parse_args():
    parser = argparse.ArgumentParser(description='Test an editor o(*￣▽￣*)ブ')
    parser.add_argument('config', help='test config file path')
    args = parser.parse_args()
    return args


def dump_flownet(model, graph_name):
    model.eval()

    data1 = mge.Tensor(np.random.random((2, 3, 180, 320)), dtype=np.float32)
    data2 = mge.Tensor(np.random.random((2, 3, 180, 320)), dtype=np.float32)

    @jit.trace(capture_as_const=True)
    def pred_func(data1, data2):
        outputs = model(data1, data2)
        return outputs

    pred_func(data1, data2)
    pred_func.dump(graph_name,
                   arg_names=["tenFirst", "tenSecond"],
                   optimize_for_inference=True,
                   enable_fuse_conv_bias_nonlinearity=True)


def dump_generator(model, graph_name):
    model.eval()

    data1 = mge.Tensor(np.random.random((2, 96, 180, 320)), dtype=np.float32)
    data2 = mge.Tensor(np.random.random((2, 2, 180, 320)), dtype=np.float32)
    data3 = mge.Tensor(np.random.random((2, 3, 180, 320)), dtype=np.float32)

    @jit.trace(capture_as_const=True)
    def pred_func(data1, data2, data3):
        outputs = model(data1, data2, data3)
        return outputs

    pred_func(data1, data2, data3)
    pred_func.dump(
        graph_name,
        arg_names=["hidden", "flow", "nowFrame"],
        optimize_for_inference=True,
        enable_fuse_conv_bias_nonlinearity=True,
    )


def dump_upsample(model, graph_name):
    model.eval()

    data1 = mge.Tensor(np.random.random((1, 96, 180, 320)), dtype=np.float32)
    data2 = mge.Tensor(np.random.random((1, 96, 180, 320)), dtype=np.float32)

    @jit.trace(capture_as_const=True)
    def pred_func(data1, data2):
        out = model.conv4(F.concat([data1, data2], axis=1))
        out = model.reconstruction(out)
        out = model.lrelu(model.upsample1(out))
        out = model.lrelu(model.upsample2(out))
        out = model.lrelu(model.conv_hr(out))
        out = model.conv_last(out)
        return out

    pred_func(data1, data2)
    pred_func.dump(
        graph_name,
        arg_names=["forward_hidden", "backward_hidden"],
        optimize_for_inference=True,
        enable_fuse_conv_bias_nonlinearity=True,
    )


"""
dump three inference model with
(Pdb) hidden.shape
(2, 96, 180, 320)
(Pdb) flow.shape
(2, 2, 180, 320)
(Pdb) now_frame.shape
(2, 3, 180, 320)


(Pdb) forward_hiddens[i].shape
(1, 96, 180, 320)
(Pdb) backward_hiddens[T-i-1].shape
(1, 96, 180, 320)
"""


def dump():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    model = build_model(cfg.model, eval_cfg=cfg.eval_cfg)
    runner = EpochBasedRunner(model=model,
                              optimizers_cfg=cfg.optimizers,
                              work_dir=cfg.work_dir)
    runner.load_checkpoint(cfg.load_from, load_optim=False)
    model.generator.eval()

    dump_flownet(model.generator.flownet, "flownet.mgb")
    dump_generator(model.generator, "generator.mgb")
    dump_upsample(model.generator, "upsample.mgb")


def test_inference_result(path,
                          inps,
                          out,
                          device=mgelite.LiteDeviceType.LITE_CUDA):
    config = mgelite.LiteConfig(device_type=device)
    net = mgelite.LiteNetwork(config=config)
    net.load(path)
    for k, v in inps.items():
        tensor = net.get_io_tensor(k)
        data = np.load(v)
        if data is None:
            assert ("input .npy unavailable")
        tensor.set_data_by_copy(data)

    time_sum = 0
    REPEAT = 50
    print(f"test model {path} ...")

    # warmup
    net.forward()
    net.wait()

    # loop test
    for _ in range(100):
        begin = time.time()
        net.forward()
        net.wait()
        time_sum = time_sum + time.time() - begin
    print(f"avg timecost {time_sum * 1000 / REPEAT} ms")

    tensor = net.get_io_tensor(net.get_all_output_name()[0])
    dt = tensor.to_numpy()
    gt = np.load(out)
    diff = gt - dt
    print(f"max diff {diff.max()}")


"""
save flownet/generator/upsample GT input and output to .npy, use this function to test
"""
def test_inference():
    test_inference_result('flownet.mgb', {
        'tenFirst': 'flownet_in1.npy',
        'tenSecond': 'flownet_in2.npy'
    }, 'flownet_out.npy')

    test_inference_result(
        'generator.mgb', {
            'hidden': 'generator_in1.npy',
            'flow': 'generator_in2.npy',
            'nowFrame': 'generator_in3.npy'
        }, 'generator_out.npy')

    test_inference_result(
        'upsample.mgb', {
            'forward_hidden': 'upsample_in1.npy',
            'backward_hidden': 'upsample_in2.npy'
        }, 'upsample_out.npy')


if __name__ == "__main__":
    # dump flownet
    dump()

    # inference
    # test_inference()
