import numpy as np
import os
import sys
from qonnx.custom_op.registry import getCustomOp

import onnx
import torch

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import finn.util.data_packing as dpk

from brevitas.export import export_qonnx

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup as qonnx_cleanup

from qonnx.transformation.insert_topk import InsertTopK

from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.builder.build_dataflow_config import DataflowBuildConfig, VerificationStepType
from finn.builder.build_dataflow_steps import verify_step, step_tidy_up

from finn.util.pytorch import ToTensor
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.core.datatype import DataType


def custom_step_qonnx_to_finn(model: ModelWrapper, cfg: DataflowBuildConfig):

    model = model.transform(ConvertQONNXtoFINN())

    if VerificationStepType.QONNX_TO_FINN_PYTHON in cfg._resolve_verification_steps():
        verify_step(model, cfg, "qonnx_to_finn_python", need_parent=False)

    return model


def custom_step_tidy_up(model: ModelWrapper, cfg: DataflowBuildConfig):

    inp_name = model.graph.input[0].name
    ishape = model.get_tensor_shape(inp_name)

    # preprocessing: torchvision's ToTensor divides uint8 inputs by 255
    chkpt_preproc_name = os.path.join(cfg.output_dir, "intermediate_models/custom_step_tidy_up_preprocess.onnx")

    totensor_pyt = ToTensor()
    export_qonnx(totensor_pyt, torch.randn(ishape), chkpt_preproc_name)
    qonnx_cleanup(chkpt_preproc_name, out_file=chkpt_preproc_name)

    pre_model = ModelWrapper(chkpt_preproc_name)
    pre_model = pre_model.transform(ConvertQONNXtoFINN())

    # join preprocessing and core model
    model = model.transform(MergeONNXModels(pre_model))

    # add input quantization annotation: UINT8 for all BNN-PYNQ models
    inp_name = model.graph.input[0].name
    model.set_tensor_datatype(inp_name, DataType["UINT8"])

    model = model.transform(InsertTopK(k=1))
    model = step_tidy_up(model, cfg)

    if VerificationStepType.TIDY_UP_PYTHON in cfg._resolve_verification_steps():
        verify_step(model, cfg, "initial_python", need_parent=False)

    # model.save("model-001.onnx")

    return model


def custom_step_gen_tb_and_io(model, cfg):
    sim_output_dir = cfg.output_dir + "/sim"
    os.makedirs(sim_output_dir, exist_ok=True)
    # load the provided input data
    inp_data = np.load("input.npy")
    batchsize = inp_data.shape[0]
    # permute input image from NCHW -> NHWC format (needed by FINN)
    # this example (MNIST) only has 1 channel, which means this doesn't
    # really do anything in terms of data layout changes, but provided for
    # completeness
    inp_data = np.transpose(inp_data, (0, 2, 3, 1))
    # this network is an MLP and takes in flattened input
    inp_data = inp_data.reshape(batchsize, -1)
    # query the parallelism-dependent folded input shape from the
    # node consuming the graph input
    inp_name = model.graph.input[0].name
    inp_node = getCustomOp(model.find_consumer(inp_name))
    inp_shape_folded = list(inp_node.get_folded_input_shape())
    inp_stream_width = inp_node.get_instream_width_padded()
    # fix first dimension (N: batch size) to correspond to input data
    # since FINN model itself always uses N=1
    inp_shape_folded[0] = batchsize
    inp_shape_folded = tuple(inp_shape_folded)
    inp_dtype = model.get_tensor_datatype(inp_name)
    # now re-shape input data into the folded shape and do hex packing
    inp_data = inp_data.reshape(inp_shape_folded)
    #print(inp_dtype)
    #print(inp_stream_width)
    #print(str(inp_data))
    inp_data_packed = dpk.pack_innermost_dim_as_hex_string(
        inp_data, inp_dtype, inp_stream_width, prefix="", reverse_inner=True
    )
    np.savetxt(sim_output_dir + "/input.dat", inp_data_packed, fmt="%s", delimiter="\n")
    # load expected output and calculate folded shape
    exp_out = np.load("expected_output.npy")
    out_name = model.graph.output[0].name
    out_node = getCustomOp(model.find_producer(out_name))
    out_shape_folded = list(out_node.get_folded_output_shape())
    out_stream_width = out_node.get_outstream_width_padded()
    out_shape_folded[0] = batchsize
    out_shape_folded = tuple(out_shape_folded)
    out_dtype = model.get_tensor_datatype(out_name)
    exp_out = exp_out.reshape(out_shape_folded)
    out_data_packed = dpk.pack_innermost_dim_as_hex_string(
        exp_out, out_dtype, out_stream_width, prefix="", reverse_inner=True
    )
    np.savetxt(
        sim_output_dir + "/expected_output.dat",
        out_data_packed,
        fmt="%s",
        delimiter="\n",
    )
    # fill in testbench template
    with open("templates/finn_testbench.template.sv", "r") as f:
        testbench_sv = f.read()
    testbench_sv = testbench_sv.replace("@N_SAMPLES@", str(batchsize))
    testbench_sv = testbench_sv.replace("@IN_STREAM_BITWIDTH@", str(inp_stream_width))
    testbench_sv = testbench_sv.replace("@OUT_STREAM_BITWIDTH@", str(out_stream_width))
    testbench_sv = testbench_sv.replace(
        "@IN_BEATS_PER_SAMPLE@", str(np.prod(inp_shape_folded[:-1]))
    )
    testbench_sv = testbench_sv.replace(
        "@OUT_BEATS_PER_SAMPLE@", str(np.prod(out_shape_folded[:-1]))
    )
    testbench_sv = testbench_sv.replace("@TIMEOUT_CYCLES@", "1000")
    with open(sim_output_dir + "/finn_testbench.sv", "w") as f:
        f.write(testbench_sv)
    # fill in testbench project creator template
    with open("templates/make_sim_proj.template.tcl", "r") as f:
        testbench_tcl = f.read()
    testbench_tcl = testbench_tcl.replace("@STITCHED_IP_ROOT@", "../stitched_ip")
    with open(sim_output_dir + "/make_sim_proj.tcl", "w") as f:
        f.write(testbench_tcl)

    return model

# update import path
cwd = os.getcwd()
import_path = os.path.join(cwd, './training')
import_path = os.path.abspath(import_path)
sys.path.append(import_path)

from models import fc

# import our model
sfc = fc()

model_name = "sfc_1w1a"
platform_name = "fpga"
#platform_name = "Zybo-z7-20"

# load checkpoint
model_filename = os.path.join(import_path, 'checkpoints/sfc_1w1a.pth')
assert os.path.isfile(model_filename)
sfc.load_checkpoint(model_filename)

build_dir = os.environ["FINN_BUILD_DIR"]

# export model into onnx
export_onnx_path = "model.onnx"
export_qonnx(sfc, torch.randn(1, 1, 28, 28), export_onnx_path)
qonnx_cleanup(export_onnx_path, out_file=export_onnx_path)

build_steps = build_cfg.default_build_dataflow_steps + [custom_step_gen_tb_and_io]
build_steps = [custom_step_qonnx_to_finn if s == "step_qonnx_to_finn" else s for s in build_steps]
build_steps = [custom_step_tidy_up if s == "step_tidy_up" else s for s in build_steps]

print(build_steps)

# step_target_fps_parallelization

cfg = build.DataflowBuildConfig(
    steps=build_steps,
    board=platform_name,
    output_dir="output_%s_%s" % (model_name, platform_name),
    #target_fps=5000,
    synth_clk_period_ns=10.0,
    folding_config_file="folding_config.json",
    fpga_part="xc7z020clg400-1",
    shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
    stitched_ip_gen_dcp=False,
    generate_outputs=[
        build_cfg.DataflowOutputType.STITCHED_IP,
    ],
    verify_steps=[
        build_cfg.VerificationStepType.TIDY_UP_PYTHON,
        build_cfg.VerificationStepType.STREAMLINED_PYTHON,
        build_cfg.VerificationStepType.FOLDED_HLS_CPPSIM,
        build_cfg.VerificationStepType.STITCHED_IP_RTLSIM,
    ],
    save_intermediate_models=True,
)

build.build_dataflow_cfg(export_onnx_path, cfg)
