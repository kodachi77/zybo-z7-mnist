import argparse
import numpy as np
import os
from qonnx.core.datatype import DataType
from mnist_overlay import MNISTOverlay
from pynq.pl_server.device import Device


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Execute FINN-generated accelerator on numpy inputs, or run throughput test"
    )
    parser.add_argument(
        "--exec_mode",
        help='Please select functional verification ("execute") or throughput test ("throughput_test")',
        default="execute",
    )
    parser.add_argument(
        "--batchsize", help="number of samples for inference", type=int, default=1
    )
    parser.add_argument(
        "--bitfile", help='name of bitfile (i.e. "resizer.bit")', default="resizer.bit"
    )
    parser.add_argument(
        "--inputfile",
        help='name(s) of input npy file(s) (i.e. "input.npy")',
        nargs="*",
        type=str,
        default=["input.npy"],
    )
    parser.add_argument(
        "--outputfile",
        help='name(s) of output npy file(s) (i.e. "output.npy")',
        nargs="*",
        type=str,
        default=["output.npy"],
    )

    # parse arguments
    args = parser.parse_args()
    exec_mode = args.exec_mode
    batch_size = args.batchsize
    bitfile = args.bitfile
    inputfile = args.inputfile
    outputfile = args.outputfile
    device = Device.devices[0]

    # instantiate FINN accelerator driver and pass batchsize and bitfile
    accel = MNISTOverlay(bitfile_name=bitfile, batch_size=batch_size)

    # for the remote execution the data from the input npy file has to be loaded,
    # packed and copied to the PYNQ buffer
    if exec_mode == "execute":
        # load desired input .npy file(s)
        ibuf_normal = []
        for ifn in inputfile:
            ibuf_normal.append(np.load(ifn))
        obuf_normal = accel.execute(ibuf_normal)
        if not isinstance(obuf_normal, list):
            obuf_normal = [obuf_normal]
        for o, obuf in enumerate(obuf_normal):
            np.save(outputfile[o], obuf)
    elif exec_mode == "throughput_test":
        # remove old metrics file
        try:
            os.remove("nw_metrics.txt")
        except FileNotFoundError:
            pass
        res = accel.throughput_test()
        file = open("nw_metrics.txt", "w")
        file.write(str(res))
        file.close()
        print("Results written to nw_metrics.txt")
    else:
        raise Exception("Exec mode has to be set to execute or throughput_test")
