"""Offline introspection of a .tflite file.
Returns input/output shape & dtype, op count, tensor count, parameter bytes."""
import sys
import json
from pathlib import Path

import tflite

_TYPE_NAMES = {v: k for k, v in vars(tflite.TensorType).items() if not k.startswith('_')}


def meta(path):
    buf = Path(path).read_bytes()
    model = tflite.Model.GetRootAsModel(buf, 0)
    sg = model.Subgraphs(0)

    def t_info(idx):
        t = sg.Tensors(idx)
        shape = [t.Shape(i) for i in range(t.ShapeLength())]
        return {'shape': shape, 'dtype': _TYPE_NAMES.get(t.Type(), str(t.Type()))}

    inputs = [t_info(sg.Inputs(i)) for i in range(sg.InputsLength())]
    outputs = [t_info(sg.Outputs(i)) for i in range(sg.OutputsLength())]

    op_count = sg.OperatorsLength()
    tensor_count = sg.TensorsLength()
    opcodes = []
    op_names = {v: k for k, v in vars(tflite.BuiltinOperator).items() if not k.startswith('_')}
    for i in range(model.OperatorCodesLength()):
        oc = model.OperatorCodes(i)
        opcodes.append(op_names.get(oc.BuiltinCode() or oc.DeprecatedBuiltinCode(), '?'))

    param_bytes = 0
    for i in range(model.BuffersLength()):
        b = model.Buffers(i)
        if b.DataLength():
            param_bytes += b.DataLength()

    return {
        'inputs': inputs,
        'outputs': outputs,
        'op_count': op_count,
        'tensor_count': tensor_count,
        'unique_opcodes': sorted(set(opcodes)),
        'param_bytes': param_bytes,
        'file_bytes': len(buf),
    }


if __name__ == '__main__':
    print(json.dumps(meta(sys.argv[1]), indent=2))
