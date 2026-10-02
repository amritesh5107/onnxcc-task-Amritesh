from pathlib import Path
import numpy as np
import onnx
from onnx import helper,numpy_helper

seed=67
rng=np.random.default_rng(seed)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures"

FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

# Model parameters. Here (a,b) means an a*b matrix.
W1 = rng.standard_normal((4, 8)).astype(np.float32)
B1 = rng.standard_normal(8).astype(np.float32)

W2 = rng.standard_normal((8, 2)).astype(np.float32)
B2 = rng.standard_normal(2).astype(np.float32)


input_data = rng.standard_normal((1, 4)).astype(np.float32)

# Model inputs and outputs
#To make 1*2 matrix we multiply 1*4 with 4*8 and 8*2
X = helper.make_tensor_value_info(
    "X", onnx.TensorProto.FLOAT, [1, 4]
)

Y = helper.make_tensor_value_info(
    "Y", onnx.TensorProto.FLOAT, [1, 2]
)

#This initializes the tensor. Python variable name is W1_init but onnx tensor name is W1??
W1_init = numpy_helper.from_array(W1, name="W1")
B1_init = numpy_helper.from_array(B1, name="B1")

W2_init = numpy_helper.from_array(W2, name="W2")
B2_init = numpy_helper.from_array(B2, name="B2")


# Model nodes
# Basic structure is: Operation type,Inputs,Output variable,Node name
nodes = [
    helper.make_node(
        "MatMul",
        ["X", "W1"],
        ["matmul1_out"],
        name="MatMul1",
    ),
    helper.make_node(
        "Add",
        ["matmul1_out", "B1"],
        ["add1_out"],
        name="Add1",
    ),
    helper.make_node(
        "Relu",
        ["add1_out"],
        ["hidden"],
        name="Relu1",
    ),
    helper.make_node(
        "MatMul",
        ["hidden", "W2"],
        ["matmul2_out"],
        name="MatMul2",
    ),
    helper.make_node(
        "Add",
        ["matmul2_out", "B2"],
        ["add2_out"],
        name="Add2",
    ),
    helper.make_node(
        "Relu",
        ["add2_out"],
        ["Y"],
        name="Relu2",
    ),
]

#It is kind of a flow where the input goes through many different nodes in some
# order to produce some output.
graph = helper.make_graph(
    nodes, #all our operations
    "tiny_mlp", #graph's name
    [X], #inputs
    [Y], #outputs
    initializer=[
        W1_init,
        B1_init,
        W2_init,
        B2_init,
    ],
)

# Build the ONNX model
model = helper.make_model(
    graph,
    opset_imports=[helper.make_opsetid("", 13)], #which version of ONNX operator model shd be used
    producer_name="onnxcc-test", #Which program generated this model??
)

onnx.checker.check_model(model)

# Save the ONNX model
model_path = FIXTURE_DIR / "tiny_mlp.onnx"
onnx.save(model, model_path)

# Save the input as raw float32 bytes
input_path = FIXTURE_DIR / "input.bin"
input_data.tofile(input_path)

