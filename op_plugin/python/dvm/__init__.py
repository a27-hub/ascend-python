from torch_npu._C import GraphSplitKernel

def kernel(func):
    k = GraphSplitKernel(False)
    func(k)
    k.setup()
    return k