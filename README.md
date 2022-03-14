# PyTorch Automatic Differentiation


## Introduction

PyTorch automatic differentiation forward and reverse mode using `autograd` and `functorch`.

## Usages

### Build Docker Image

```bash
$ docker build -f docker/pytorch.Dockerfile --no-cache --tag=pytorch:1.11.0 .
```

### Run Docker Container

```bash
$ docker run -it --rm --gpus device=0 --ipc=host -v $(pwd):/workspace pytorch:1.11.0
```

### Run PyTorch Automatic Differentiation


```bash
$ python autograd.py
N: 16, M: 10240, Device: cuda:0
Forward mode jacfwd time: 0.44039 ms
Reverse mode jacrev time: 2.29482 ms
N: 10240, M: 16, Device: cuda:0
Forward mode jacfwd time: 3.49669 ms
Reverse mode jacrev time: 0.41990 ms
N: 16, M: 10240, Device: cpu
Forward mode jacfwd time: 0.56911 ms
Reverse mode jacrev time: 138.44172 ms
N: 10240, M: 16, Device: cpu
Forward mode jacfwd time: 143.29820 ms
Reverse mode jacrev time: 0.38777 ms
```




```
$ python autograd_weights.py
M x N: 16384, M: 1024, Device: cuda:0
Forward mode jacfwd time: 5.58655 ms
Reverse mode jacrev time: 0.42456 ms
M x N: 16384, M: 16, Device: cuda:0
Forward mode jacfwd time: 4.09694 ms
Reverse mode jacrev time: 0.43367 ms
M x N: 16384, M: 1024, Device: cpu
Forward mode jacfwd time: 394.06240 ms
Reverse mode jacrev time: 19.91690 ms
M x N: 16384, M: 16, Device: cpu
Forward mode jacfwd time: 293.15642 ms
Reverse mode jacrev time: 0.35758 ms
```



```
$ python autograd_batch.py
B: 4, N: 16, M: 10240, Device: cuda:0
Forward mode jacfwd time: 0.60135 ms
Reverse mode jacrev time: 8.38698 ms
B: 4, N: 10240, M: 16, Device: cuda:0
Forward mode jacfwd time: 13.83617 ms
Reverse mode jacrev time: 0.53783 ms
B: 4, N: 16, M: 10240, Device: cpu
Forward mode jacfwd time: 1.39016 ms
Reverse mode jacrev time: 569.47133 ms
B: 4, N: 10240, M: 16, Device: cpu
Forward mode jacfwd time: 576.74174 ms
Reverse mode jacrev time: 0.69496 ms
```

## References

* [PyTorch Automatic Differentiation](https://leimao.github.io/blog/PyTorch-Automatic-Differentiation/)
