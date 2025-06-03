## Repository for the research project "Accelerating matrix-vector products using entropy coding on the GPU" ##

## Structure of the project:

- python script `create_matfiles.py` for creating required compressed and raw `.bin` 
- `cuBLAS_baseline_float/` and `cuBLAS_baseline_int/` directories contain baselines
- `matvans/` contains proposed alternative 

## creating and writing dummy-weights files
First make sure that you have created your `*matrices*.bin` files by using `create_matfiles.py` or `mock-data-old-adapted.ipynb`. To run `create_matfiles.py` simply execute 
```bash
python create_matfiles.py
```
in the terminal and adapt the variables to your needs.

## Compiling your own binaries
First, adapt the path in the respective kernel to the location of your dummy weights file created in step beforehand.
 To achieve this change the `filename` string variable the respective`*.cu` files.
To compile your own binary run inside one of the three folders
```bash
mkdir build && cd build && cmake .. 
```
if you want to debug the kernel run `cmake` with
```bash
mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Debug ..
```
after a succesful `make` execute `./baseline` or `./matvans` 

To check if a binary includes the required debug symbols, run:
```bash
file binary_file
```


## further setup notes
CUDA kernels can be compiled manually with `nvcc` or using `cmake`. When using a `CMakeLists.txt`, linking to **cuBLAS** or **cuBLASLt** is required. To enable kernel debugging in VSCode, a `launch.json` file needs to be placed in the `.vscode/` directory. The program should be compiled manually (by running `make`) before debugging to include the latest changes. Alternatively, one can define a `tasks.json` to automate the build step. Debugging is very similar to host-based programming, except you can specify the inspected thread and block using the format block

```
(a,0,0) thread (x,0,0)
```

A more manual approach involves placing 
`printf("...",...)` statements inside conditionals like `if (... && threadIdx.x == x).`
