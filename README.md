

# MLIRod

Welcome to the home page of MLIRod's repository, which contains four folders:

``src`` contains the source code of MLIRod.

``exp`` contains the experiment script of MLIRod, as well as the report of comparison between MLIRod and NNSmith.

``data`` contains the experiment data of MLIRod.

```scripts``` contains the fuzz script of MLIRod.

```example``` contains detailed examples of four mutation rules in MLIRod.

This page mainly describes the ``src`` and ```scripts```, the description of ```exp``` and ```data``` are shown in the README file in the corresponding folder.

# Build MLIRod

**Note that we integrated MLIRod, MLIRSmith, and MLIR compiler infrastructure together into**
```src/mlir-fuzz-smith```
**, you can build all of them together!**

Use the following commands to build MLIRod, MLIRSmith, and MLIR compiler infrastructure together:

```
cd src/mlir-fuzz-smith
mkdir build
cd build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON
cmake --build .
```

---

You can use additional settings:

```
-DCMAKE_C_COMPILER=afl-cc \
-DCMAKE_CXX_COMPILER=afl-c++ \
```
to support AFL instrumentation for edge coverage collection.

---

or use settings:
```
-DCMAKE_C_FLAGS="-g -O0 -fprofile-arcs -ftest-coverage" \
-DCMAKE_CXX_FLAGS="-g -O0 -fprofile-arcs -ftest-coverage" \
-DCMAKE_EXE_LINKER_FLAGS="-g -fprofile-arcs -ftest-coverage -lgcov" \
```
to enable gcov for line coverage collection.

```src/mlir-fuzz-smith``` is developed based on LLVM repository (git version ```eb601430d3d7f45c30ef8d793a45cbcedf910577```), and more detailed information about building MLIR compiler infrastructure can be found in https://mlir.llvm.org/getting_started/

---

To use MLIRod, you can find ```mlirfuzz.py``` in ```script```, and run the following command:

```
python mlirfuzz.py \
--existing_seed_dir ${seed_dir} \
--replace_data_edge ${build}/bin/ReplaceDataEdge \
--replace_control_edge ${build}/bin/ReplaceControlEdge \
--delete_node ${build}/bin/DeleteNode \
--create_node ${build}/bin/CreateNode \
--mlir_opt ${path_of_mlir-opt_you_want_to_test} \
--optfile opt.txt \
--collector ${build}/bin/CollectCoverage \
--clean True > log.txt
```

The ```${seed_dir}``` denotes the path of the existing seed directory.

The ```${build}``` denotes the build path of MLIRod.

You can use ```--help``` to obtain more information about the settings.

