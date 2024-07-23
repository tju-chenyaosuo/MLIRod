```data``` is the evaluation experiment result of MLIRod, which contains three folders:

### ```0.bug detection```

```0.bug detection``` contains the detected bugs of MLIRod and compared approaches. The folder name denotes the approach name, and each folder contains five files that record the detected bugs in the experiment.
We de-duplicate the detected bugs according to the first five lines of the crash message for all experiment groups.

### ```1.covered dialects```

```1.covered dialects``` contains the comparison of the covered dialects and operations between MLIRod, MLIRSmith, and $MLIRod^{pass}_{w/o}$. The covered dialects can be easily collected from covered operations.

### ```2.line coverage```

```2.line coverage``` contains 24-hour gcov files of MLIRod and MLIRSmith.
