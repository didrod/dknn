# DKNN
A toy implementation of distributed k nearest neighbors.

## Building & running examples
``` bash
$ git clone https://github.com/didrod/dknn
$ cmake -H./dknn -B./dknn-build -DCMAKE_BUILD_TYPE=Release
$ cmake --build dknn-build --target main -- -j
$ mpiexec -n 4 dknn-build/main dknn/examples/data.csv dknn/examples/q1.csv
```

## Running unit tests
``` bash
$ cmake -H./dknn -B./dknn-build-dbg -DCMAKE_BUILD_TYPE=RelWithDebInfo
$ cmake --build ./dknn-build-dbg --target run_test_single_node
$ cmake --build ./dknn-build-dbg --target run_test_distributed
```
