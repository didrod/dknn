# DKNN
A toy implementation of distributed k nearest neighbors.

## Build instruction
* clone this repository. you might want to use the download button on the github webpage.
* change working directory to the path where you just cloned.
* copy-paste the following commands to initialize the build space.
  ``` bash
  $ cmake -H. -Bbuild -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=yes
  ```
* run the following commands to actually build & run.
  ``` bash
  $ cmake --build build --target main
  $ mpiexec -n 4 build/main
  ```

## Running testcase
``` bash
$ cmake -H. -Bbuild -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=yes
$ cmake --build build --target run_test
```

## Running formatter
Copy-paste the following command.

``` bash
$ find . -regex '.*\.\(cpp\|hpp\|c\|h\)' \
    -not -path './build/*' -and -not -path './.*' -exec clang-format -i -style=file {} \;
```
