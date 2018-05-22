# test-eigen-cuda

## Instructions
- `ssh username@techlab-gpu-nvidiap100-01`
- `cd /data/<username>`
- clone `git clone https://github.com/vkhristenko/eigen-git-mirror`
- clone `git clone https://github.com/vkhristenko/test-eigen-cuda`
- source `source /data/vkhriste/setup.sh`
- `cd test-eigen-cuda`
- `mkdir build; cd build`
- `cmake ../ -DEIGEN_HOME=/data/<username>/eigen-git-mirror`
- `make -j 4`
