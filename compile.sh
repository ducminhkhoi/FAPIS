FOLDER= #enter your folder here
CUDA_FOLDER= # enter your cuda folder


source "${FOLDER}/miniconda3/bin/activate"
conda activate siamese

rm -rf build
find ./mmdet/ops -name "*.so" -type f -delete


export CC="${FOLDER}/GCC-7.4.0/bin/gcc"
export CXX="${FOLDER}/GCC-7.4.0/bin/g++"
export PATH="${FOLDER}/cmake-3.17.0-Linux-x86_64/bin:$PATH"
export PATH="${FOLDER}/GCC-7.4.0/bin:$PATH"
export PATH="${CUDA_FOLDER}/cuda/cuda-10.0/bin:$PATH"
export CUDA_HOME="${CUDA_FOLDER}/cuda/cuda-10.0"
export LD_LIBRARY_PATH="${FOLDER}/GCC-7.4.0/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="${CUDA_FOLDER}/cuda/cuda-10.0/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="${CUDA_FOLDER}/cuda/cuda-10.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH$"


python setup.py build develop

# cd data 
# ln -s ../../datasets/coco .