# NNetEigen 

A fast matrix-based approach for computing the output from a Multi layer perceptron neural network using Stochastic gradient descent  
with MNIST data

### Usage

The code works on visual studio and Mingw. It has MKL support and it can be build with Intel C++ compiler as well. 

G++ command for Mingw 

```
g++ -O3 -DNDEBUG -std=c++14 NNetFast.cpp -o NNetFast.exe  -I"eigenSrc" \
-I"%mklroot%\include" -I"D:\Software\Software\curl\include" -L"%mklroot%\lib\intel64"\
-L"D:\Software\Software\curl\lib" -L"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.14393.0\um\x64" \
-DCURL_STATICLIB -static  -fopenmp -lmkl_rt -lcurl -lwldap32 -lws2_32
```

## Reference

Neural Networks and Deep Learning  http://neuralnetworksanddeeplearning.com/chap2.html  via "Michael Nielson"
