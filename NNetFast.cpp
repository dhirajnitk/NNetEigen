//
#include "stdafx.h"
#if (_MSC_VER )
#pragma once
#pragma warning(push)
#pragma warning(disable: 4996)

#include <direct.h>
#endif
//
// CopyRights<GNU Public License>
//
// Compiled with windows
//set MKL_NUM_THREADS=4
//set OMP_NUM_THREADS=4 
//g++ -O3 -DNDEBUG -std=c++14 NNetFast.cpp -o NNetFast.exe  -I"eigenSrc" \
-I"%mklroot%\include" -I"D:\Software\Software\curl\include" -L"%mklroot%\lib\intel64"\
-L"D:\Software\Software\curl\lib" -L"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.14393.0\um\x64" \
-DCURL_STATICLIB -static  -fopenmp -lmkl_rt -lcurl -lwldap32 -lws2_32

#define EIGEN_USE_MKL_ALL
#include "Eigen/Core"
#include <sys/stat.h>
#include "curl/curl.h"
#include "vector"
#include "iostream"
#include <fstream>
#include <random>
#include <chrono>
using namespace Eigen;
using namespace std;
const char kPathSeparator =
#ifdef _WIN32
        '\\';
#else
        '/';
#endif

struct FtpFile {
    const char *filename;
    FILE *stream;
}; 
auto ReverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

static size_t my_fwrite(void *buffer, size_t size, size_t nmemb, void *stream);
static size_t my_fwrite(void *buffer, size_t size, size_t nmemb, void *stream);
void download_mnist(std::string folder);

void ReadTrainMNIST(std::string folder, float* data, int* labels);


inline MatrixXf sigmoid(MatrixXf& input) {

    return(1.0 + (-input).array().exp()).inverse().matrix();
}

inline MatrixXf sigmoidDerivative(MatrixXf& input) {

    return sigmoid(input).cwiseProduct((1 - sigmoid(input).array()).matrix()) ;
}

inline MatrixXf costDerivative(MatrixXf output, MatrixXf y){
    return (output - y);
}
static const int num_images = 60000;
static const int train_num_images = 50000;
static const int test_num_images = num_images - train_num_images;
static const int rows = 28;
static const int cols = 28;
static const int batch_size = 10;
static const int layers[3] = {784, 30,10};
static const double eta = 3.0;
#include <chrono>       // std::chrono::system_clock

struct BiasWts {
    std::vector<MatrixXf> weights;
    std::vector<MatrixXf> biases;
};
typedef struct BiasWts NNetWts;
NNetWts BWts;

// Mersenne Twister is a pseudorandom number generator but very slow
/*
std::random_device rd;
std::mt19937 gen{rd()};
*/

//Works fine on VS/Intel c++ compiler. mingw implementation is deterministic(zero entropy), so same output for every run of the program.
/*
std::random_device rd;
std::default_random_engine gen{ rd() };//seed_seq is to increase the entropy of the generated sequence initialized from multiple numbers
*/



//if your system does not have a random device then you can use time(0) as a seed to the random_engine

unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
std::default_random_engine gen(seed);


void RandomShuffle(MatrixXf& A, Map<VectorXi> &l_in){
    VectorXi indices = VectorXi::LinSpaced(A.rows(), 0, A.rows());
    shuffle (indices.data(), indices.data() + A.rows(), gen);
    // Changing  original copy
	#pragma omp parallel sections
	{
		#pragma omp section
		A = indices.asPermutation() * A;
		#pragma omp section
		l_in = indices.asPermutation() * l_in;
	}
}

auto  zerosMatrix = [](NNetWts& temp) {
     //#pragma omp  for
       for(int i=1;i<sizeof(layers)/sizeof(layers[0]);i++){
            temp.weights.push_back(MatrixXf::Zero(layers[i-1], layers[i]) );
            temp.biases.push_back(RowVectorXf::Zero(layers[i]));
        }
        return 0;
    };
 MatrixXf feedforward (MatrixXf data) {
    for(int i = 0;i <BWts.biases.size();i++){
            data = (data * BWts.weights[i]) + BWts.biases[i].replicate((int)data.rows(),1);
            
            data  = sigmoid(data);
        }
    return data;
}
void backprop(MatrixXf data, MatrixXf lVec, NNetWts& temp){
    std::vector<MatrixXf> activations;
    activations.push_back(data);
    std::vector<MatrixXf> zs;
    MatrixXf z;
    // Forward pass
    for(int i = 0;i <BWts.biases.size();i++){
            z = (data * BWts.weights[i]) + BWts.biases[i].replicate((int)data.rows(),1);
            zs.push_back(z);
            data  = sigmoid(z);
            activations.push_back(data);
        }
    // Backward pass
    MatrixXf delta = costDerivative(activations[activations.size() -1 ], lVec).cwiseProduct(sigmoidDerivative(zs[zs.size() -1 ]));
    temp.biases[BWts.biases.size() -1 ] = delta.colwise().sum();
    temp.weights[BWts.weights.size() -1 ] = activations[activations.size() -2 ].transpose() * delta ;  
    for ( int layer = 2; layer < sizeof(layers)/sizeof(layers[0]) ; layer++) {
        z = zs[zs.size() -layer];
        delta = delta * BWts.weights[BWts.weights.size() -layer + 1].transpose().cwiseProduct(sigmoidDerivative(z));
         temp.biases[BWts.biases.size() -layer ] = delta.colwise().sum();
         temp.weights[BWts.biases.size() -layer ] =   activations[activations.size() -layer -1  ].transpose() * delta;
    }
}

MatrixXf createLabelMatrix ( VectorXi l_in){
	MatrixXf lVec = MatrixXf::Zero((int)l_in.rows(), 10);
	// divides loop iterations with 
	#pragma omp for
    for (int index = 0 ; index < l_in.rows(); index++){
		lVec(index, (int)l_in(index)) = 1.0;
	}
    return lVec;

}

void sgdMiniBatch(MatrixXf data, VectorXi l_in){
    NNetWts temp;
    zerosMatrix(temp);
    MatrixXf lVec = createLabelMatrix(l_in);
    backprop(data, lVec, temp);
	#pragma omp  for
    for(int i = 0;i <BWts.biases.size();i++){
        BWts.weights[i] =  BWts.weights[i] - (eta/ batch_size)* temp.weights[i];
        BWts.biases[i] =  BWts.biases[i] - (eta/ batch_size)* temp.biases[i];   
    }

}

auto normaldist(float dummy) {
	std::normal_distribution<float> nd(0.0, 1.0);
    return nd(gen);
    }

int main(int argc, char **argv) {
    std::string name = "MnistData";
   // Download is commented. Run it once
	/*
    #if (_MSC_VER) 
    _mkdir(name.c_str());
    #else
    mkdir(name.c_str());
    #endif
    download_mnist(name);
    */
	float * data_ptr = new float[28*28*num_images]{};
    int* labels_ptr = new int[num_images]{};
    ReadTrainMNIST(name, data_ptr, labels_ptr);
	Map<MatrixXf> data(data_ptr, num_images, rows * cols);
	MatrixXf traindata = data.middleRows(0, train_num_images);
	MatrixXf testdata = data.middleRows(train_num_images, test_num_images);
    Map<VectorXi> trainl_in(labels_ptr, train_num_images);
    Map<VectorXi> testl_in(labels_ptr + train_num_images , test_num_images);
    #pragma omp for
    for(int i=1;i<sizeof(layers)/sizeof(layers[0]);i++){
        // Normal distribution weights
        BWts.weights.push_back(MatrixXf::Zero(layers[i-1], layers[i]).unaryExpr(ptr_fun(normaldist)));
        BWts.biases.push_back(RowVectorXf::Zero(layers[i]).unaryExpr(ptr_fun(normaldist)));

        // Naive method to create random float in [-1, 1] then divided by 100
        /*
		BWts.weights.push_back(MatrixXf::Random(layers[i-1], layers[i]) / 100.0 );
		BWts.biases.push_back(RowVectorXf::Random(layers[i]) / 100.0 );
		*/
     
        
    }
    // Number of epochs 
    int epochs =30 ;
    int   accuracy;
    clock_t start = clock();
    for(int i=0;i<epochs;i++){
         RandomShuffle(traindata,trainl_in);
         #pragma omp for
         for (int j = 0 ;j <train_num_images; j+= batch_size){
              if(j + batch_size <train_num_images){
                sgdMiniBatch(traindata.middleRows(j,batch_size),trainl_in.middleRows(j,batch_size));

              }
              else{
                int minBatch = (int)traindata.rows() - j;
                sgdMiniBatch(traindata.middleRows(j,minBatch),trainl_in.middleRows(j,minBatch));
              }
            } 
           
              MatrixXf outputVec= feedforward(testdata);
              MatrixXf::Index   maxIndex;
              VectorXf maxVal(test_num_images);
              VectorXi output(test_num_images);
              for(int k =0;k<test_num_images;++k) {
                    maxVal(k) = outputVec.row(k).maxCoeff( &maxIndex );
                    output(k)= (int)maxIndex - testl_in(k);
                }
                accuracy = (output.array() == 0).count();
              printf("Epoch:%d, %d / %d\n",i,accuracy,test_num_images);
        }
    
    long int time = (clock() - start);
     printf("Total time: %f sec, Accuracy: %f %%\n",(float)time/ CLOCKS_PER_SEC,(float)accuracy/100.0);
    
}

static size_t my_fwrite(void *buffer, size_t size, size_t nmemb, void *stream)
{
    struct FtpFile *out=(struct FtpFile *)stream;
    if(out && !out->stream) {
        /* open file for writing */
        #if (_MSC_VER )
        errno_t err;
        err=fopen_s(&out->stream, out->filename, "wb");
        if(err)
            return -1; /* failure, can't open file to write */
        #else
        out->stream=fopen(out->filename, "wb");
        if(!out->stream)
            return -1; /* failure, can't open file to write */
        #endif
    }
    return fwrite(buffer, size, nmemb, out->stream);
}


inline bool exists(const std::string& name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

void download_mnist(std::string folder) {
    CURL *curl;
    CURLcode res;
    string  data = folder;
    data += kPathSeparator;
    data += "train-images-idx3-ubyte.gz";
    struct FtpFile training_data = {
        data.c_str(), /* name to store the file as if successful */
        NULL
    };
    auto labels = folder;
    labels += kPathSeparator;
    labels += "train-labels-idx1-ubyte.gz";
    struct FtpFile training_labels = {
        labels.c_str(), /* name to store the file as if successful */
        NULL
    };


    curl_global_init(CURL_GLOBAL_DEFAULT);

    curl = curl_easy_init();
    if (curl) {
        /*
        * You better replace the URL with one that works!
        */
        if (! exists(data.substr(0, data.length() - 3))) {
            curl_easy_setopt(curl, CURLOPT_URL,
                "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz");
            /* Define our callback to get called when there's data to be written */
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, my_fwrite);
            /* Set a pointer to our struct to pass to the callback */
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &training_data);
            res = curl_easy_perform(curl);
            if (CURLE_OK != res) {
                /* we failed */
                fprintf(stderr, "curl told us %d\n", res);
            }
        }
        if (! exists(labels.substr(0, data.length() - 3))) {
            curl_easy_setopt(curl, CURLOPT_URL,
                "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz");
            /* Define our callback to get called when there's data to be written */
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, my_fwrite);
            /* Set a pointer to our struct to pass to the callback */
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &training_labels);
            res = curl_easy_perform(curl);
            if (CURLE_OK != res) {
                /* we failed */
                fprintf(stderr, "curl told us %d\n", res);
            }
        }
    }
    /* always cleanup */
    curl_easy_cleanup(curl);

    if (training_data.stream) {
        fclose(training_data.stream); /* close the local file */
        int res = system(("gzip -d " + data).c_str());
    }

    if (training_labels.stream) {
        fclose(training_labels.stream); /* close the local file */
        int res = system(("gzip -d " + labels).c_str());
    }

    curl_global_cleanup();
}

void ReadTrainMNIST(std::string folder, float* data, int* labels) {
	// setNbThreads is used for OpenMP Threads. Use mkl_set_num_threads ( N ) for MKL;
	#pragma omp parallel sections
	{
		#pragma omp section
		{
			std::string file_name1 = folder;
			file_name1 += kPathSeparator;
			file_name1 += "train-images-idx3-ubyte";
			std::ifstream file1(file_name1, std::ios::binary);
			if (file1.is_open())
			{
				int magic_number = 0;
				int number_of_images = 0;
				int n_rows = 0;
				int n_cols = 0;
				file1.read((char*)&magic_number, sizeof(magic_number));
				magic_number = ReverseInt(magic_number);
				file1.read((char*)&number_of_images, sizeof(number_of_images));
				number_of_images = ReverseInt(number_of_images);
				file1.read((char*)&n_rows, sizeof(n_rows));
				n_rows = ReverseInt(n_rows);
				file1.read((char*)&n_cols, sizeof(n_cols));
				n_cols = ReverseInt(n_cols);
				//std::cout << number_of_images << ", " << rows << ", " << cols << std::endl;
				for (int i = 0; i < number_of_images; ++i)
				{
					for (int r = 0; r < n_rows; ++r)
					{
						for (int c = 0; c < n_cols; ++c)
						{
							unsigned char temp = 0;
							file1.read((char*)&temp, sizeof(temp));
							data[(r*cols + c)*num_images + i] = ((float)temp) / float(255.0);

						}
					}
				}
			}
			file1.close();
		}

		#pragma omp section
		{
			std::string file_name2 = folder;
			file_name2 += kPathSeparator;
			file_name2 += "train-labels-idx1-ubyte";
			std::ifstream file2;
			file2.open(file_name2, std::ios::binary);
			if (file2.is_open())
			{
				int magic_number = 0;
				int number_of_images = 0;
				file2.read((char*)&magic_number, sizeof(magic_number));
				magic_number = ReverseInt(magic_number);
				file2.read((char*)&number_of_images, sizeof(number_of_images));
				number_of_images = ReverseInt(number_of_images);
				//std::cout << number_of_images << std::endl;
				for (int i = 0; i < number_of_images; ++i)
				{
					unsigned char temp = 0;
					file2.read((char*)&temp, sizeof(temp));
					labels[i] = (int)temp;
				}
			}
			file2.close();
		}
	}
}

