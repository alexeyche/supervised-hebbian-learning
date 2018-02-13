#include <stdio.h>

#ifdef SHL_DLL
	#ifdef SHL_EXPORTS
		#define SHL_API __declspec(dllexport)
	#else
		#define SHL_API __declspec(dllimport)
	#endif
#else		
	#define SHL_API extern 
#endif /* SHL_DLL */

struct LayerConfig;
struct Config;
struct Matrix;

extern "C" {
	
	SHL_API void run_model(Config config, Matrix m);

}


#include <iostream>

#include <Eigen/Dense>

using ui32 = unsigned int;

template <int nrows, int ncols>
using TMatrix = Eigen::Matrix<float, nrows, ncols>;

using TMatrixD = TMatrix<Eigen::Dynamic, Eigen::Dynamic>;


struct LayerConfig {
 	ui32 layer_size;
}; 

struct Config {
 	ui32 layer_num;
  	LayerConfig* layer_configs;
};

struct Matrix {
  	float* data;
  	ui32 nrows; 
  	ui32 ncols;
};

void run_model(Config config, Matrix m) {
	std::cout << config.layer_num << " " << m.nrows << "x" << m.ncols << "\n";

	TMatrixD mmm;	

	for (ui32 i=0; i<m.nrows; ++i) {
		for (ui32 j=0; j<m.ncols; ++j) {
			std::cout << m.data[i * m.ncols + j] << ",";
		}
		std::cout << "\n";
	}

}
