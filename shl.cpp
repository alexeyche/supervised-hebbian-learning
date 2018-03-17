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


struct TMatrixFlat;

using ui32 = unsigned int;


struct TStructure {
	int InputSize;
	int LayerSize;
	int OutputSize;
	int BatchSize;
	int LayersNum;
	int SeqLength;
};


extern "C" {
	
	SHL_API int run_model(
		ui32 epochs,
		TConfig c,
		TStateFlat trainState,
		TStateFlat testState,
		TDataFlat trainDataFlat, 
		TDataFlat testDataFlat, 
		TStatisticsFlat trainStatFlat, 
		TStatisticsFlat testStatFlat
	);
	SHL_API TStructure get_structure_info();
}



// #define EIGEN_USE_BLAS = 1

#include <iostream>
#include <tuple>
#include <Eigen/Dense>
#include <sstream>
#include <tuple>
#include <utility>

#define ENSURE(cond, error) \
    if(!(cond)) { \
    	std::stringstream ss; \
    	ss << error; \
    	auto s = ss.str(); \
        throw std::invalid_argument(s.c_str()); \
    }\



template <int nrows, int ncols>
using TMatrix = Eigen::Matrix<float, nrows, ncols, Eigen::RowMajor>;

template <int nrows, int ncols>
using TMatrixCM = Eigen::Matrix<float, nrows, ncols, Eigen::ColMajor>;

using TMatrixD = TMatrix<Eigen::Dynamic, Eigen::Dynamic>;


struct TMatrixFlat {
  	float* Data;
  	ui32 NRows; 
  	ui32 NCols;

  	template <int NRows, int NCols>
  	static TMatrix<NRows, NCols> ToEigen(TMatrixFlat flat) {
  		return Eigen::Map<TMatrix<NRows, NCols>>(
  			flat.Data, flat.NRows, flat.NCols
  		);
  	}

  	static TMatrixD ToEigenDynamic(TMatrixFlat flat) {
  		return Eigen::Map<TMatrixD>(flat.Data, flat.NRows, flat.NCols);
  	}

  	template <int NRows, int NCols>
  	static void FromEigen(TMatrix<NRows, NCols> m, TMatrixFlat* dst) {
  		ENSURE(NRows == dst->NRows, "Rows are not aligned for saving eigen matrix");
  		ENSURE(NCols == dst->NCols, "Cols are not aligned for saving eigen matrix");

  		memcpy(dst->Data, m.data(), sizeof(float) * NRows * NCols);
  	}

  	static void FromEigenDynamic(TMatrixD m, TMatrixFlat* dst) {
  		ENSURE(m.rows() == dst->NRows, "Rows are not aligned for saving eigen matrix");
  		ENSURE(m.cols() == dst->NCols, "Cols are not aligned for saving eigen matrix");

  		memcpy(dst->Data, m.data(), sizeof(float) * m.rows() * m.cols());
  	}
};







template <int BatchSize, int InputSize, int LayerSize, int FeedbackSize>
class THiddenLayer {
	THiddenLayer() {

	}

	void Run(TMatrix<BatchSize, InputSize> ff, TMatrix<BatchSize, FeedbackSize> fb) {
		
		TMatrix<BatchSize, LayerSize> dU = (ff * F - U) + fb;

		U += c.Dt * dU / c.TauSyn;
		A = Act(U, c.Threshold);
	}

	TConfig c;
	
	TMatrix<InputSize, LayerSize> F;

	TMatrix<BatchSize, LayerSize> U;
	TMatrix<BatchSize, LayerSize> A;
	
	TMatrixD UStat;
	TMatrixD AStat;
};



