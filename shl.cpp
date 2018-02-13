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

struct TLayerConfig;
struct TConfig;
struct TMatrixFlat;

struct TStructure {
	int InputSize;
	int LayerSize;
	int OutputSize;
	int BatchSize;
	int LayersNum;
	int SeqLength;
};


extern "C" {
	
	SHL_API int run_model(TConfig c, TMatrixFlat inputSeqFlat, TMatrixFlat outStatFlat);
	SHL_API TStructure get_structure_info();
}


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


using ui32 = unsigned int;

template <int nrows, int ncols>
using TMatrix = Eigen::Matrix<float, nrows, ncols>;

using TMatrixD = TMatrix<Eigen::Dynamic, Eigen::Dynamic>;


struct TMatrixFlat {
  	float* Data;
  	ui32 NRows; 
  	ui32 NCols;
};


static constexpr int InputSize = 4;
static constexpr int LayerSize = 25;
static constexpr int OutputSize = 2;
static constexpr int BatchSize = 4;
static constexpr int LayersNum = 1;
static constexpr int SeqLength = 25;


struct TConfig {
	TMatrixFlat F0;
	double Dt;
 	double SynTau;
};


void run_model_impl(TConfig c, TMatrixFlat inputSeqFlat, TMatrixFlat outStatFlat) {
	ENSURE(inputSeqFlat.NRows == BatchSize, 
		"Batch size is expected to be " << BatchSize << ", not: `" << inputSeqFlat.NRows << "`");
	ENSURE(inputSeqFlat.NCols == InputSize*SeqLength, 
		"Input number of columns is expected to be " << InputSize*SeqLength << ", not: `" << inputSeqFlat.NCols << "`");


	TMatrix<BatchSize, InputSize*SeqLength> inputSeq = Eigen::Map<
		TMatrix<BatchSize, InputSize*SeqLength>
	>(inputSeqFlat.Data, inputSeqFlat.NRows, inputSeqFlat.NCols);


	TMatrix<BatchSize, InputSize*SeqLength> outStat = Eigen::Map<
		TMatrix<BatchSize, InputSize*SeqLength>
	>(outStatFlat.Data, outStatFlat.NRows, outStatFlat.NCols);

	TMatrix<InputSize, LayerSize> F0 = Eigen::Map<
		TMatrix<InputSize, LayerSize>
	>(c.F0.Data, c.F0.NRows, c.F0.NCols);


	TMatrix<BatchSize, InputSize> inputSpikesState = TMatrix<BatchSize, InputSize>::Zero();
	TMatrix<BatchSize, LayerSize> layerState = TMatrix<BatchSize, LayerSize>::Zero();


	for (ui32 t=0; t<SeqLength; ++t) {
		TMatrix<BatchSize, InputSize> x = \
			inputSeq.block<BatchSize, InputSize>(0, t*InputSize);

		inputSpikesState += c.Dt * (x - inputSpikesState) / c.SynTau;

		outStat.block<BatchSize, InputSize>(0, t*InputSize) = inputSpikesState;
	}

	for (ui32 i=0; i<outStatFlat.NRows; ++i) {
		for (ui32 j=0; j<outStatFlat.NCols; ++j) {
			outStatFlat.Data[i * outStatFlat.NCols + j] = outStat(i, j);
		}
	}
}

int run_model(TConfig c, TMatrixFlat inputSeqFlat, TMatrixFlat outStatFlat) {
	try {
		run_model_impl(c, inputSeqFlat, outStatFlat);
		return 0;
	} catch (const std::exception& e) {
		std::cerr << e.what() << "\n";
		return 1;
	}
}

TStructure get_structure_info() {
	return {
		InputSize,
		LayerSize,
		OutputSize,
		BatchSize,
		LayersNum,
		SeqLength
	};
}