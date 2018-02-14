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
struct TStatFlat;
struct TInputFlat;

struct TStructure {
	int InputSize;
	int LayerSize;
	int OutputSize;
	int BatchSize;
	int LayersNum;
	int SeqLength;
};


extern "C" {
	
	SHL_API int run_model(TConfig c, TInputFlat inputFlat, TStatFlat stat);
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

  	template <int NRows, int NCols>
  	static void FromEigen(TMatrix<NRows, NCols> m, TMatrixFlat* dst) {
  		ENSURE(NRows == dst->NRows, "Rows are not aligned for saving eigen matrix");
  		ENSURE(NCols == dst->NCols, "Cols are not aligned for saving eigen matrix");

  		memcpy(dst->Data, m.data(), sizeof(float) * NRows * NCols);
  	}
};


struct TStatFlat {
	TMatrixFlat Input;
	TMatrixFlat U;
	TMatrixFlat A;
	TMatrixFlat Output;
	TMatrixFlat De;
};

struct TInputFlat {
	TMatrixFlat TrainInput;
	TMatrixFlat TrainOutput;
	TMatrixFlat TestInput;
	TMatrixFlat TestOutput;
};

static constexpr int InputSize = 2;
static constexpr int LayerSize = 25;
static constexpr int OutputSize = 2;
static constexpr int BatchSize = 4;
static constexpr int LayersNum = 1;
static constexpr int SeqLength = 100;

template <int NRows, int NCols>
TMatrix<NRows, NCols> Act(TMatrix<NRows, NCols> x) {
	return x.cwiseMax(0.0);
}


struct TConfig {
	TMatrixFlat F0;
	TMatrixFlat F1;
	double Dt;
 	double SynTau;
 	double FbFactor;
};

struct TInput {
	TMatrix<BatchSize, InputSize*SeqLength> Input;
	TMatrix<BatchSize, OutputSize*SeqLength> Output;

	static TInput FromFlat(TInputFlat s) {
		ENSURE(s.Input.NRows == BatchSize, 
			"Batch size is expected to be " << BatchSize << ", not: `" << s.Input.NRows << "`");
		ENSURE(s.Input.NCols == InputSize*SeqLength, 
			"Input number of columns is expected to be " << InputSize*SeqLength << ", not: `" << s.Input.NCols << "`");


		TInput o;
		o.Input = TMatrixFlat::ToEigen<BatchSize, InputSize*SeqLength>(s.Input);
		o.Output = TMatrixFlat::ToEigen<BatchSize, OutputSize*SeqLength>(s.Output);
		return o;
	}

	static void ToFlat(TInput s, TInputFlat* f) {
		TMatrixFlat::FromEigen(s.Input, &f->Input);
		TMatrixFlat::FromEigen(s.Output, &f->Output);
	}
};


struct TStat {
	TMatrix<BatchSize, InputSize*SeqLength> Input;
	TMatrix<BatchSize, LayerSize*SeqLength> U;
	TMatrix<BatchSize, LayerSize*SeqLength> A;
	TMatrix<BatchSize, OutputSize*SeqLength> Output;
	TMatrix<BatchSize, OutputSize*SeqLength> De;
	
	static TStat FromFlat(TStatFlat s) {
		TStat o;
		o.Input = TMatrixFlat::ToEigen<BatchSize, InputSize*SeqLength>(s.Input);
		o.U = TMatrixFlat::ToEigen<BatchSize, LayerSize*SeqLength>(s.U);
		o.A = TMatrixFlat::ToEigen<BatchSize, LayerSize*SeqLength>(s.A);
		o.Output = TMatrixFlat::ToEigen<BatchSize, OutputSize*SeqLength>(s.Output);
		o.De = TMatrixFlat::ToEigen<BatchSize, OutputSize*SeqLength>(s.De);
		return o;
	}

	static void ToFlat(TStat s, TStatFlat* f) {
		TMatrixFlat::FromEigen(s.Input, &f->Input);
		TMatrixFlat::FromEigen(s.U, &f->U);
		TMatrixFlat::FromEigen(s.A, &f->A);
		TMatrixFlat::FromEigen(s.Output, &f->Output);
		TMatrixFlat::FromEigen(s.De, &f->De);
	}
};


void run_model_impl(
	TConfig c, 
	TInputFlat trainInputFlat, 
	TInputFlat testInputFlat, 
	TStatFlat trainStatFlat, 
	TStatFlat testStatFlat
) {
	TInput trainInput = TInput::FromFlat(trainInputFlat);
	TInput testInput = TInput::FromFlat(testInputFlat);

	TStat trainStat = TStat::FromFlat(trainStatFlat);
	TStat testStat = TStat::FromFlat(testStatFlat);

	TMatrix<InputSize, LayerSize> F0 = \
		TMatrixFlat::ToEigen<InputSize, LayerSize>(c.F0);
	TMatrix<LayerSize, OutputSize> F1 = \
		TMatrixFlat::ToEigen<LayerSize, OutputSize>(c.F1);
		
	TMatrix<BatchSize, InputSize> trainInputSpikesState = \
		TMatrix<BatchSize, InputSize>::Zero();
	TMatrix<BatchSize, InputSize> testInputSpikesState = \
		TMatrix<BatchSize, InputSize>::Zero();

	// TMatrix<BatchSize, LayerSize> layerState = \
	// 	TMatrix<BatchSize, LayerSize>::Zero();

	TMatrix<BatchSize, OutputSize> trainDe = TMatrix<BatchSize, OutputSize>::Zero();
	TMatrix<BatchSize, OutputSize> testDe = TMatrix<BatchSize, OutputSize>::Zero();

	for (ui32 t=0; t<SeqLength; ++t) {
		TMatrix<BatchSize, InputSize> x = input.TrainInput.block<BatchSize, InputSize>(0, t*InputSize);

		inputSpikesState += c.Dt * (x - inputSpikesState) / c.SynTau;

		TMatrix<BatchSize, LayerSize> u = inputSpikesState * F0 + c.FbFactor * de * F1.transpose();
		// layerState += c.Dt * (u - layerState) / c.SynTau;

		TMatrix<BatchSize, LayerSize> a = Act(u);

		TMatrix<BatchSize, OutputSize> uo = a * F1;

		TMatrix<BatchSize, OutputSize> uo_t = \
			input.TrainOutput.block<BatchSize, OutputSize>(0, t*OutputSize);

		de = uo_t - uo;


		stat.Input.block<BatchSize, InputSize>(0, t*InputSize) = inputSpikesState;
		stat.U.block<BatchSize, LayerSize>(0, t*LayerSize) = u;
		stat.A.block<BatchSize, LayerSize>(0, t*LayerSize) = a;
		stat.Output.block<BatchSize, OutputSize>(0, t*OutputSize) = uo;
		stat.De.block<BatchSize, OutputSize>(0, t*OutputSize) = de;
	}

	TStat::ToFlat(stat, &statFlat);
}

int run_model(
	TConfig c, 
	TInputFlat trainInputFlat, 
	TInputFlat testInputFlat, 
	TStatFlat trainStatFlat, 
	TStatFlat testStatFlat
) {
	try {
		run_model_impl(c, trainInputFlat, testInputFlat, trainStatFlat, testStatFlat);
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