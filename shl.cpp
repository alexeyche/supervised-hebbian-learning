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
};

struct TInputFlat {
	TMatrixFlat TrainInput;
	TMatrixFlat TrainOutput;
};

static constexpr int InputSize = 2;
static constexpr int LayerSize = 25;
static constexpr int OutputSize = 1;
static constexpr int BatchSize = 4;
static constexpr int LayersNum = 1;
static constexpr int SeqLength = 25;

template <int NRows, int NCols>
TMatrix<NRows, NCols> Act(TMatrix<NRows, NCols> x) {
	return x.cwiseMax(0.0);
}


struct TConfig {
	TMatrixFlat F0;
	TMatrixFlat F1;
	double Dt;
 	double SynTau;
};


struct TStat {
	TMatrix<BatchSize, InputSize*SeqLength> Input;
	TMatrix<BatchSize, LayerSize*SeqLength> U;
	TMatrix<BatchSize, LayerSize*SeqLength> A;
	TMatrix<BatchSize, OutputSize*SeqLength> Output;
	
	static TStat FromFlat(TStatFlat s) {
		TStat o;
		o.Input = TMatrixFlat::ToEigen<BatchSize, InputSize*SeqLength>(s.Input);
		o.U = TMatrixFlat::ToEigen<BatchSize, LayerSize*SeqLength>(s.U);
		o.A = TMatrixFlat::ToEigen<BatchSize, LayerSize*SeqLength>(s.A);
		o.Output = TMatrixFlat::ToEigen<BatchSize, OutputSize*SeqLength>(s.Output);
		return o;
	}

	static void ToFlat(TStat s, TStatFlat* f) {
		TMatrixFlat::FromEigen(s.Input, &f->Input);
		TMatrixFlat::FromEigen(s.U, &f->U);
		TMatrixFlat::FromEigen(s.A, &f->A);
		TMatrixFlat::FromEigen(s.Output, &f->Output);
	}
};

struct TInput {
	TMatrix<BatchSize, InputSize*SeqLength> TrainInput;
	TMatrix<BatchSize, OutputSize*SeqLength> TrainOutput;
	
	static TInput FromFlat(TInputFlat s) {
		ENSURE(s.TrainInput.NRows == BatchSize, 
			"Batch size is expected to be " << BatchSize << ", not: `" << s.TrainInput.NRows << "`");
		ENSURE(s.TrainInput.NCols == InputSize*SeqLength, 
			"Input number of columns is expected to be " << InputSize*SeqLength << ", not: `" << s.TrainInput.NCols << "`");


		TInput o;
		o.TrainInput = TMatrixFlat::ToEigen<BatchSize, InputSize*SeqLength>(s.TrainInput);
		o.TrainOutput = TMatrixFlat::ToEigen<BatchSize, OutputSize*SeqLength>(s.TrainOutput);
		return o;
	}

	static void ToFlat(TInput s, TInputFlat* f) {
		TMatrixFlat::FromEigen(s.TrainInput, &f->TrainInput);
		TMatrixFlat::FromEigen(s.TrainOutput, &f->TrainOutput);
	}
};


void run_model_impl(TConfig c, TInputFlat inputFlat, TStatFlat statFlat) {
	TInput input = TInput::FromFlat(inputFlat);
	TStat stat = TStat::FromFlat(statFlat);

	TMatrix<InputSize, LayerSize> F0 = \
		TMatrixFlat::ToEigen<InputSize, LayerSize>(c.F0);
	TMatrix<LayerSize, OutputSize> F1 = \
		TMatrixFlat::ToEigen<LayerSize, OutputSize>(c.F1);
		
	TMatrix<BatchSize, InputSize> inputSpikesState = \
		TMatrix<BatchSize, InputSize>::Zero();
	TMatrix<BatchSize, LayerSize> layerState = \
		TMatrix<BatchSize, LayerSize>::Zero();


	for (ui32 t=0; t<SeqLength; ++t) {
		TMatrix<BatchSize, InputSize> x = input.TrainInput.block<BatchSize, InputSize>(0, t*InputSize);

		inputSpikesState += c.Dt * (x - inputSpikesState) / c.SynTau;

		TMatrix<BatchSize, LayerSize> u = inputSpikesState * F0;
		// layerState += c.Dt * (u - layerState) / c.SynTau;

		TMatrix<BatchSize, LayerSize> a = Act(u);

		TMatrix<BatchSize, OutputSize> uo = a * F1;

		
		stat.Input.block<BatchSize, InputSize>(0, t*InputSize) = inputSpikesState;
		stat.U.block<BatchSize, LayerSize>(0, t*LayerSize) = u;
		stat.A.block<BatchSize, LayerSize>(0, t*LayerSize) = a;
		stat.Output.block<BatchSize, OutputSize>(0, t*OutputSize) = uo;
	}

	TStat::ToFlat(stat, &statFlat);
}

int run_model(TConfig c, TInputFlat inputFlat, TStatFlat stat) {
	try {
		run_model_impl(c, inputFlat, stat);
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