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
struct TStatisticsFlat;
struct TDataFlat;
struct TStateFlat;

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

  	template <int NRows, int NCols>
  	static void FromEigen(TMatrix<NRows, NCols> m, TMatrixFlat* dst) {
  		ENSURE(NRows == dst->NRows, "Rows are not aligned for saving eigen matrix");
  		ENSURE(NCols == dst->NCols, "Cols are not aligned for saving eigen matrix");

  		memcpy(dst->Data, m.data(), sizeof(float) * NRows * NCols);
  	}
};


struct TStatisticsFlat {
	TMatrixFlat Input;
	TMatrixFlat U;
	TMatrixFlat A;
	TMatrixFlat dA;
	TMatrixFlat Output;
	TMatrixFlat De;
	TMatrixFlat dF0;
};

struct TDataFlat {
	TMatrixFlat Input;
	TMatrixFlat Output;
};

struct TStateFlat {
	TMatrixFlat A0m;
	TMatrixFlat dF0;
	TMatrixFlat dF1;
};


static constexpr int InputSize = 2;
static constexpr int LayerSize = 30;
static constexpr int OutputSize = 2;
static constexpr int BatchSize = 4;
static constexpr int LayersNum = 1;
static constexpr int SeqLength = 50;

template <int NRows, int NCols>
TMatrix<NRows, NCols> Act(TMatrix<NRows, NCols> x) {
	return x.cwiseMax(0.0); //.cwiseMin(1.0);
}

template <int NRows, int NCols>
TMatrix<NRows, NCols> ActDeriv(TMatrix<NRows, NCols> x) {
	return x.unaryExpr(
		[](const float xv) { 
			return xv > 0.0f ? 1.0f : 0.0f;
		}
	);
}

struct TConfig {
	TMatrixFlat F0;
	TMatrixFlat F1;
	double Dt;
 	double TauSyn;
 	double TauMean;
 	double FbFactor;
 	double LearningRate;
 	double Lambda;
};



struct TData {
	TMatrix<BatchSize, InputSize*SeqLength> Input;
	TMatrix<BatchSize, OutputSize*SeqLength> Output;

	static TData FromFlat(TDataFlat s) {
		ENSURE(s.Input.NRows == BatchSize, 
			"Batch size is expected to be " << BatchSize << ", not: `" << s.Input.NRows << "`");
		ENSURE(s.Input.NCols == InputSize*SeqLength, 
			"Input number of columns is expected to be " << InputSize*SeqLength << ", not: `" << s.Input.NCols << "`");

		ENSURE(s.Output.NRows == BatchSize, 
			"Batch size is expected to be " << BatchSize << ", not: `" << s.Output.NRows << "`");
		ENSURE(s.Output.NCols == OutputSize*SeqLength, 
			"Input number of columns is expected to be " << OutputSize*SeqLength << ", not: `" << s.Output.NCols << "`");

		TData o;
		o.Input = TMatrixFlat::ToEigen<BatchSize, InputSize*SeqLength>(s.Input);
		o.Output = TMatrixFlat::ToEigen<BatchSize, OutputSize*SeqLength>(s.Output);
		return o;
	}

	static void ToFlat(TData s, TDataFlat* f) {
		TMatrixFlat::FromEigen(s.Input, &f->Input);
		TMatrixFlat::FromEigen(s.Output, &f->Output);
	}
};


struct TStatistics {
	TMatrix<BatchSize, InputSize*SeqLength> Input;
	TMatrix<BatchSize, LayerSize*SeqLength> U;
	TMatrix<BatchSize, LayerSize*SeqLength> A;
	TMatrix<BatchSize, LayerSize*SeqLength> dA;
	TMatrix<BatchSize, OutputSize*SeqLength> Output;
	TMatrix<BatchSize, OutputSize*SeqLength> De;
	TMatrix<SeqLength, InputSize*LayerSize> dF0;
	
	static TStatistics FromFlat(TStatisticsFlat s) {
		TStatistics o;
		o.Input = TMatrixFlat::ToEigen<BatchSize, InputSize*SeqLength>(s.Input);
		o.U = TMatrixFlat::ToEigen<BatchSize, LayerSize*SeqLength>(s.U);
		o.A = TMatrixFlat::ToEigen<BatchSize, LayerSize*SeqLength>(s.A);
		o.dA = TMatrixFlat::ToEigen<BatchSize, LayerSize*SeqLength>(s.dA);
		o.Output = TMatrixFlat::ToEigen<BatchSize, OutputSize*SeqLength>(s.Output);
		o.De = TMatrixFlat::ToEigen<BatchSize, OutputSize*SeqLength>(s.De);
		o.dF0 = TMatrixFlat::ToEigen<SeqLength, InputSize*LayerSize>(s.dF0);
		return o;
	}

	static void ToFlat(TStatistics s, TStatisticsFlat* f) {
		TMatrixFlat::FromEigen(s.Input, &f->Input);
		TMatrixFlat::FromEigen(s.U, &f->U);
		TMatrixFlat::FromEigen(s.A, &f->A);
		TMatrixFlat::FromEigen(s.dA, &f->dA);
		TMatrixFlat::FromEigen(s.Output, &f->Output);
		TMatrixFlat::FromEigen(s.De, &f->De);
		TMatrixFlat::FromEigen(s.dF0, &f->dF0);
	}
};


struct TState {
	TMatrix<BatchSize, LayerSize> A0m;
	TMatrix<InputSize, LayerSize> dF0;
	TMatrix<LayerSize, OutputSize> dF1;
	
	static TState FromFlat(TStateFlat s) {
		TState o;
		o.A0m = TMatrixFlat::ToEigen<BatchSize, LayerSize>(s.A0m);
		o.dF0 = TMatrixFlat::ToEigen<InputSize, LayerSize>(s.dF0);
		o.dF1 = TMatrixFlat::ToEigen<LayerSize, OutputSize>(s.dF1);
		return o;
	}

	static void ToFlat(TState s, TStateFlat* f) {
		TMatrixFlat::FromEigen(s.A0m, &f->A0m);
		TMatrixFlat::FromEigen(s.dF0, &f->dF0);
		TMatrixFlat::FromEigen(s.dF1, &f->dF1);
	}
};



void run_model_impl(
	TConfig c, 
	TStateFlat sf,
	TDataFlat dataFlat, 
	TStatisticsFlat statsFlat,
	double fbFactor,
	bool learn
) {
	TData data = TData::FromFlat(dataFlat);
	TStatistics stats = TStatistics::FromFlat(statsFlat);
	TState s = TState::FromFlat(sf);

	TMatrix<InputSize, LayerSize> F0 = \
		TMatrixFlat::ToEigen<InputSize, LayerSize>(c.F0);
	TMatrix<LayerSize, OutputSize> F1 = \
		TMatrixFlat::ToEigen<LayerSize, OutputSize>(c.F1);
		

	TMatrix<BatchSize, InputSize> inputSpikesState = \
		TMatrix<BatchSize, InputSize>::Zero();

	TMatrix<BatchSize, LayerSize> u = \
		TMatrix<BatchSize, LayerSize>::Zero();

	TMatrix<BatchSize, OutputSize> de = TMatrix<BatchSize, OutputSize>::Zero();
	
	s.dF0 = TMatrix<InputSize, LayerSize>::Zero();
	s.dF1 = TMatrix<LayerSize, OutputSize>::Zero();
	TMatrix<BatchSize, LayerSize> A0 = TMatrix<BatchSize, LayerSize>::Zero();

	for (ui32 t=0; t<SeqLength; ++t) {
		TMatrix<BatchSize, InputSize> x = data.Input.block<BatchSize, InputSize>(0, t*InputSize);

		// inputSpikesState = x;

		TMatrix<BatchSize, InputSize> feedforward = x;
		// TMatrix<BatchSize, InputSize> feedforward = x - A0 * F0.transpose();

		// inputSpikesState += c.Dt * (x - inputSpikesState) / c.TauSyn;

		TMatrix<BatchSize, LayerSize> du = \
			(feedforward * F0 - u) + \
			(fbFactor * de * F1.transpose() - u);
		
		u += c.Dt * du / c.TauSyn;

		// layerState += c.Dt * (u - layerState) / c.TauSyn;

		A0 = Act(u);

		TMatrix<BatchSize, OutputSize> uo = A0 * F1;

		TMatrix<BatchSize, OutputSize> uo_t = \
			data.Output.block<BatchSize, OutputSize>(0, t*OutputSize);

		de = uo_t - uo;


		TMatrix<InputSize, LayerSize> dF0t = \
			feedforward.transpose() * ((de * F1.transpose()).array() * (ActDeriv(A0).array())).matrix();

		// TMatrix<InputSize, LayerSize> dF0t = \
		// 	feedforward.transpose() * A0;
		
		// TMatrix<InputSize, LayerSize> dF0t = \
			// feedforward.transpose() * ((A0.array() - c.Lambda)).matrix();
		
		// TMatrix<InputSize, LayerSize> dF0t = \
		// 	feedforward.transpose() * ((A0.array() - s.A0m.array())).matrix();

		// TMatrix<InputSize, LayerSize> dF0t = \
			// feedforward.transpose() * (A0. array() * (A0.array() - s.A0m.array())).matrix();


		// TMatrix<InputSize, LayerSize> dF0t = \
			// (feedforward.transpose() * A0) - (c.Lambda * F0.array()).matrix();

		// TMatrix<InputSize, LayerSize> dF0t = \
			// ((feedforward - A0 * F0.transpose()).transpose() * A0);
		// TMatrix<InputSize, LayerSize> dF0t = \
		// 	(feedforward.transpose() * (a.array() - 0.02).matrix());


		TMatrix<LayerSize, OutputSize> dF1t = \
			A0.transpose() * de;


		s.dF0 += dF0t / SeqLength;
		s.dF1 += dF1t / SeqLength;

		s.A0m += (A0 - s.A0m)/c.TauMean;

		stats.Input.block<BatchSize, InputSize>(0, t*InputSize) = inputSpikesState;
		stats.U.block<BatchSize, LayerSize>(0, t*LayerSize) = u;
		stats.A.block<BatchSize, LayerSize>(0, t*LayerSize) = A0;
		stats.Output.block<BatchSize, OutputSize>(0, t*OutputSize) = uo;
		stats.De.block<BatchSize, OutputSize>(0, t*OutputSize) = de;
		stats.dF0.row(t) = Eigen::Map<TMatrix<1, InputSize*LayerSize>>(dF0t.data());
		stats.dA.block<BatchSize, LayerSize>(0, t*LayerSize) = (A0.array() - s.A0m.array()).matrix();
	}
	
	// if (learn) {
	// 	F0 += c.LearningRate * s.dF0;
	// 	// F0 = (F0.array().rowwise())/(F0.colwise().lpNorm<2>().array());
	// }

	TMatrixFlat::FromEigen(F0, &c.F0);
	TStatistics::ToFlat(stats, &statsFlat);
	TState::ToFlat(s, &sf);
}

int run_model(
	ui32 epochs,
	TConfig c, 
	TStateFlat trainState,
	TStateFlat testState,
	TDataFlat trainDataFlat, 
	TDataFlat testDataFlat, 
	TStatisticsFlat trainStatFlat, 
	TStatisticsFlat testStatFlat
) {
	for (ui32 e=0; e<epochs; ++e) {
		try {
			run_model_impl(c, trainState, trainDataFlat, trainStatFlat, c.FbFactor, true);
			run_model_impl(c, testState, testDataFlat, testStatFlat, 0.0, false);
		} catch (const std::exception& e) {
			std::cerr << e.what() << "\n";
			return 1;
		}		
	}
	return 0;
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