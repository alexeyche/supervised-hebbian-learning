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
struct TConfig;
struct TLayerState;
struct TData;

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
		TLayerState* layerStates,
		TConfig c,
		TData trainData,
		TData testData
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

template<typename T>
std::function<T> make_function(T *t) {
  return { t };
}


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

  	static TMatrixD ToEigenDynamic(TMatrixFlat flat, ui32 offset=0) {
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


enum EActivation {
	RELU = 0,
	SIGMOID = 1
};

template <int NRows, int NCols>
TMatrix<NRows, NCols> Relu(TMatrix<NRows, NCols> x) {
	return x.array().cwiseMax(0.0); //.cwiseMin(1.0);
}

template <int NRows, int NCols>
TMatrix<NRows, NCols> ReluDeriv(TMatrix<NRows, NCols> x, double threshold) {
	return x.unaryExpr(
		[threshold](const float xv) { 
			return xv > static_cast<float>(threshold) ? 1.0f : 0.0f;
		}
	);
}

template <int NRows, int NCols>
TMatrix<NRows, NCols> Sigmoid(TMatrix<NRows, NCols> x) {
	return 1.0/(1.0 + (-x.array()).exp());
}


struct TConfig {
	double Dt;
 	double LearningRate;
 	ui32 FeedbackDelay;
};


static constexpr int InputSize = 2;
static constexpr int LayerSize = 30;
static constexpr int OutputSize = 2;
static constexpr int BatchSize = 4;
static constexpr int LayersNum = 2;
static constexpr int SeqLength = 50;

struct TData {
	TMatrixFlat X;
	TMatrixFlat Y;

	TMatrixD ReadInputBatch(ui32 bi) {
		return TMatrixFlat::ToEigenDynamic(X, bi);
	}
	TMatrixD ReadOutputBatch(ui32 bi) {
		return TMatrixFlat::ToEigenDynamic(Y, bi);
	}
};

struct TLayerState {
	double TauSoma;
 	double TauMean;
 	double ApicalGain;
 	EActivation Act;

	TMatrixFlat F;
	TMatrixFlat UStat;
	TMatrixFlat AStat;
};


template <int InputSize, int LayerSize>
class TLayer {
public:
	TLayer(TLayerState s0, TConfig c0): c(c0), s(s0) {
		UStat = TMatrixFlat::ToEigenDynamic(s.UStat);
		AStat = TMatrixFlat::ToEigenDynamic(s.AStat);
		F = TMatrixFlat::ToEigen<InputSize, LayerSize>(s.F);

		U = TMatrix<BatchSize, LayerSize>::Zero();
		A = TMatrix<BatchSize, LayerSize>::Zero();

		if (s.Act == RELU) {
			Act = make_function(&Relu<BatchSize, LayerSize>);
		} else
		if (s.Act == SIGMOID) {
			Act = make_function(&Sigmoid<BatchSize, LayerSize>);
		} else {
			ENSURE(0, "Failed to find activation function #" << s.Act);
		}
	}

	~TLayer() {
		TMatrixFlat::FromEigenDynamic(UStat, &s.UStat);
		TMatrixFlat::FromEigenDynamic(AStat, &s.AStat);
	}

	auto& Run(ui32 t, TMatrix<BatchSize, InputSize> ff, TMatrix<BatchSize, LayerSize> fb) {
		
		TMatrix<BatchSize, LayerSize> dU = ff * F + fb - U;

		U += c.Dt * dU / s.TauSoma;
		A = Act(U);

		UStat.block(0, t*LayerSize, BatchSize, LayerSize) = U;
		AStat.block(0, t*LayerSize, BatchSize, LayerSize) = A;

		return A;
	}

	TConfig c;
	TLayerState s;

	TMatrix<InputSize, LayerSize> F;

	TMatrix<BatchSize, LayerSize> U;
	TMatrix<BatchSize, LayerSize> A;
	
	TMatrixD UStat;
	TMatrixD AStat;
	std::function<TMatrix<BatchSize, LayerSize>(TMatrix<BatchSize, LayerSize>)> Act;
};


using TNet = std::tuple<
	TLayer<InputSize, LayerSize>,
	TLayer<LayerSize, OutputSize>
>;


void run_over_batch(TNet& net, TMatrixD inputData, TMatrixD outputData) {
	TMatrixD deSeq = TMatrixD::Zero(BatchSize, OutputSize*SeqLength);
	TMatrix<BatchSize, OutputSize> zeros;

	for (ui32 t=0; t < SeqLength; ++t) {
		TMatrix<BatchSize, InputSize> x = \
			inputData.block<BatchSize, InputSize>(0, t*InputSize);
		TMatrix<BatchSize, OutputSize> y = \
			outputData.block<BatchSize, OutputSize>(0, t*OutputSize);
		TMatrix<BatchSize, OutputSize> de = \
			deSeq.block<BatchSize, OutputSize>(0, t*OutputSize);

		auto& a0 = std::get<0>(net).Run(t, x, de * std::get<1>(net).F.transpose());
		auto& a1 = std::get<1>(net).Run(t, a0, zeros);
	}
}

int run_model(
	ui32 epochs,
	TLayerState* layerStates,
	TConfig c,
	TData trainData,
	TData testData
) {

	TNet net = std::make_tuple(
		std::tuple_element<0, TNet>::type(layerStates[0], c),
		std::tuple_element<1, TNet>::type(layerStates[1], c)
	);

	// TLayer<TrainBatchSize, InputSize, OutputSize> l0(layerStates[0], c);
	// TLayer<TrainBatchSize, OutputSize, 0> l1(layerStates[1], c);

	for (ui32 e=0; e<epochs; ++e) {
		try {
			ui32 bi = 0;
			run_over_batch(
				net, 
				trainData.ReadInputBatch(bi), 
				trainData.ReadOutputBatch(bi)
			);

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
