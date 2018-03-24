#include <stdio.h>

#define SHL_API extern

struct TMatrixFlat;
struct TNetConfig;
struct TLayerConfig;
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
		TLayerConfig* layerStates,
		TNetConfig c,
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

#define CheckSizeMatch(m, mf) \
	ENSURE(m.rows() == mf.NRows,  \
		"Rows are not aligned in the matrix for " << #m << ": " \
			<< mf.NRows << " != " << m.rows() << " (latter is expected)");\
	ENSURE(m.cols() == mf.NCols, \
		"Cols are not aligned in the matrix for " << #m << ": " \
			<< mf.NCols << " != " << m.cols() << " (latter is expected)");


struct TMatrixFlat {
  	float* Data;
  	ui32 NRows; 
  	ui32 NCols;

  	template <int NRows, int NCols>
  	static TMatrix<NRows, NCols> ToEigen(TMatrixFlat flat) {
  		ENSURE(flat.NRows == NRows, "Rows do not match while reading flat matrix from the outside");
  		ENSURE(flat.NCols == NCols, "Cols do not match while reading flat matrix from the outside");

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

enum EActivation {
	RELU = 0,
	SIGMOID = 1
};

template <int NRows, int NCols>
TMatrix<NRows, NCols> Relu(TMatrix<NRows, NCols> x) {
	return x.array().cwiseMax(0.0); //.cwiseMin(1.0);
}

template <int NRows, int NCols>
TMatrix<NRows, NCols> ReluDeriv(TMatrix<NRows, NCols> x) {
	return x.unaryExpr(
		[](const float xv) { 
			return xv > static_cast<float>(0.0) ? 1.0f : 0.0f;
		}
	);
}

template <int NRows, int NCols>
TMatrix<NRows, NCols> Sigmoid(TMatrix<NRows, NCols> x) {
	return 1.0/(1.0 + (-x.array()).exp());
}


struct TNetConfig {
	double Dt;
 	double LearningRate;
 	ui32 FeedbackDelay;
 	double OutputTau;

 	TMatrixFlat DeStat;
 	TMatrixFlat YMeanStat;
};


static constexpr int InputSize = 4;
static constexpr int LayerSize = 30;
static constexpr int OutputSize = 2;
static constexpr int BatchSize = 40;
static constexpr int LayersNum = 2;
static constexpr int SeqLength = 50;

struct TData {
	static constexpr ui32 TimeOfDataSpike = 10;

	TMatrixFlat X;
	TMatrixFlat Y;

	TMatrix<BatchSize, InputSize> ReadInput(ui32 bi, ui32 ti) {
		if (TimeOfDataSpike == ti) {
			return Eigen::Map<TMatrix<BatchSize, InputSize>>(
  				X.Data + bi*BatchSize*InputSize, BatchSize, InputSize
  			);
		}
		return TMatrix<BatchSize, InputSize>::Zero();
	}

	TMatrix<BatchSize, OutputSize> ReadOutput(ui32 bi, ui32 ti) {
		if (TimeOfDataSpike == ti) {
			return Eigen::Map<TMatrix<BatchSize, OutputSize>>(
  				Y.Data + bi*BatchSize*OutputSize, BatchSize, OutputSize
  			);
		}
		return TMatrix<BatchSize, OutputSize>::Zero();
	}
};

template <int Input, int Output>
struct TSGDLearningRule {
	TSGDLearningRule(TMatrix<Input, Output>& param): Param(param) {}

	void Update(TMatrix<Input, Output>& dparam, double learningRate) {
		Param += learningRate * dparam;
		dparam = TMatrix<Input, Output>::Zero(); 
	}

	TMatrix<Input, Output>& Param;
};


template <int Input, int Output>
struct TAdadeltaLearningRule {
	TAdadeltaLearningRule(TMatrix<Input, Output>& param): Param(param) {
		AverageGradient = TMatrix<Input, Output>::Zero();
	}

	void Update(TMatrix<Input, Output>& dparam, double learningRate) {
		AverageGradient += (dparam.array().square().matrix() - AverageGradient) / 1000.0;
		
		Param += learningRate * (dparam.array() / (AverageGradient.array().sqrt() + 1e-05)).matrix();

		dparam = TMatrix<Input, Output>::Zero(); 
	}

	TMatrix<Input, Output>& Param;
	TMatrix<Input, Output> AverageGradient;
};


template <template <int, int> class TLearningRule, int Input, int Output>
auto MakeLearningRule(TMatrix<Input, Output>& param) {
	return TLearningRule<Input, Output>(param);
}

struct TLayerConfig {
	double TauSoma;
	double TauSyn;
 	double TauMean;
 	double ApicalGain;
 	double FbFactor;
 	EActivation Act;

	TMatrixFlat W;
	TMatrixFlat B;
	TMatrixFlat dW;
	TMatrixFlat dB;
	TMatrixFlat UStat; 
	TMatrixFlat AStat; 
	TMatrixFlat FbStat;
};


template <int InputSize, int LayerSize, template <int, int> class TLearningRule = TAdadeltaLearningRule>
struct TLayer {
	TLayer(TLayerConfig s0, TNetConfig c0)
		: c(c0)
		, s(s0) 
		, WLearning(MakeLearningRule<TLearningRule>(W))
		, BLearning(MakeLearningRule<TLearningRule>(B))
	{
		W = TMatrixFlat::ToEigen<InputSize, LayerSize>(s.W);
		B = TMatrixFlat::ToEigen<1, LayerSize>(s.B);

		UStat = TMatrixD::Zero(BatchSize, LayerSize*SeqLength);
		AStat = TMatrixD::Zero(BatchSize, LayerSize*SeqLength);
		FbStat = TMatrixD::Zero(BatchSize, LayerSize*SeqLength);
		
		CheckSizeMatch(UStat, s.UStat);
		CheckSizeMatch(AStat, s.AStat);
		CheckSizeMatch(FbStat, s.FbStat);

		dW = TMatrix<InputSize, LayerSize>::Zero(); 
		dB = TMatrix<1, LayerSize>::Zero(); 
		
		if (s.Act == RELU) {
			Act = &Relu<BatchSize, LayerSize>;
			ActDeriv = &ReluDeriv<BatchSize, LayerSize>;
		} else
		if (s.Act == SIGMOID) {
			Act = &Sigmoid<BatchSize, LayerSize>;
		} else {
			ENSURE(0, "Failed to find activation function #" << s.Act);
		}
	}

	~TLayer() {
		TMatrixFlat::FromEigenDynamic(UStat, &s.UStat);
		TMatrixFlat::FromEigenDynamic(AStat, &s.AStat);
		TMatrixFlat::FromEigenDynamic(FbStat, &s.FbStat);
		TMatrixFlat::FromEigen<InputSize, LayerSize>(W, &s.W);
		TMatrixFlat::FromEigen<1, LayerSize>(B, &s.B);
		TMatrixFlat::FromEigen<InputSize, LayerSize>(dW, &s.dW);
		TMatrixFlat::FromEigen<1, LayerSize>(dB, &s.dB);
	}

	void Reset() {
		Syn = TMatrix<BatchSize, InputSize>::Zero();
		U = TMatrix<BatchSize, LayerSize>::Zero();
		A = TMatrix<BatchSize, LayerSize>::Zero();
	}

	template <bool learn = true>
	auto& Run(ui32 t, TMatrix<BatchSize, InputSize> ff, TMatrix<BatchSize, LayerSize> fb) {
		Syn += c.Dt * (ff - Syn) / s.TauSyn;
		
		TMatrix<BatchSize, LayerSize> dU = Syn * W + s.FbFactor * fb - U;			

		U += c.Dt * dU / s.TauSoma;
		A = Act(U);

		UStat.block(0, t*LayerSize, BatchSize, LayerSize) = U;
		AStat.block(0, t*LayerSize, BatchSize, LayerSize) = A;
		FbStat.block(0, t*LayerSize, BatchSize, LayerSize) = fb;
		
		if (learn) {
			dW += Syn.transpose() * fb;
			dB += fb.colwise().mean();
		}
	
		return A;
	}

	void ApplyGradients() {
		WLearning.Update(dW, c.LearningRate);
		BLearning.Update(dB, c.LearningRate);

	}

	TNetConfig c;
	TLayerConfig s;

	std::function<TMatrix<BatchSize, LayerSize>(TMatrix<BatchSize, LayerSize>)> Act;
	std::function<TMatrix<BatchSize, LayerSize>(TMatrix<BatchSize, LayerSize>)> ActDeriv;
	
	TMatrix<BatchSize, InputSize> Syn;
	TMatrix<BatchSize, LayerSize> U;
	TMatrix<BatchSize, LayerSize> A;
	
	TMatrixD UStat;
	TMatrixD AStat;
	TMatrixD FbStat;
	
	TMatrix<InputSize, LayerSize> W;
	TMatrix<1, LayerSize> B;

	TMatrix<InputSize, LayerSize> dW;
	TMatrix<1, LayerSize> dB;
	
	TLearningRule<InputSize, LayerSize> WLearning;
	TLearningRule<1, LayerSize> BLearning;
};

struct TStats {
	double SquaredError = 0.0;
	double ClassificationError = 0.0;
};

struct TNet {
	TNet(TLayerConfig l0, TLayerConfig l1, TNetConfig c0)
		: L0(l0, c0)
		, L1(l1, c0)
		, c(c0)
	{
		ENSURE(c.FeedbackDelay > 0, "FeedbackDelay should be greater than zero");
	}

	
	template <bool learn = true>
	void RunOverBatch(TData data, ui32 batchIdx, TStats* stats) {
		TMatrixD deSeq = TMatrixD::Zero(BatchSize, OutputSize*SeqLength);
		
		TMatrixD yMeanStat = TMatrixD::Zero(BatchSize, OutputSize*SeqLength);
		TMatrix<BatchSize, OutputSize> yMean = TMatrixD::Zero(BatchSize, OutputSize);
		
		TMatrix<BatchSize, OutputSize> zeros = TMatrix<BatchSize, OutputSize>::Zero();
		TMatrix<BatchSize, LayerSize> lzeros = TMatrix<BatchSize, LayerSize>::Zero();

		L0.Reset(); L1.Reset();

		TMatrix<BatchSize, OutputSize> yAcc = TMatrix<BatchSize, OutputSize>::Zero();
		TMatrix<BatchSize, OutputSize> aAcc = TMatrix<BatchSize, OutputSize>::Zero();

		for (ui32 t=0; t < SeqLength; ++t) {	
			TMatrix<BatchSize, InputSize> x = data.ReadInput(batchIdx, t);
			TMatrix<BatchSize, OutputSize> y = data.ReadOutput(batchIdx, t);
			TMatrix<BatchSize, OutputSize> deFeedback = \
				deSeq.block<BatchSize, OutputSize>(0, t*OutputSize);

			TMatrix<BatchSize, LayerSize> dUdE = \
				L0.ActDeriv(L0.U).array() * (deFeedback * L1.W.transpose()).array();
			auto& a0 = L0.Run<learn>(t, x, dUdE);
			auto& a1 = L1.Run<learn>(t, a0, deFeedback);

			yMean += c.Dt * (y - yMean / c.OutputTau);

			TMatrix<BatchSize, OutputSize> de = yMean - a1;
			
			if (t < SeqLength-c.FeedbackDelay) {
				deSeq.block<BatchSize, OutputSize>(0, (t+c.FeedbackDelay)*OutputSize) = de;
			}
			yMeanStat.block<BatchSize, OutputSize>(0, t*OutputSize) = yMean;
			yAcc += y;
			aAcc += a1;
		}

		for (ui32 bi=0; bi < BatchSize; ++bi) {
			Eigen::MatrixXf::Index yMaxCol;
			yAcc.row(bi).maxCoeff(&yMaxCol);
			
			Eigen::MatrixXf::Index aMaxCol;
			aAcc.row(bi).maxCoeff(&aMaxCol);

			if (yMaxCol != aMaxCol) {
				stats->ClassificationError += 1.0 / BatchSize;
			}
		}
		stats->SquaredError += deSeq.squaredNorm();

		TMatrixFlat::FromEigenDynamic(deSeq, &c.DeStat);
		TMatrixFlat::FromEigenDynamic(yMeanStat, &c.YMeanStat);
	}

	void ApplyGradients() {
		L0.ApplyGradients();
		L1.ApplyGradients();
	}

	TNetConfig c;

	TLayer<InputSize, LayerSize> L0;
	TLayer<LayerSize, OutputSize> L1;
};


int run_model(
	ui32 epochs,
	TLayerConfig* layerStates,
	TNetConfig c,
	TData trainData,
	TData testData
) {
	try {
		std::cout.precision(5);
		ENSURE(trainData.X.NRows % BatchSize == 0, \
			"Row size of train input data should has no remainder while division on " << BatchSize);
		ENSURE(trainData.X.NCols == InputSize, "Col size of train input data should be " << InputSize);
		ENSURE(trainData.Y.NCols == OutputSize, "Col size of train output data should be " << OutputSize);
		ENSURE(testData.X.NRows % BatchSize == 0, \
			"Row size of test input data should has no remainder while division on " << BatchSize);
		ENSURE(testData.X.NCols == InputSize, "Col size of test input data should be " << InputSize);
		ENSURE(testData.Y.NCols == OutputSize, "Col size of test output data should be " << OutputSize);

		TNet net(layerStates[0], layerStates[1], c);
		
		ui32 trainNumBatches = trainData.X.NRows / BatchSize;
		ui32 testNumBatches = testData.X.NRows / BatchSize;


		for (ui32 e=0; e<epochs; ++e) {

			TStats trainStats, testStats;

			for (ui32 bi = 0; bi < trainNumBatches; ++bi) {
				net.RunOverBatch(trainData, bi, &trainStats);
			}

			for (ui32 bi = 0; bi < testNumBatches; ++bi) {
				net.RunOverBatch</*learn*/false>(testData, bi, &testStats);
			}
			std::cout << "Epoch: " << e << "\n";
			std::cout << "\tTrain; sq.error: " << trainStats.SquaredError / trainNumBatches;
			std::cout << " class.error: " << trainStats.ClassificationError / trainNumBatches << "\n"; 
			std::cout << "\tTest; sq.error: " << testStats.SquaredError / testNumBatches;
			std::cout << " class.error: " << testStats.ClassificationError / testNumBatches << "\n"; 

			net.ApplyGradients();
		}

	} catch (const std::exception& e) {
		std::cerr << "====================================\n";
		std::cerr << "ERROR:\n";
		std::cerr << "\t" << e.what() << "\n";
		std::cerr << "====================================\n";
		return 1;
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
