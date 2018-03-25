#include <stdio.h>

#define SHL_API extern

struct TMatrixFlat;
struct TNetConfig;
struct TLayerConfig;
struct TData;
struct TStats;

using ui32 = unsigned int;

extern "C" {
	
	SHL_API int run_model(
		ui32,
		TLayerConfig*,
		ui32,
		TNetConfig,
		TData,
		TData,
		TStats,
		TStats,
		ui32
	);
}

////////////////////////////////////
// Basics
////////////////////////////////////

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


using TMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

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

  	static TMatrix ToEigen(TMatrixFlat flat) {
  		return Eigen::Map<TMatrix>(
  			flat.Data, flat.NRows, flat.NCols
  		);
  	}

  	static void FromEigen(TMatrix m, TMatrixFlat* dst) {
  		ENSURE(m.rows() == dst->NRows, "Rows are not aligned for saving eigen matrix");
  		ENSURE(m.cols() == dst->NCols, "Cols are not aligned for saving eigen matrix");

  		memcpy(dst->Data, m.data(), sizeof(float) * dst->NRows * dst->NCols);
  	}
};

int sgn(double d) { 
	if (d < -1e-10) { 
		return -1; 
	} else { 
		return d > 1e-10; 
	} 
}

struct TStatsRecord {
	double SquaredError = 0.0;
	double ClassificationError = 0.0;
	double SignAgreement = 0.0;
	double AverageActivity = 0.0;
	double Sparsity = 0.0;
};


////////////////////////////////////
// Activation function
////////////////////////////////////

enum EActivation {
	EA_RELU = 0,
	EA_SIGMOID = 1
};

TMatrix Relu(TMatrix x) {
	return x.array().cwiseMax(0.0); //.cwiseMin(1.0);
}

TMatrix ReluDeriv(TMatrix x) {
	return x.unaryExpr(
		[](const float xv) { 
			return xv > static_cast<float>(0.0) ? 1.0f : 0.0f;
		}
	);
}

TMatrix Sigmoid(TMatrix x) {
	return 1.0/(1.0 + (-x.array()).exp());
}

////////////////////////////////////
// Learning rules
////////////////////////////////////

struct TSGDLearningRule {
	TSGDLearningRule(TMatrix param) {}

	void Update(TMatrix* param, TMatrix* dparam, double learningRate) {
		*param += learningRate * (*dparam);
		dparam->setZero();
	}
};


struct TAdadeltaLearningRule {
	TAdadeltaLearningRule(TMatrix param) {
		AverageGradient = TMatrix::Zero(param.rows(), param.cols());
	}

	void Update(TMatrix* param, TMatrix* dparam, double learningRate) {
		AverageGradient += (dparam->array().square().matrix() - AverageGradient) / 1000.0;
		
		*param += learningRate * (dparam->array() / (AverageGradient.array().sqrt() + 1e-05)).matrix();

		dparam->setZero();
	}

	TMatrix AverageGradient;
};



////////////////////////////////////
// Gradient processing
////////////////////////////////////

enum EGradientProcessing {
	EGP_NO_GRADIENT_PROCESSING = 0,
	EGP_LOCAL_LTD = 1,
	EGP_NONLINEAR = 2,
	EGP_HEBB = 3
};

void NoGradientProcessing(TMatrix a, TMatrix* de, TStatsRecord* stat) {
	stat->SignAgreement += 1.0;
}

void LocalLtdGradientProcessing(TMatrix a, TMatrix* de, TStatsRecord* stat) {
	ui32 signAgreement = 0;
	for (ui32 i=0; i<de->rows(); ++i) {
		for (ui32 j=0; j<de->cols(); ++j) {
			float origValue = (*de)(i, j);
			if ((*de)(i, j) < 0.0) {
				(*de)(i, j) = - a(i, j);
			}
			signAgreement += sgn(origValue) == sgn((*de)(i, j));
		}
	}
	stat->SignAgreement += signAgreement / (de->rows() * de->cols());
}

void NonLinearGradientProcessing(TMatrix a, TMatrix* de, TStatsRecord* stat) {
	ui32 signAgreement = 0;
	for (ui32 i=0; i<de->rows(); ++i) {
		for (ui32 j=0; j<de->cols(); ++j) {
			float origValue = (*de)(i, j);
			float& value = (*de)(i, j);
			
			if (value < 0.0) {
				value = 0.0;  // Rectification
			}

			value = 1.0/(1.0 + exp(-100.0*value)) - 0.5;  // Non-linear step			
			
			if (origValue <= 0.0) {
				value -= a(i, j);  // Local ltd	
			}

			signAgreement += sgn(origValue) == sgn(value);
		}
	}
	stat->SignAgreement += signAgreement / (de->rows() * de->cols());
}


// TODO: 
// - gain for A mean
// - adaptation
void HebbGradientProcessing(TMatrix a, TMatrix* de, TStatsRecord* stat) {
	double aMean = a.mean();
	ui32 signAgreement = 0;

	for (ui32 i=0; i<de->rows(); ++i) {
		for (ui32 j=0; j<de->cols(); ++j) {
			float origValue = (*de)(i, j);
			float& value = (*de)(i, j);
			
			if (value < 0.0) {
				value = 0.0;  // Rectification
			}
			value = 1.0/(1.0 + exp(-100.0*value)) - 0.5;  // Non-linear step			
			value *= 10.0;
			value += a(i, j);
			value *= sgn(value - 3.0*aMean);

			signAgreement += sgn(origValue) == sgn(value);
		}
	}
	stat->SignAgreement += signAgreement / (de->rows() * de->cols());
}

////////////////////////////////////
// Configuration
////////////////////////////////////

struct TNetConfig {
	double Dt;
	ui32 SeqLength;
	ui32 BatchSize;
 	double LearningRate;
 	ui32 FeedbackDelay;
 	double OutputTau;

 	TMatrixFlat DeStat;
 	TMatrixFlat YMeanStat;
};

using TLearningRule = TAdadeltaLearningRule;

struct TData {
	static constexpr ui32 TimeOfDataSpike = 10;

	TMatrixFlat X;
	TMatrixFlat Y;
	ui32 BatchSize;

	TMatrix ReadInput(ui32 bi, ui32 ti) {
		if (TimeOfDataSpike == ti) {
			return Eigen::Map<TMatrix>(
  				X.Data + bi*BatchSize*X.NCols, BatchSize, X.NCols
  			);
		}
		return TMatrix::Zero(BatchSize, X.NCols);
	}

	TMatrix ReadOutput(ui32 bi, ui32 ti) {
		if (TimeOfDataSpike == ti) {
			return Eigen::Map<TMatrix>(
  				Y.Data + bi*BatchSize*Y.NCols, BatchSize, Y.NCols
  			);
		}
		return TMatrix::Zero(BatchSize, Y.NCols);
	}
};


struct TLayerConfig {
	ui32 Size;
	double TauSoma;
	double TauSyn;
 	double TauMean;
 	double ApicalGain;
 	double FbFactor;
 	EActivation Act;
 	EGradientProcessing GradProc;

	TMatrixFlat W;
	TMatrixFlat B;
	TMatrixFlat dW;
	TMatrixFlat dB;
	TMatrixFlat Am;
	TMatrixFlat UStat; 
	TMatrixFlat AStat; 
	TMatrixFlat FbStat;
};



struct TLayer {
	TLayer(ui32 inputSize, TLayerConfig s0, TNetConfig c0)
		: BatchSize(c0.BatchSize)
		, InputSize(inputSize)
		, LayerSize(s0.Size)
		, c(c0)
		, s(s0)
		, UStat(TMatrix::Zero(BatchSize, LayerSize*c.SeqLength))
		, AStat(TMatrix::Zero(BatchSize, LayerSize*c.SeqLength))
		, FbStat(TMatrix::Zero(BatchSize, LayerSize*c.SeqLength))
		, W(TMatrixFlat::ToEigen(s.W))
		, B(TMatrixFlat::ToEigen(s.B))
		, dW(TMatrix::Zero(InputSize, LayerSize))
		, dB(TMatrix::Zero(1, LayerSize))
		, Am(TMatrixFlat::ToEigen(s.Am))
		, WLearning(TLearningRule(W))
		, BLearning(TLearningRule(B))
	{
		
		CheckSizeMatch(UStat, s.UStat);
		CheckSizeMatch(AStat, s.AStat);
		CheckSizeMatch(FbStat, s.FbStat);
		
		if (s.Act == EA_RELU) {
			Act = &Relu;
			ActDeriv = &ReluDeriv;
		} else
		if (s.Act == EA_SIGMOID) {
			Act = &Sigmoid;
		} else {
			ENSURE(0, "Failed to find activation function #" << s.Act);
		}

		if (s.GradProc == EGP_NO_GRADIENT_PROCESSING) {
			GradProc = &NoGradientProcessing;
		} 
		else if (s.GradProc == EGP_LOCAL_LTD) {
			GradProc = &LocalLtdGradientProcessing;
		}
		else if (s.GradProc == EGP_NONLINEAR) {
			GradProc = &NonLinearGradientProcessing;
		}
		else if (s.GradProc == EGP_HEBB) {
			GradProc = &HebbGradientProcessing;
		} else {
			ENSURE(0, "Failed to find gradient processing function #" << s.GradProc);
		}		
	}

	~TLayer() {
		TMatrixFlat::FromEigen(UStat, &s.UStat);
		TMatrixFlat::FromEigen(AStat, &s.AStat);
		TMatrixFlat::FromEigen(FbStat, &s.FbStat);
		TMatrixFlat::FromEigen(W, &s.W);
		TMatrixFlat::FromEigen(B, &s.B);
		TMatrixFlat::FromEigen(dW, &s.dW);
		TMatrixFlat::FromEigen(dB, &s.dB);
		TMatrixFlat::FromEigen(Am, &s.Am);
	}

	void Reset() {
		Syn = TMatrix::Zero(BatchSize, InputSize);
		U = TMatrix::Zero(BatchSize, LayerSize);
		A = TMatrix::Zero(BatchSize, LayerSize);
	}

	template <bool learn = true>
	auto& Run(ui32 t, TMatrix ff, TMatrix fb, bool collectStats) {
		Syn += c.Dt * (ff - Syn) / s.TauSyn;
		
		// GradProc(A, &fb);

		TMatrix dU = Syn * W + s.FbFactor * fb - U;			

		U += c.Dt * dU / s.TauSoma;
		
		A = Act(U); //.rowwise()) - 10.0*Am.row(0));

		if (collectStats) {
			UStat.block(0, t*LayerSize, BatchSize, LayerSize) = U;
			AStat.block(0, t*LayerSize, BatchSize, LayerSize) = A;
			FbStat.block(0, t*LayerSize, BatchSize, LayerSize) = fb;			
		}
		
		if (learn) {
			if (s.TauMean > 1e-10) {
				Am += (A.colwise().mean() - Am) / s.TauMean;	
			}

			dW += Syn.transpose() * fb;
			dB += fb.colwise().mean();
		}
	
		return A;
	}

	void ApplyGradients() {
		WLearning.Update(&W, &dW, c.LearningRate);
		BLearning.Update(&B, &dB, c.LearningRate);
	}

	ui32 BatchSize;
	ui32 InputSize;
	ui32 LayerSize;

	TNetConfig c;
	TLayerConfig s;

	std::function<TMatrix(TMatrix)> Act;
	std::function<TMatrix(TMatrix)> ActDeriv;
	
	std::function<void(TMatrix, TMatrix*, TStatsRecord*)> GradProc;

	TMatrix Syn;
	TMatrix U;
	TMatrix A;
	
	TMatrix UStat;
	TMatrix AStat;
	TMatrix FbStat;
	
	TMatrix W;
	TMatrix B;

	TMatrix dW;
	TMatrix dB;

	TMatrix Am;
	
	TLearningRule WLearning;
	TLearningRule BLearning;
};

struct TNet {
	TNet(ui32 inputSize, TLayerConfig* layersConfigs, ui32 layersNum, TNetConfig c0)
		: c(c0)
	{
		for (ui32 li=0; li < layersNum; ++li) {
			Layers.emplace_back(
				li == 0 ? inputSize : layersConfigs[li-1].Size,
				layersConfigs[li], 
				c
			);
		}
		OutputSize = Layers.back().LayerSize;

		ENSURE(c.FeedbackDelay > 0, "FeedbackDelay should be greater than zero");
	}

	
	template <bool learn = true>
	void RunOverBatch(TData data, ui32 batchIdx, TStatsRecord* stats, bool collectStats = false) {
		TMatrix deSeq = TMatrix::Zero(c.BatchSize, OutputSize*c.SeqLength);
		
		TMatrix yMeanStat = TMatrix::Zero(c.BatchSize, OutputSize*c.SeqLength);
		TMatrix yMean = TMatrix::Zero(c.BatchSize, OutputSize);
		
		TMatrix zeros = TMatrix::Zero(c.BatchSize, OutputSize);

		for (auto& l: Layers) { l.Reset(); }
		
		TMatrix yAcc = TMatrix::Zero(c.BatchSize, OutputSize);
		TMatrix aAcc = TMatrix::Zero(c.BatchSize, OutputSize);
		
		

		for (ui32 t=0; t < c.SeqLength; ++t) {	
			TMatrix x = data.ReadInput(batchIdx, t);
			TMatrix y = data.ReadOutput(batchIdx, t);
			TMatrix deFeedback = \
				deSeq.block(0, t*OutputSize, c.BatchSize, OutputSize);

			std::vector<TMatrix> feedback(Layers.size());

			for (int li=Layers.size()-1; li>=0; --li) {
				if (li == Layers.size()-1) {
					feedback[li] = deFeedback;
				} else {
					TMatrix a0Deriv = Layers[li].ActDeriv(Layers[li].U);
					feedback[li] = \
						a0Deriv.array() * (feedback[li+1] * Layers[li+1].W.transpose()).array();
					Layers[li].GradProc(Layers[li].A, &feedback[li], stats);
				}
				
			}

			TMatrix current;
			for (ui32 li=0; li < Layers.size(); ++li) {
				current = Layers[li].Run<learn>(t, li==0? x : current, feedback[li], collectStats);
				
				if (li < Layers.size()-1) {
					stats->AverageActivity += current.mean();

					stats->Sparsity += current.unaryExpr([](double v) {
						if (std::fabs(v) < 1e-10) {
							return 1.0;
						}
						return 0.0;
					}).mean();
				}
			}
			
			yMean += c.Dt * (y - yMean / c.OutputTau);

			TMatrix de = yMean - current;
			
			if (t < c.SeqLength-c.FeedbackDelay) {
				deSeq.block(0, (t+c.FeedbackDelay)*OutputSize, c.BatchSize, OutputSize) = de;
			}

			if (collectStats) {
				yMeanStat.block(0, t*OutputSize, c.BatchSize, OutputSize) = yMean;
			}

			yAcc += y;
			aAcc += current;
		}
		
		for (ui32 bi=0; bi < c.BatchSize; ++bi) {
			Eigen::MatrixXf::Index yMaxCol;
			yAcc.row(bi).maxCoeff(&yMaxCol);
			
			Eigen::MatrixXf::Index aMaxCol;
			aAcc.row(bi).maxCoeff(&aMaxCol);

			if (yMaxCol != aMaxCol) {
				stats->ClassificationError += 1.0 / c.BatchSize;
			}
		}
		stats->SquaredError += deSeq.squaredNorm();

		TMatrixFlat::FromEigen(deSeq, &c.DeStat);
		TMatrixFlat::FromEigen(yMeanStat, &c.YMeanStat);
	}

	void ApplyGradients() {
		for (auto& l: Layers) { l.ApplyGradients(); }
	}

	ui32 OutputSize;
	ui32 InputSize;

	TNetConfig c;

	std::vector<TLayer> Layers;
};

struct TStats {
	TMatrixFlat SquaredError;
	TMatrixFlat ClassificationError;
	TMatrixFlat SignAgreement;
	TMatrixFlat AverageActivity;
	TMatrixFlat Sparsity;

	void WriteStats(TStatsRecord rec, ui32 epoch, ui32 numBatches, ui32 seqLength) {
		SquaredError.Data[epoch] = rec.SquaredError / numBatches;
		ClassificationError.Data[epoch] = rec.ClassificationError / numBatches;
		SignAgreement.Data[epoch] = rec.SignAgreement / numBatches / seqLength;
		AverageActivity.Data[epoch] = rec.AverageActivity / numBatches / seqLength;
		Sparsity.Data[epoch] = rec.Sparsity / numBatches / seqLength;
	}
};

int run_model(
	ui32 epochs,
	TLayerConfig* layerStates,
	ui32 layersNum,
	TNetConfig c,
	TData trainData,
	TData testData,
	TStats trainStats,
	TStats testStats,
	ui32 testFreq
) {
	try {
		std::cout.precision(5);
		ENSURE(trainData.X.NRows % c.BatchSize == 0, \
			"Row size of train input data should has no remainder while division on " << c.BatchSize);
		
		ui32 inputSize = trainData.X.NCols;
		TNet net(inputSize, layerStates, layersNum, c);
		
		ui32 trainNumBatches = trainData.X.NRows / c.BatchSize;
		ui32 testNumBatches = testData.X.NRows / c.BatchSize;

		clock_t beginTime = clock();
		for (ui32 e=0; e<epochs; ++e) {

			TStatsRecord trainStatsRec;

			for (ui32 bi = 0; bi < trainNumBatches; ++bi) {
				net.RunOverBatch(trainData, bi, &trainStatsRec);
			}

			net.ApplyGradients();

			trainStats.WriteStats(trainStatsRec, e, trainNumBatches, c.SeqLength);

			if ((e == 0) || (e % testFreq == 0) || (e == (epochs-1))) {
				TStatsRecord testStatsRec;
				for (ui32 bi = 0; bi < testNumBatches; ++bi) {
					net.RunOverBatch</*learn*/false>(
						testData, 
						bi, 
						&testStatsRec, 
						/*collectStats*/bi == testNumBatches-1
					);
				}

				testStats.WriteStats(testStatsRec, e, testNumBatches, c.SeqLength);

				std::cout << "Epoch: " << e << ", " << 1000.0 * float(clock() - beginTime)/ CLOCKS_PER_SEC << "ms\n";
				std::cout << "\tTrain; sq.error: " << trainStatsRec.SquaredError / trainNumBatches;
				std::cout << " class.error: " << trainStatsRec.ClassificationError / trainNumBatches << "\n"; 
				std::cout << "\tTest; sq.error: " << testStatsRec.SquaredError / testNumBatches;
				std::cout << " class.error: " << testStatsRec.ClassificationError / testNumBatches << "\n";
				beginTime = clock();
			}
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
