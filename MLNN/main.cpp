#include <iostream>
#include <windows.h>
#include <chrono>
#include <vector>
#include <fstream>
#include "xrand.h"
#include "matrix_math.h"


constexpr uint32_t firstlayerneurons = 5;

uint32_t import_flag = 0;

constexpr uint32_t training_cycles = 1000;
constexpr uint32_t dlen = 3;
constexpr uint32_t qlen = 7;

// [0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0])
const double training_inputs_buffer[qlen][dlen] =
{
	{0, 0, 1},
	{0, 1, 1},
	{1, 0, 1},
	{0, 1, 0},
	{1, 0, 0},
	{1, 1, 1},
	{0, 0, 0}
};

// [0],[1],[1],[1],[1],[0],[0]
const double training_answers_buffer[qlen][1] =
{
	{0},
	{1},
	{1},
	{1},
	{1},
	{0},
	{0}
};

// [1], [1], [0]
const double validation_case[3] =
{
	1, 1, 0
};


int main()
{
	auto starttime = std::chrono::high_resolution_clock::now();

	uint32_t OLD_INDEX = 0;

	// for weights_1; first number in the init is the amount of inputs that it takes.
	// then the next number (before 1.0) is the amount of outputs it will give
	// for weights_2 it's the same deal, which is why weights_1 must be 3x4 and weights_2 must be 4x1
	// so after the weights_2 is calculated, it will give one answer!
	double_matrix weights_1(dlen, vector<double>(firstlayerneurons, 1.0));
	double_matrix weights_2(firstlayerneurons, vector<double>(1, 1.0));

	double_matrix training_inputs(qlen, vector<double>(dlen, 0.0));
	double_matrix training_answers(qlen, vector<double>(1, 0.0));

	if (import_flag == 1)
	{
		// import weights from file here


		// then skip training
		//goto POST_TRAINING;
	}

	for (size_t i = 0; i < weights_1.size(); i++)
	{
		for (size_t u = 0; u < weights_1[0].size(); u++)
		{
			weights_1[i][u] = XRAND(1.0, -1.0);
		}
	}
	for (size_t i = 0; i < weights_2.size(); i++)
	{
		for (size_t u = 0; u < weights_2[0].size(); u++)
		{
			weights_2[i][u] = XRAND(1.0, -1.0);
		}
	}

	for (size_t i = 0; i < training_inputs.size(); i++)
	{
		for (size_t u = 0; u < training_inputs[0].size(); u++)
		{
			training_inputs[i][u] = training_inputs_buffer[i][u];
			std::cout << "training_inputs[" << i << "][" << u << "]: " << training_inputs[i][u] << '\n';
		}
		training_answers[i][0] = training_answers_buffer[i][0];
		std::cout << "training_asnwers[" << i << "][" << 0 << "]: " << training_answers[i][0] << '\n';
	}

	for (size_t i = 0; i < training_cycles; i++)
	{
		//First, we calculate an output from layer 1. This is done by multiplying the inputs and the weights.
		double_matrix output_layer_1 = M_Sigmoid(Multiply(training_inputs, weights_1));

		//Then we calculate a derivative (rate of change) for the output of layer 1.
		double_matrix output_layer_1_derivative = M_DerivativeOfSigmoid(output_layer_1);

		//Next, we calculate the outputs of the second layer.
		double_matrix output_layer_2 = M_Sigmoid(Multiply(output_layer_1, weights_2));

		//And than we also calculate a derivative (rate of change) for the outputs of layer 2.
		double_matrix output_layer_2_derivative = M_DerivativeOfSigmoid(output_layer_2);

		//Next, we check the errors of layers 2. Since layer 2 is the last, this is just a difference between calculated results and expected results.
		double_matrix layer_2_error = Sub(training_answers, output_layer_2);

		//Now we calculate a delta for layer 2. A delta is a rate of change: how much a change will affect the results.
		double_matrix layer_2_delta = MultiplyMbm(layer_2_error, output_layer_2_derivative);

		//Then, we transpose the matrix of weights (this is just to allow matricial multiplication, we are just reseting the dimensions of the matrix).
		double_matrix weights_2_transposed = Transpose(weights_2);

		/*
		; So, we multiply (matricial multiplication) the delta (rate of change) of layer 2 and the transposed matrix of weights of layer 2.
		; This is what gives us a matrix that represents the error of layer 1 (REMEBER: The error of layer 1 is measured by the rate of change of layer 2).
		; It may seem counter-intuitive at first that the error of layer 1 is calculated solely with arguments about layer 2, but you have to interpret this line alongside the line below (just read it).
		*/

		double_matrix layer_1_error = Multiply(layer_2_delta, weights_2_transposed);

		/*
		; Thus, when we calculate the delta (rate of change) of layer 1, we are finally connecting the layer 2 arguments (by the means of LAYER_1_ERROR) to layer 1 arguments (by the means of layer_1_derivative).
		; The rates of change (deltas) are the key to understand multi-layer neural networks. Their calculation answer this: If i change the weights of layer 1 by X, how much will it change layer 2s output?
		; This Delta defines the adjustment of the weights of layer 1 a few lines below...
		*/

		double_matrix layer_1_delta = MultiplyMbm(layer_1_error, output_layer_1_derivative);

		//Then, we transpose the matrix of training inputs (this is just to allow matricial multiplication, we are just reseting the dimensions of the matrix to better suit it).
		double_matrix training_inputs_transposed = Transpose(training_inputs);

		//Finally, we calculate how much we have to adjust the weights of layer 1. The delta of the Layer 1 versus the inputs we used this time are the key here.
		double_matrix adjust_layer_1 = Multiply(training_inputs_transposed, layer_1_delta);

		//Another matricial transposition to better suit multiplication...
		double_matrix output_layer_1_transposed = Transpose(output_layer_1);

		//And finally, we also calculate how much we have to adjust the weights of layer 2. The delta of the Layer 2 versus the inputs of layer 2 (which are really the outputs of layer 1) are the key here.
		double_matrix adjust_layer_2 = Multiply(output_layer_1_transposed, layer_2_delta);

		
		double_matrix SpeedMatrix1(adjust_layer_1.size(), vector<double>(adjust_layer_1[0].size(), 4.0));
		double_matrix SpeedMatrix2(adjust_layer_2.size(), vector<double>(adjust_layer_2[0].size(), 4.0));
		adjust_layer_1 = MultiplyMbm(SpeedMatrix1, adjust_layer_1);
		adjust_layer_2 = MultiplyMbm(SpeedMatrix2, adjust_layer_2);
		

		//And then we adjust the weights to aproximate intended results.
		weights_1 = Add(weights_1, adjust_layer_1);
		weights_2 = Add(weights_2, adjust_layer_2);

		if (i > OLD_INDEX + (training_cycles / 100))
		{
			//std::cout << i / (training_cycles / 100) << "% done" << '\n';
			OLD_INDEX = i;
		}
	}

//POST_TRAINING:



	for (size_t i = 0; i < weights_1.size(); i++)
	{
		for (size_t u = 0; u < weights_1[0].size(); u++)
		{
			//std::cout << "weights_1[" << i << "][" << u << "]: " << weights_1[i][u] << '\n';
		}
	}

	for (size_t i = 0; i < weights_2.size(); i++)
	{
		for (size_t u = 0; u < weights_2[0].size(); u++)
		{
			//std::cout << "weights_2[" << i << "][" << u << "]: " << weights_2[i][u] << '\n';
		}
	}


	double layer_1_neuron_results[firstlayerneurons];
	for (size_t i = 0; i < firstlayerneurons; i++)
	{
		layer_1_neuron_results[i] = Sigmoid(validation_case[0] * weights_1[0][i] + validation_case[1] * weights_1[1][i] + validation_case[2] * weights_1[2][i]);
	}
	double layer_1_and_2_result = 0;
	for (size_t i = 0; i < firstlayerneurons; i++)
	{
		layer_1_and_2_result += layer_1_neuron_results[i] * weights_2[i][0];
	}
	layer_1_and_2_result = Sigmoid(layer_1_and_2_result);

	double confidence = 0;
	double final_result;
	if (layer_1_and_2_result < 0.5)
	{
		confidence = 1 - layer_1_and_2_result;
		final_result = 0;
	}
	else
	{
		confidence = layer_1_and_2_result;
		final_result = 1;
	}


	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = (finish - starttime);
	std::cout << "time: " << elapsed.count() * 1000 << "ms\n";

	std::cout << "RESULT: " << final_result << '\n' << "With " << confidence * 100 << "% confidence\n";
	std::cout << layer_1_and_2_result;

	// save weights to file
	//std::cout << "exporting...\n";

	//std::cout << "weights exported!\n";
}


// origional ahk way of calc final result
/*
double out_1 = Sigmoid(validation_case[0] * weights_1[0][0] + validation_case[1] * weights_1[1][0] + validation_case[2] * weights_1[2][0]);
double out_2 = Sigmoid(validation_case[0] * weights_1[0][1] + validation_case[1] * weights_1[1][1] + validation_case[2] * weights_1[2][1]);
double out_3 = Sigmoid(validation_case[0] * weights_1[0][2] + validation_case[1] * weights_1[1][2] + validation_case[2] * weights_1[2][2]);
double out_4 = Sigmoid(validation_case[0] * weights_1[0][3] + validation_case[1] * weights_1[1][3] + validation_case[2] * weights_1[2][3]);
double out_final = Sigmoid(out_1 * weights_2[0][0] + out_2 * weights_2[1][0] + out_3 * weights_2[2][0] + out_4 * weights_2[3][0]);

*/
