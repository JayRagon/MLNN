#pragma once
#include <iostream>
#include <vector>

#define vector std::vector
#define double_matrix vector<vector<double>>


// keeps number between 1 and 0 depending on if number is negative or positive, big or small
double Sigmoid(double x)
{
	return 1 / (1 + exp(-1 * x));
}

// derivative(e^2x) = 2e^2x
// this function is magic
double Derivative(double x)
{
	return x * (1 - x);
}

double_matrix M_Sigmoid(double_matrix mat)
{
	double_matrix calc_buffer(mat.size(), vector<double>(mat[0].size(), 0.0));

	for (size_t i = 0; i < mat.size(); i++)
	{
		for (size_t u = 0; u < mat[0].size(); u++)
		{
			calc_buffer[i][u] = Sigmoid(mat[i][u]);
		}
	}

	return calc_buffer;
}

double_matrix M_DerivativeOfSigmoid(double_matrix mat)
{
	double_matrix calc_buffer(mat.size(), vector<double>(mat[0].size(), 0.0));

	for (size_t i = 0; i < mat.size(); i++)
	{
		for (size_t u = 0; u < mat[0].size(); u++)
		{
			calc_buffer[i][u] = Derivative(mat[i][u]);
			//RESULT_MATRIX[CURRENT_ROW, CURRENT_COLUMN] : = A[CURRENT_ROW, CURRENT_COLUMN] * (1 - A[CURRENT_ROW, CURRENT_COLUMN])
		}
	}

	return calc_buffer;
}

double_matrix MultiplyMbm(double_matrix mat1, double_matrix mat2)
{
	double_matrix calc_buffer(mat1.size(), vector<double>(mat1[0].size(), 0.0));
	if (mat1.size() != mat2.size() || mat1[0].size() != mat2[0].size())
	{
		std::cout << "cannot multiply MBM\n";
		return calc_buffer;
	}


	for (size_t i = 0; i < mat1.size(); i++)
	{
		for (size_t u = 0; u < mat1[0].size(); u++)
		{
			calc_buffer[i][u] = mat1[i][u] * mat2[i][u];
		}
	}

	return calc_buffer;
}

double_matrix Transpose(double_matrix mat)
{
	double_matrix calc_buffer(mat[0].size(), vector<double>(mat.size(), 0.0));

	for (size_t i = 0; i < mat.size(); i++)
	{
		for (size_t u = 0; u < mat[0].size(); u++)
		{
			calc_buffer[u][i] = mat[i][u];
		}
	}

	return calc_buffer;
}

double_matrix Add(double_matrix mat1, double_matrix mat2)
{
	double_matrix calc_buffer(mat1.size(), vector<double>(mat1[0].size(), 0.0));
	if (mat1.size() != mat2.size() || mat1[0].size() != mat2[0].size())
	{
		std::cout << "matricies not correct sizing for addition\n";
		return calc_buffer;
	}

	for (size_t i = 0; i < mat1.size(); i++)
	{
		for (size_t u = 0; u < mat1[0].size(); u++)
		{
			calc_buffer[i][u] = mat1[i][u] + mat2[i][u];
		}
	}

	return calc_buffer;
}

double_matrix Sub(double_matrix mat1, double_matrix mat2)
{
	double_matrix calc_buffer(mat1.size(), vector<double>(mat1[0].size(), 0.0));
	if (mat1.size() != mat2.size() || mat1[0].size() != mat2[0].size())
	{
		std::cout << "matricies not correct sizing for subtraction\n";
		return calc_buffer;
	}

	for (size_t i = 0; i < mat1.size(); i++)
	{
		for (size_t u = 0; u < mat1[0].size(); u++)
		{
			calc_buffer[i][u] = mat1[i][u] - mat2[i][u];
		}
	}

	return calc_buffer;
}

double_matrix Multiply(double_matrix mat1, double_matrix mat2)
{
	double_matrix calc_buffer(mat1.size(), vector<double>(mat2[0].size(), 0.0));
	if (mat1[0].size() != mat2.size())
	{
		std::cout << "matricies not halal for multiplication\n";
		return calc_buffer;
	}

	for (size_t i = 0; i < mat1.size(); i++)
	{
		for (size_t u = 0; u < mat2[0].size(); u++)
		{
			for (size_t x = 0; x < mat1[0].size(); x++)
			{
				calc_buffer[i][u] += mat1[i][x] * mat2[x][u];
			}
		}
	}

	return calc_buffer;
}