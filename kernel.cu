#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include<cuda.h>
#include <stdio.h>
#include "Matrix.h"
#include <vector>
#include<random>
#include<iostream>
#include <device_functions.h>
#include<time.h>
using namespace std;

void load_matrix_to_device(Matrix* pA, Matrix* pdA) {
	int ht = pA->height;
	int wt = pA->width;
	pdA->width = wt;
	pdA->height = ht;
	size_t size = ht*wt * sizeof(double);
	cudaMalloc(&(pdA->array_), size);
	cudaMemcpy(pdA->array_, pA->array_, size, cudaMemcpyHostToDevice);



}

void load_matrix_to_host(Matrix* pA, Matrix*pdA) {
	int ht = pdA->height;
	int wt = pdA->width;
	size_t size = ht*wt * sizeof(double);
	cudaMemcpy(pA->array_, pdA->array_, size, cudaMemcpyDeviceToHost);

}

__global__ void MatMulKernel(Matrix d_W, Matrix  d_H, Matrix d_AR) {


	int row = threadIdx.y + blockIdx.y*blockDim.y; // t  
	int col = threadIdx.x + blockIdx.x*blockDim.x;
	double res = 0;
	for (int e = 0; e < d_W.width; ++e) {
		res += d_W.array_[row * d_W.width + e] * d_H.array_[col * d_H.width + e];
	}
	d_AR.array_[row * d_AR.width + col] = res;




}
__global__ void extractKernel(Matrix d_W, Matrix d_W_col, int t) {

	int index = threadIdx.y + blockIdx.y*blockDim.y;  
	d_W_col.array_[index] = d_W.array_[t + index*d_W.width];
	
}
__global__ void ConstructR_hatKernel (Matrix d_R_hat,Matrix  d_W_col, Matrix d_H_col) {
	int col = threadIdx.y + blockIdx.y*blockDim.y;   
	int row  = threadIdx.x + blockIdx.x*blockDim.x; 
	if (d_R_hat.array_[col + row*d_R_hat.width] != 0.0) {
		d_R_hat.array_[col + row*d_R_hat.width] = d_R_hat.array_[col + row*d_R_hat.width] +d_W_col.array_[row]*d_H_col.array_[col] ;

	}


}
__global__ void updateR(Matrix W_col, Matrix  H_col, Matrix R_hat, Matrix  R) {

	int col = threadIdx.y + blockIdx.y*blockDim.y;
	int row = threadIdx.x + blockIdx.x*blockDim.x;
	if (R.array_[col + row*R.width] != 0.0) {
		R.array_[col + row*R.width] = R_hat.array_[col + row*R.width] - W_col.array_[row] * H_col.array_[col];
	}



}

__global__ void CCDPPKernel(Matrix W_col, Matrix H_col, Matrix R_hat, double lambda) {

	int row = threadIdx.y + blockIdx.y*blockDim.y; // t  
	
	int m = R_hat.height;
	int n = R_hat.width;

	double z_star = 0;

	double num_z_star = 0;
	double denum_z_star = lambda;
	double s_star = 0;
	double num_s_star = 0;
	double denum_s_star = lambda;
	// this array will enable us to update the R array
	 

	 
	if (row < m) {
		// we're still updating  t-th column of W 

		for (int j = 0; j < n; ++j) {
			
			if (R_hat.array_[row*n + j] != 0.0) {
				
				
				num_z_star += (R_hat.array_[row*n + j])*H_col.array_[j];
				denum_z_star += H_col.array_[j] *H_col.array_[j];
			}

		}
		denum_z_star += lambda; 
		
		z_star = num_z_star / denum_z_star;
		W_col.array_[row] = z_star;
		
		
	}

	// we must synchronyze threads before updating H 
	void __syncthreads();
	if (row >= m) {

		// we're now updating H_col
		for (int i = 0; i < m; ++i) {
			if (R_hat.array_[i*n + row - m] != 0.0) {



				num_s_star += R_hat.array_[i*n + row - m] * W_col.array_[i];
				denum_s_star += W_col.array_[i]*W_col.array_[i];
			}

		}
		denum_s_star += lambda;
		s_star = num_s_star / denum_s_star;
		
		H_col.array_[row - m] = s_star;
		
	}






}







int main() {

	for (int iter = 1; iter < 50; iter++) {
		clock_t tStart = clock();
		
		
		
		//int iter = 3;
		double lambda = 500;
		// height <-> rows , width <-> column
		// matrix A is a squared matrix with missing values
		// we first generate A randomly
		double* ele = new double[9216];
		double* el = new double[9216];
		double* elem = new double[9216];
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> dis(0, 5);
		for (int i = 0; i < 9216; i++) {
			double num_gene = dis(gen);
			if (num_gene <= 1) {
				ele[i] = 0.0;
			}
			else {
				ele[i] = num_gene;
			}
			elem[i] = 0.0;
			el[i] = num_gene;

		}
		Matrix A = Matrix(96, 96, ele);
		Matrix W = Matrix(96, 96, elem); // zeros
		Matrix H = Matrix(96, 96, el); // zeros
		Matrix R = A;
		Matrix R_hat;
		// load A,W,H,R to the device memory 
		Matrix d_A;


		Matrix d_W;
		Matrix d_H;
		Matrix d_R;
		Matrix d_R_hat;

		// Invoke kernel
		//vector<double> error; 
		dim3 dimBlock(1, 192);
		dim3 dimBlockR(16, 16);
		dim3 dimBlockcol(1, 96);
		int gri_wd = W.height + H.height;

		// verify division
		dim3 dimGrid(1, 1);
		dim3 dimGridR(6, 6);
		dim3 dimGridcol(1, 1);
		// prepare column for w and h to be used
		double* a = new double[W.height];
		double* b = new double[H.height];
		
		Matrix W_col = Matrix(W.height,1,a );
		Matrix H_col = Matrix(H.height, 1, b);
		Matrix d_W_col;
		Matrix d_H_col;
		

		for (int t = 0; t < W.width ; ++t) {
			// contruct R_hat
			R_hat=R;
			// get the t-th column of W
			load_matrix_to_device(&W_col, &d_W_col);
			load_matrix_to_device(&W, &d_W);
			extractKernel << <dimGridcol,dimBlockcol  >> > (d_W, d_W_col, t);
			load_matrix_to_host(&W, &d_W);
			load_matrix_to_host(&W_col, &d_W_col);
			//////////////////////////////////
			// get the t-th column of H
			load_matrix_to_device(&H_col, &d_H_col);
			load_matrix_to_device(&H, &d_H);
			extractKernel << <dimGridcol, dimBlockcol >> > (d_H, d_H_col, t);
			load_matrix_to_host(&H, &d_H);
			load_matrix_to_host(&H_col, &d_H_col);
			////////////////////////////////////
			//W
			load_matrix_to_device(&W_col, &d_W_col);
			//H
			load_matrix_to_device(&H_col, &d_H_col);
			// R_hat
			load_matrix_to_device(&R_hat, &d_R_hat);
			ConstructR_hatKernel <<<dimGridR, dimBlockR >> > (d_R_hat,d_W_col, d_H_col);
			load_matrix_to_host(&W_col, &d_W_col);
			load_matrix_to_host(&H_col, &d_H_col);
			load_matrix_to_host(&R_hat, &d_R_hat);



			load_matrix_to_device(&W_col, &d_W_col);


			//H

			load_matrix_to_device(&H_col, &d_H_col);
			//  R_hat
			
			load_matrix_to_device(&R_hat, &d_R_hat);
			// ccd++ algorithm
			CCDPPKernel << <dimGrid, dimBlock >> > (d_W_col, d_H_col, d_R_hat, lambda);
			
			load_matrix_to_host(&W_col, &d_W_col);


			load_matrix_to_host(&H_col, &d_H_col);
			load_matrix_to_host(&R_hat, &d_R_hat);
			
			// update W and H
			for (int i = 0; i < W.height; i++) {
				W.array_[t + i*W.width] = W_col.array_[i]; 
			}
			for (int i = 0; i < H.height; i++) {
				H.array_[t + i*H.width] = H_col.array_[i];
			}
			load_matrix_to_device(&H_col, &d_H_col);
			load_matrix_to_device(&W_col, &d_W_col);
			load_matrix_to_device(&R_hat, &d_R_hat);
			load_matrix_to_device(&R, &d_R);

			updateR << <dimGridR, dimBlockR >> > (d_W_col, d_H_col, d_R_hat, d_R);
			load_matrix_to_host(&H_col, &d_H_col);
			load_matrix_to_host(&W_col, &d_W_col);
			load_matrix_to_host(&R_hat, &d_R_hat);
			load_matrix_to_host(&R, &d_R);
			

		}
		dim3 dimBlock1(16, 16);
		dim3 dimGrid1(6, 6);
		Matrix AR = Matrix(96, 96, elem);
		Matrix d_AR;
		load_matrix_to_device(&AR, &d_AR);
		MatMulKernel << <dimGrid1, dimBlock1 >> > (d_W, d_H, d_AR);
		load_matrix_to_host(&AR, &d_AR);
		// Read W,H,R from device memory
		double erro = 0;
		for (int i = 0; i < 9216; i++) {
			if (A.array_[i] != 0.0) {
				erro += (AR.array_[i] - A.array_[i])*(AR.array_[i] - A.array_[i]);
			}
			
		}
		erro = erro / 9216;
		cout << sqrt(erro) << " iter : " << iter ;
		printf("Time taken: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

		// Free device memory
		cudaFree(d_W.array_); cudaFree(d_H.array_); cudaFree(d_R.array_); cudaFree(d_A.array_);



	}
	system("pause");

}