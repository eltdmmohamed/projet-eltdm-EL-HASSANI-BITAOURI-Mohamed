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

__global__ void MatMulKernel (Matrix d_W,Matrix  d_H, Matrix d_AR) {
 

	int row = threadIdx.y + blockIdx.y*blockDim.y; // t  
	int col = threadIdx.x + blockIdx.x*blockDim.x;
	double res = 0; 
	for (int e = 0; e < d_W.width; ++e) {
		res += d_W.array_[row * d_W.width + e] * d_H.array_[col * d_H.width + e];
	}
	d_AR.array_[row * d_AR.width + col] = res;

	


}
__global__ void CCDKernel(Matrix A, Matrix W, Matrix R, Matrix H, double lambda) {

	int row = threadIdx.y + blockIdx.y*blockDim.y; // t  
	int col = threadIdx.x + blockIdx.x*blockDim.x; // the same for each block of k threads , i 
	int m = A.height;
	int n = A.width; 
	int k = W.width;
	double z_star = 0;
	
	double num_z_star = 0;
	double denum_z_star = lambda;
	double s_star = 0;
	double num_s_star = 0;
	double denum_s_star = lambda;
	
	
	// update Rij for all j in omega_i 
	if (col < m) {
		// we're still updating W
		
		for (int j = 0; j < n; ++j) {
			double res = 0;
			if (A.array_[col*A.width + j] != 0.0){
			for (int e = 0; e < k; ++e) {
				res = res + W.array_[col*k + e] * H.array_[e*n + j]; 
				 
			}
			
			R.array_[col*n + j] = A.array_[col*n + j] - res;
			num_z_star += (R.array_[col*n + j] + W.array_[col*W.width + row] * H.array_[j*H.width + row])*H.array_[j*H.width + row]; 
			denum_z_star += H.array_[j*H.width + row] * H.array_[j*H.width + row]; 
			}
			 
		}
		// Rij update for all j in omega_i ( i =col )
		z_star = num_z_star / denum_z_star;
		for (int j = 0; j < n; ++j) {
			if (A.array_[col*A.width + j] != 0.0) {
				R.array_[col*A.width + j] = R.array_[col*A.width + j] - (z_star - W.array_[col*W.width + row])*H.array_[j*H.width + row]; 

			}
		}
		
		W.array_[col*k + row] = z_star;
	}
	
	// we must synchronyze threads before updating H 
	void __syncthreads();
	if ( col >= m ) {
		
	 // we're now updating H
		for (int i = 0; i < m; ++i) {
			if (A.array_[i*A.width + col-m] != 0.0) {
				
			    
				
				num_s_star += (R.array_[i*A.width + col-m] + W.array_[i*W.width + row] * H.array_[(col-m)*H.width + row])*W.array_[i*W.width + row];
				denum_s_star += W.array_[i*W.width + row] * W.array_[i*W.width + row];
			}

		}
		// Rij update for all j in omega_i ( i =col )
		s_star = num_s_star / denum_s_star;
		for (int i = 0; i < m; ++i) {
			if (A.array_[i*A.width + col-m] != 0) {
				R.array_[i*A.width + col-m] = R.array_[i*A.width + col-m] - (s_star - H.array_[(col-m)*H.width + row])*W.array_[i*W.width + row];

			}
		}
		H.array_[(col-m)*H.width + row] = s_star;


	 }


	
	


	}







int main() {
	
	for (int  iter = 1; iter < 50; iter++) {
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
		// load A,W,H,R to the device memory 
		Matrix d_A;


		Matrix d_W;
		Matrix d_H;
		Matrix d_R;

		// Invoke kernel
		//vector<double> error; 
		dim3 dimBlock(16, 16);
		int gri_wd = W.height + H.height;

		// verify division
		dim3 dimGrid(12, 6);
		for (int i = 0; i < iter; ++i) {
			load_matrix_to_device(&A, &d_A);
			//W

			load_matrix_to_device(&W, &d_W);


			//H

			load_matrix_to_device(&H, &d_H);
			// R

			load_matrix_to_device(&R, &d_R);
			CCDKernel << <dimGrid, dimBlock >> > (d_A, d_W, d_R, d_H, lambda);
			// store error in host
			load_matrix_to_host(&W, &d_W);


			load_matrix_to_host(&H, &d_H);
			load_matrix_to_host(&R, &d_R);
			load_matrix_to_host(&A, &d_A);
			/*double res = 0;
			for (int i = 0; i < 10; i++) {
				res += W.array_[70 + i] * H.array_[30 + i];
			}
			cout << "iter " << i <<  " A : " << A.array_[73] << " R : " << R.array_[73] << " res " << res << "\n" ;
			*/

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
			//cout << "AR : " << AR.array_[i] << " A :" << A.array_[i] << "\n " ; 
		}
		erro = erro / 9216; 
		cout << sqrt(erro) << " iter : " <<iter ;
		printf("Time taken: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
		
		// Free device memory
		cudaFree(d_W.array_); cudaFree(d_H.array_); cudaFree(d_R.array_); cudaFree(d_A.array_);



	}
	system("pause");

}