#include <iostream>
#include <string>
#include <cmath>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "path_handler.h"
#include "image_rw_cuda.h"
#include "padding.h"
#include "postprocessing.h"
#include "get_micro_second.h"

template <typename T>
__global__ void imageFilteringKernel( const T *d_f, const unsigned int paddedW, 
				const unsigned int paddedH, const T *d_g, const int S,
				T *d_h, const unsigned int W, const unsigned int H )
{
	unsigned int paddingSize = S;
	unsigned int filterSize = 2 * S + 1;

	const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x + paddingSize;
	const unsigned int i = blockIdx.y * blockDim.y + threadIdx.y + paddingSize;

	// The multiply-add operation for the pixel coordinate ( j, i )
	if( j >= paddingSize && j < paddedW - paddingSize && i >= paddingSize && i < paddedH - paddingSize ) {
		unsigned int oPixelPos = ( i - paddingSize ) * W + ( j - paddingSize );
		d_h[oPixelPos] = 0.0;
		for( int k = -S; k <=S; k++ ) {
			for( int l = -S; l <= S; l++ ) {
				unsigned int iPixelPos = ( i + k ) * paddedW + ( j + l );
				unsigned int coefPos = ( k + S ) * filterSize + ( l + S );
				d_h[oPixelPos] += d_f[iPixelPos] * d_g[coefPos];
			}
		}
	}

}

inline unsigned int iDivUp( const unsigned int &a, const unsigned int &b ) 
{ 
	return ( a%b != 0 ) ? (a/b+1):(a/b); 
}

int main( int argc, char *argv[] )
{
	// list of input images and filters
	std::string images[3] = { "data/images/socks.png",
				"data/images/llama.png",
				"data/images/ape.png" };

	std::string filters[5] = { "data/filters/edge_filter.txt",
				"data/filters/gaussian_filter.txt",
				"data/filters/gradient_filter.txt",
				"data/filters/identity_filter.txt",
				"data/filters/sharpen_filter.txt" };

	std::string prefix[5] = { "edge", "gaussian", "gradient", "identity", "sharpen" };

	for( unsigned int a = 0; a < sizeof(images)/sizeof(images[0]); a = a + 1 ) {
		for( unsigned int b = 0; b < sizeof(prefix)/sizeof(prefix[0]); b = b + 1 ) {

			// read input image and filter
			std::string inputImageFilePath = images[a];
			std::string filterDataFilePath = filters[b];
			std::string outputImageFilePrefix = prefix[b];

			std::string imageFileDir;
			std::string imageFileName;
			getDirFileName( inputImageFilePath, &imageFileDir, &imageFileName );

			std::string imageFilePrefix;
			std::string imageFileExt;
			getPrefixExtension( imageFileName, &imageFilePrefix, &imageFileExt );

			hsaImage<float> h_inputImage;
			h_inputImage.pngGetImageSize( inputImageFilePath );
			h_inputImage.allocImage( PAGEABLE_MEMORY );
			h_inputImage.pngReadImage( inputImageFilePath );

			float *h_image;
			unsigned int iWidth = h_inputImage.getImageWidth();
			unsigned int iHeight = h_inputImage.getImageHeight();
			h_image = new float[ iWidth * iHeight ];

			// Compute Y component
			for( unsigned int i = 0; i < h_inputImage.getImageHeight(); i++ ) {
				for( unsigned int j = 0; j < h_inputImage.getImageWidth(); j++ ) {
					unsigned int pixelPos = i * h_inputImage.getImageWidth() + j;
					h_image[pixelPos] = 0.2126 * h_inputImage.getImagePtr( 0 )[pixelPos] +
					0.7152 * h_inputImage.getImagePtr( 1 )[pixelPos] +
					0.0722 * h_inputImage.getImagePtr( 2 )[pixelPos];
				}
			}

			//
			// Read the filter data file

			std::ifstream fin;
			fin.open( filterDataFilePath.c_str() );

			// Read the size of the filter
			unsigned int filterSize;
			fin >> filterSize;

			// Read the filter kernel
			float *h_filterKernel;
			h_filterKernel = new float[ filterSize * filterSize ];
		 
			for( unsigned int i = 0; i < filterSize; i++ )
				for( unsigned int j = 0; j < filterSize; j++ )
					fin >> h_filterKernel[ i * filterSize + j ];

			// replication padding
			int hFilterSize = filterSize / 2;
			unsigned int paddedIWidth = iWidth + 2 * hFilterSize;
			unsigned int paddedIHeight = iHeight + 2 * hFilterSize;
			float *h_paddedImage;

			h_paddedImage = new float[ paddedIWidth * paddedIHeight ];


			replicationPadding( h_image, iWidth, iHeight,
					hFilterSize,
					h_paddedImage, paddedIWidth, paddedIHeight );
			
			//
			// Perform image filtering by a GPU 
			//

			// Transfer the padded image to a device 
			float *d_paddedImage;
			unsigned int paddedImageSizeByte = paddedIWidth * paddedIHeight * sizeof(float);
			checkCudaErrors( cudaMalloc( reinterpret_cast<void **>(&d_paddedImage), paddedImageSizeByte ) );
			checkCudaErrors( cudaMemcpy( d_paddedImage, h_paddedImage, paddedImageSizeByte, cudaMemcpyHostToDevice ) );

			// Set the execution configuration
			const unsigned int blockW = 32;
			const unsigned int blockH = 32;

			//const unsigned int threadBlockH = 8;
			const dim3 grid( iDivUp( iWidth, blockW ), iDivUp( iHeight, blockH ) );
			const dim3 threadBlock( blockW, blockH );
			
			// Allocate the memory space for the filter on a device
			float *d_g;
			unsigned int filterKernelSizeByte = filterSize * filterSize * sizeof(float);
			cudaMalloc( reinterpret_cast<void **>(&d_g), filterKernelSizeByte );
			cudaMemcpy( d_g, h_filterKernel, filterKernelSizeByte, cudaMemcpyHostToDevice ); // Host to Device

			float *d_filteringResult; // the filtering result
			const unsigned int imageSizeByte = iWidth * iHeight * sizeof(float);
			cudaMalloc( reinterpret_cast<void **>(&d_filteringResult), imageSizeByte );

			//checkCudaErrors( cudaDeviceSynchronize() );
			imageFilteringKernel<<<grid,threadBlock>>>( d_paddedImage, paddedIWidth, paddedIHeight,
											 d_g, hFilterSize,
											 d_filteringResult, iWidth, iHeight );
			checkCudaErrors( cudaDeviceSynchronize() );

			// Back-transfer the filtering result to a host
			float *h_filteringResultGPU;
			h_filteringResultGPU =new float[ iWidth * iHeight ];

			//checkCudaErrors( cudaMemcpy( h_filteringResultGPU, d_filteringResult, imageSizeByte, cudaMemcpyDeviceToHost ) );
			cudaMemcpy( h_filteringResultGPU, d_filteringResult, imageSizeByte, cudaMemcpyDeviceToHost );

			//
			// Save the fitlering results
			//
			hsaImage<float> filteringResultImage;
			filteringResultImage.allocImage( iWidth, iHeight, PAGEABLE_MEMORY );

			// Set the number of channels
			const unsigned int RGB = 3;

			// The GPU result
			for( unsigned int i = 0; i < iHeight; i++ ) {
				for( unsigned int j = 0; j < iWidth; j++ ) {
					unsigned int pixelPos = i * iWidth + j;
					for( unsigned int k = 0; k < RGB; k++ )
						filteringResultImage.getImagePtr( k )[pixelPos] = h_filteringResultGPU[pixelPos];
				}
			}

			std::string filteringResultGPUFileName = std::to_string(a) + "_" + outputImageFilePrefix + "_result.png";
			filteringResultImage.pngSaveImage( "data/results/" + filteringResultGPUFileName, RGB_DATA );

			//clean memory
			filteringResultImage.freeImage();
			delete [] h_filteringResultGPU;
			h_filteringResultGPU = 0;
			checkCudaErrors( cudaFree( d_filteringResult ) );
			d_filteringResult = 0;
			checkCudaErrors( cudaFree( d_paddedImage ) );
			d_paddedImage = 0;
			delete [] h_paddedImage;
			h_paddedImage = 0;
			delete [] h_image;
			h_image = 0;
			delete [] h_filterKernel;
			h_filterKernel = 0;
			h_inputImage.freeImage();
		}
	}
	return 0;  
}
