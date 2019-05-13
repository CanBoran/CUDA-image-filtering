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
__global__ void imageFilteringKernel( const T *device_f, const unsigned int paddedW, 
				const unsigned int paddedH, const T *device_g, const int S,
				T *device_h, const unsigned int W, const unsigned int H )
{
	unsigned int paddingSize = S;
	unsigned int filterSize = 2 * S + 1;

	const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x + paddingSize;
	const unsigned int i = blockIdx.y * blockDim.y + threadIdx.y + paddingSize;

	// The multiply-add operation for the pixel coordinate ( j, i )
	if( j >= paddingSize && j < paddedW - paddingSize && i >= paddingSize && i < paddedH - paddingSize ) {
		unsigned int oPixelPos = ( i - paddingSize ) * W + ( j - paddingSize );
		device_h[oPixelPos] = 0.0;
		for( int k = -S; k <=S; k++ ) {
			for( int l = -S; l <= S; l++ ) {
				unsigned int iPixelPos = ( i + k ) * paddedW + ( j + l );
				unsigned int coefPos = ( k + S ) * filterSize + ( l + S );
				device_h[oPixelPos] += device_f[iPixelPos] * device_g[coefPos];
			}
		}
	}

}

inline unsigned int iCalculateGridSize( const unsigned int &a, const unsigned int &b ) 
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

			hsaImage<float> host_inputImage;
			host_inputImage.pngGetImageSize( inputImageFilePath );
			host_inputImage.allocImage( PAGEABLE_MEMORY );
			host_inputImage.pngReadImage( inputImageFilePath );

			float *host_image;
			unsigned int imageWidth = host_inputImage.getImageWidth();
			unsigned int imageHeight = host_inputImage.getImageHeight();
			host_image = new float[ imageWidth * imageHeight ];

			// Compute Y component
			for( unsigned int i = 0; i < host_inputImage.getImageHeight(); i++ ) {
				for( unsigned int j = 0; j < host_inputImage.getImageWidth(); j++ ) {
					unsigned int pixelPos = i * host_inputImage.getImageWidth() + j;
					host_image[pixelPos] = 0.2126 * host_inputImage.getImagePtr( 0 )[pixelPos] +
					0.7152 * host_inputImage.getImagePtr( 1 )[pixelPos] +
					0.0722 * host_inputImage.getImagePtr( 2 )[pixelPos];
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
			float *host_filterKernel;
			host_filterKernel = new float[ filterSize * filterSize ];
		 
			for( unsigned int i = 0; i < filterSize; i++ )
				for( unsigned int j = 0; j < filterSize; j++ )
					fin >> host_filterKernel[ i * filterSize + j ];

			// replication padding
			int hFilterSize = filterSize / 2;
			unsigned int paddedimageWidth = imageWidth + 2 * hFilterSize;
			unsigned int paddedimageHeight = imageHeight + 2 * hFilterSize;
			float *host_paddedImage;

			host_paddedImage = new float[ paddedimageWidth * paddedimageHeight ];


			replicationPadding( host_image, imageWidth, imageHeight,
					hFilterSize,
					host_paddedImage, paddedimageWidth, paddedimageHeight );
			
			//
			// Perform image filtering by a GPU 
			//

			// Transfer the padded image to a device 
			float *device_paddedImage;
			unsigned int paddedImageSizeByte = paddedimageWidth * paddedimageHeight * sizeof(float);
			checkCudaErrors( cudaMalloc( reinterpret_cast<void **>(&device_paddedImage), paddedImageSizeByte ) );
			checkCudaErrors( cudaMemcpy( device_paddedImage, host_paddedImage, paddedImageSizeByte, cudaMemcpyHostToDevice ) );

			// Set the execution configuration
			const unsigned int blockWidth = 32;
			const unsigned int blockHeight = 32;

			//const unsigned int threadblockHeight = 8;
			const dim3 grid( iCalculateGridSize( imageWidth, blockWidth ), iCalculateGridSize( imageHeight, blockHeight ) );
			const dim3 threadBlock( blockWidth, blockHeight );
			
			// Allocate the memory space for the filter on a device
			float *device_g;
			unsigned int filterKernelSizeByte = filterSize * filterSize * sizeof(float);
			cudaMalloc( reinterpret_cast<void **>(&device_g), filterKernelSizeByte );
			cudaMemcpy( device_g, host_filterKernel, filterKernelSizeByte, cudaMemcpyHostToDevice ); // Host to Device

			float *device_filteringResult; // the filtering result
			const unsigned int imageSizeByte = imageWidth * imageHeight * sizeof(float);
			cudaMalloc( reinterpret_cast<void **>(&device_filteringResult), imageSizeByte );

			//checkCudaErrors( cudaDeviceSynchronize() );
			imageFilteringKernel<<<grid,threadBlock>>>( device_paddedImage, paddedimageWidth, paddedimageHeight,
											 device_g, hFilterSize,
											 device_filteringResult, imageWidth, imageHeight );
			checkCudaErrors( cudaDeviceSynchronize() );

			// Back-transfer the filtering result to a host
			float *host_filteringResultGPU;
			host_filteringResultGPU =new float[ imageWidth * imageHeight ];

			//checkCudaErrors( cudaMemcpy( host_filteringResultGPU, device_filteringResult, imageSizeByte, cudaMemcpyDeviceToHost ) );
			cudaMemcpy( host_filteringResultGPU, device_filteringResult, imageSizeByte, cudaMemcpyDeviceToHost );

			//
			// Save the fitlering results
			//
			hsaImage<float> filteringResultImage;
			filteringResultImage.allocImage( imageWidth, imageHeight, PAGEABLE_MEMORY );

			// Set the number of channels
			const unsigned int RGB = 3;

			// The GPU result
			for( unsigned int i = 0; i < imageHeight; i++ ) {
				for( unsigned int j = 0; j < imageWidth; j++ ) {
					unsigned int pixelPos = i * imageWidth + j;
					for( unsigned int k = 0; k < RGB; k++ )
						filteringResultImage.getImagePtr( k )[pixelPos] = host_filteringResultGPU[pixelPos];
				}
			}

			std::string filteringResultGPUFileName = std::to_string(a) + "_" + outputImageFilePrefix + "_result.png";
			filteringResultImage.pngSaveImage( "data/results/" + filteringResultGPUFileName, RGB_DATA );

			//clean memory
			filteringResultImage.freeImage();
			delete [] host_filteringResultGPU;
			host_filteringResultGPU = 0;
			checkCudaErrors( cudaFree( device_filteringResult ) );
			device_filteringResult = 0;
			checkCudaErrors( cudaFree( device_paddedImage ) );
			device_paddedImage = 0;
			delete [] host_paddedImage;
			host_paddedImage = 0;
			delete [] host_image;
			host_image = 0;
			delete [] host_filterKernel;
			host_filterKernel = 0;
			host_inputImage.freeImage();
		}
	}
	return 0;  
}
