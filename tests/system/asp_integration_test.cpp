
#include <filesystem>
#include <iostream>
#include <vector>

#include <gdal_priv.h>

using namespace std;

// ---------------------------------------------------------------------------
// main
//
// This application can be used in lieu of a correlator to test aspects of
// integration as an ASP plug in.
//
// g++ asp_integration_test.cpp -Lgdal -o asp_integration_test -I/usr/include/gdal -lgdal
//
// asp_integration_test /explore/nobackup/people/pmontesa/outASP/blacksky/BSG-STEREO-117-20221009-125017-40530679-stereo/BSG-117-20221009-125017-40530679_georeferenced-pan.tif /explore/nobackup/people/pmontesa/outASP/blacksky/BSG-STEREO-117-20221009-125017-40530679-stereo/BSG-117-20221009-125025-40530680_georeferenced-pan.tif
// ---------------------------------------------------------------------------
int main(int argc, char** argv) 
{
	cout << "This is the asp_integration_test." << endl;
	
    if (argc < 2) 
	{
        cout << "Usage: " 
			 << argv[0] 
			 << " <left.tif> <right.tif>" 
			 << endl;

        return 1;
    }

    filesystem::path left_path(argv[1]);
    filesystem::path right_path(argv[2]);
	
	// gpu_output-512_512_512_512  -->  512_512_512_512-aligned-disparity.tif
	filesystem::path parent = left_path.parent_path();
	string parentStr = parent.string();
	auto start = parentStr.find("-");
	int length = parentStr.length() - start;
	string prefix = parentStr.substr(start, length);
    filesystem::path out_path = parent / (prefix + "-aligned-disparity.tif");
	
	// Read the images.
    GDALAllRegister();
	
    GDALDataset* l_ds = 
		(GDALDataset*)GDALOpen(left_path.string().c_str(), GA_ReadOnly);
	
    GDALDataset* r_ds = 
		(GDALDataset*)GDALOpen(right_path.string().c_str(), GA_ReadOnly);
    
	if (!l_ds || !r_ds) 
	{
        cerr << "Failed to open input images." << endl;
        return 1;
    }
	
	// Open and read the left image.
	GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");

	if (!driver)
	{
	    cerr << "Could not get GTiff driver" << endl;
	    return 1;
	}

	int width = l_ds->GetRasterXSize();
	int height = l_ds->GetRasterYSize();
    vector<float> buffer(width * height, 0.0);
	
    l_ds->GetRasterBand(1)->RasterIO(GF_Read,
									 0,
									 0,
									 width,
									 height,
									 buffer.data(),
									 width,
									 height,
									 GDT_Float32,
									 0,
									 0);

	// Create the output data set.
    GDALDataset* out_ds = driver->Create(out_path.c_str(),
										 width,
    									 height,
									 	 1,
									 	 GDT_Float32,
									 	 nullptr);

    if (!out_ds)
	{
        cerr << "Could not create output dataset" << endl;
        return 1;
    }

	out_ds->GetRasterBand(1)->RasterIO(GF_Write, 
								   	   0,
								   	   0, 
								   	   width,
								       height,
	        					       buffer.data(), 
								       width,
								       height,
	        					       GDT_Float32,
									   0,
									   0);
								   
	out_ds->FlushCache();
	GDALClose(out_ds);
}
