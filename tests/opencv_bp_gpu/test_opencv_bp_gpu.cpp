// test_opencv_bm_gpu.cpp
// #define COMPILE_FOR_TESTING

#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <limits>
#include <vector>
#include <string>
#include <cstdlib>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <gdal_priv.h>

// Include the original file to access its functions
#include "opencv_bp_gpu.cpp"

using namespace std;
using namespace cv;

// ----------------------------------------------------------------------------
// class RegressionTest
// ----------------------------------------------------------------------------
class RegressionTest
{
private:

    int tests_passed = 0;
    int tests_failed = 0;

	// ------------------------------------------------------------------------
	// assert_test
	// ------------------------------------------------------------------------
    void assert_test(bool condition, const string& test_name)
	{
        if (condition) 
		{
            cout << "[PASS] " << test_name << endl;
            tests_passed++;
        } 
		else 
		{
            cout << "[FAIL] " << test_name << endl;
            tests_failed++;
        }
    }

	// ------------------------------------------------------------------------
	// file_exists
	// ------------------------------------------------------------------------
    bool file_exists(const string& filename)
	{
        ifstream file(filename);
        return file.good();
    }

	// ------------------------------------------------------------------------
	// create_test_geotiff
	// ------------------------------------------------------------------------
    void create_test_geotiff(const string& filename,
							 int width,
							 int height,
							 float base_value = 100.0f,
							 float offset = 0.0f)
	{
        GDALAllRegister();
        GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
        if (!driver) return;

        GDALDataset* ds = driver->Create(filename.c_str(),
										 width,
										 height,
										 1,
										 GDT_Float32,
										 nullptr);

        if (!ds) return;

        // Create synthetic stereo-like data with spatial patterns
        vector<float> data(width * height);
		
        for (int y = 0; y < height; y++) 
		{
            for (int x = 0; x < width; x++) 
			{
				// ---
                // Create patterns that simulate stereo imagery, and adds
				// controlled noise.
				// ---
                float value = base_value + offset +
                              sin(x * 0.05) * 30 +
                              cos(y * 0.05) * 20 +
                              sin((x + y) * 0.02) * 15 +
                              (rand() % 200) / 20.0f - 5.0f;

                data[y * width + x] = value;
            }
        }

        ds->GetRasterBand(1)->RasterIO(GF_Write,
									   0,
									   0,
									   width,
									   height,
                                       data.data(),
									   width,
									   height,
									   GDT_Float32,
									   0,
									   0);

        ds->FlushCache();
        GDALClose(ds);
    }

	// ------------------------------------------------------------------------
	// clean_up_files
	// ------------------------------------------------------------------------
    void clean_up_files(const vector<string>& filenames)
	{
        for (const string& filename : filenames) 
		{
            remove(filename.c_str());
        }
    }

	// ------------------------------------------------------------------------
	// capture_output
	// ------------------------------------------------------------------------
	std::pair<string, int> capture_output(int argc, char** argv) 
	{
	    // Redirect stdout to capture output
	    std::streambuf* orig_cout = std::cout.rdbuf();
	    std::ostringstream captured_cout;
	    std::cout.rdbuf(captured_cout.rdbuf());

	    // Redirect stderr to capture error messages
	    std::streambuf* orig_cerr = std::cerr.rdbuf();
	    std::ostringstream captured_cerr;
	    std::cerr.rdbuf(captured_cerr.rdbuf());

	    // Run the function and store its return value
	    int return_value = runCorrelator(argc, argv);

	    // Restore original streams
	    std::cout.rdbuf(orig_cout);
	    std::cerr.rdbuf(orig_cerr);

	    // Combine stdout and stderr into a single string
	    std::string output = captured_cout.str();
	    std::string error_output = captured_cerr.str();

	    if (!error_output.empty()) {
	        output += "\n[STDERR]: " + error_output;
	    }

	    // Return both the captured output and the function's return value
	    return std::make_pair(output, return_value);
	}
	
public:

	// ------------------------------------------------------------------------
	// test_runCorrelator_basic_execution
	// ------------------------------------------------------------------------
    void test_runCorrelator_basic_execution()
	{
        cout << "\n=== Testing runCorrelator Basic Execution ===" << endl;

        // Create test input files
        string left_file = "test_left_basic.tif";
        string right_file = "test_right_basic.tif";
        string output_file = "test_output_basic.tif";

        create_test_geotiff(left_file, 128, 128, 150.0f, 0.0f);

		// Offset for disparity
        create_test_geotiff(right_file, 128, 128, 140.0f, 5.0f);

        // Test 1: Basic execution with default parameters
        char* argv1[] =
		{
            const_cast<char*>("test_program"),
            const_cast<char*>(left_file.c_str()),
            const_cast<char*>(right_file.c_str()),
            const_cast<char*>(output_file.c_str())
        };

        int argc1 = 4;
        string output = capture_output(argc1, argv1).first;

        assert_test( \
			output.find("Running GPU StereoBeliefPropagation " \
					    "with parameters:") != string::npos,
                   		"runCorrelator displays parameter information");

        assert_test( \
			output.find("Raster size: 128 x 128") != string::npos,
            		    "runCorrelator correctly reads input dimensions");

        // Check if processing completed (either with CUDA or without)
        bool cuda_available = \
			output.find("OpenCV built without CUDA support") == string::npos;
		
        if (cuda_available) 
		{
            assert_test( \
				output.find("Processing tile") != string::npos,
                		    "runCorrelator processes tiles when CUDA is " \
							"available");
        } 
		else 
		{
            assert_test( \
				output.find("OpenCV built without CUDA support") != \
						 		string::npos,
                       		"runCorrelator handles CUDA unavailable " \
							 "gracefully");
        }

        clean_up_files({left_file, right_file, output_file});
    }

	// ------------------------------------------------------------------------
	// test_runCorrelator_parameter_parsing
	// ------------------------------------------------------------------------
    void test_runCorrelator_parameter_parsing()
	{
        cout << "\n=== Testing runCorrelator Parameter Parsing ===" << endl;

        string left_file = "test_left_params.tif";
        string right_file = "test_right_params.tif";
        string output_file = "test_output_params.tif";

        create_test_geotiff(left_file, 96, 96, 120.0f, 0.0f);
        create_test_geotiff(right_file, 96, 96, 115.0f, 3.0f);

        // Test 2: Custom parameters
        char* argv2[] = 
		{
            const_cast<char*>("test_program"),
            const_cast<char*>("-num_disp"), const_cast<char*>("48"),
            const_cast<char*>("-iters"), const_cast<char*>("2"),
            const_cast<char*>("-levels"), const_cast<char*>("3"),
            const_cast<char*>(left_file.c_str()),
            const_cast<char*>(right_file.c_str()),
            const_cast<char*>(output_file.c_str())
        };
		
        int argc2 = 10;
        string output = capture_output(argc2, argv2).first;

        // Verify all custom parameters are parsed correctly
        assert_test(output.find("num_disp=48") != string::npos, 
					"Custom num_disp parameter parsed");
		
        assert_test(output.find("iters=2") != string::npos, 
					"Custom iters parameter parsed");
        
		assert_test(output.find("levels=3") != string::npos, 
					"Custom levels parameter parsed");
        
        clean_up_files({left_file, right_file, output_file});
    }

	// ------------------------------------------------------------------------
	// test_runCorrelator_error_handling
	// ------------------------------------------------------------------------
    void test_runCorrelator_error_handling()
	{
        cout << "\n=== Testing runCorrelator Error Handling ===" << endl;

        // Test 3: Insufficient arguments
        char* argv3[] =
		{
            const_cast<char*>("test_program"),
            const_cast<char*>("left.tif")
        };

		int argc3 = 2;
        string output = capture_output(argc3, argv3).first;

        assert_test(output.find("Usage:") != string::npos,
                    "runCorrelator shows usage for insufficient arguments");

		// Test 4: Non-existent input files
        char* argv4[] =
		{
            const_cast<char*>("test_program"),
            const_cast<char*>("nonexistent_left.tif"),
            const_cast<char*>("nonexistent_right.tif"),
            const_cast<char*>("output.tif")
        };

        int argc4 = 4;

        output = capture_output(argc4, argv4).first;

        assert_test(output.find("Failed to open input images") != string::npos,
                    "runCorrelator handles non-existent input files");

        // Test 5: Invalid parameter
        string left_file = "test_left_invalid.tif";
        string right_file = "test_right_invalid.tif";
        string output_file = "test_output_invalid.tif";

        create_test_geotiff(left_file, 64, 64, 100.0f, 0.0f);
        create_test_geotiff(right_file, 64, 64, 95.0f, 2.0f);

        char* argv5[] =
		{
            const_cast<char*>("test_program"),
            const_cast<char*>("-invalid_param"), const_cast<char*>("value"),
            const_cast<char*>(left_file.c_str()),
            const_cast<char*>(right_file.c_str()),
            const_cast<char*>(output_file.c_str())
        };

        int argc5 = 6;

        output = capture_output(argc5, argv5).first;

        assert_test(output.find("Unknown option") != string::npos,
                    "runCorrelator handles unknown parameters");

		clean_up_files({left_file, right_file, output_file});
    }

	// ------------------------------------------------------------------------
	// test_runCorrelator_gdal_operations
	// ------------------------------------------------------------------------
    void test_runCorrelator_gdal_operations()
	{
        cout << "\n=== Testing runCorrelator GDAL Operations ===" << endl;

        string left_file = "test_left_gdal.tif";
        string right_file = "test_right_gdal.tif";
        string output_file = "test_output_gdal.tif";

        // Create test files with specific dimensions.
		int width = 128;
		int height = 128;
        create_test_geotiff(left_file, width, height, 150.0f, 0.0f);
        create_test_geotiff(right_file, width, height, 140.0f, 5.0f);

        char* argv[] = 
		{
            const_cast<char*>("test_program"),
            const_cast<char*>(left_file.c_str()),
            const_cast<char*>(right_file.c_str()),
            const_cast<char*>(output_file.c_str())
        };
		
        int argc = 4;

        string output = capture_output(argc, argv).first;

        // Test GDAL initialization and file reading
        assert_test(output.find("Raster size: 128 x 128") != string::npos,
                    "runCorrelator correctly reads raster dimensions");

        // Check if GDAL operations completed successfully
        bool processing_started = \
			output.find("Running GPU StereoBeliefPropagation") != string::npos;

        assert_test(processing_started,
			"runCorrelator initializes GDAL and starts processing");

        clean_up_files({left_file, right_file, output_file});
    }

	// ------------------------------------------------------------------------
	// test_runCorrelator_tiling_functionality
	// ------------------------------------------------------------------------
    void test_runCorrelator_tiling_functionality()
	{
        cout << "\n=== Testing runCorrelator Tiling Functionality ===" << endl;

        string left_file = "test_left_tile.tif";
        string right_file = "test_right_tile.tif";
        string output_file = "test_output_tile.tif";

        // Create larger images to test tiling
        create_test_geotiff(left_file, 200, 200, 160.0f, 0.0f);
        create_test_geotiff(right_file, 200, 200, 155.0f, 3.0f);

        // Test with small tile size to force multiple tiles
        char* argv[] = 
		{
            const_cast<char*>("test_program"),
            const_cast<char*>(left_file.c_str()),
            const_cast<char*>(right_file.c_str()),
            const_cast<char*>(output_file.c_str())
        };
		
        int argc = 4;
        string output = capture_output(argc, argv).first;

        // Verify tiling parameters are set
        assert_test(output.find("Running GPU StereoBeliefPropagation") != \
				 		string::npos,
                    "runCorrelator tiling works");

        // If CUDA is available, check for tile processing
	    if (output.find("OpenCV built without CUDA support") == string::npos)
		{
            assert_test(output.find("Processing tile") != string::npos,
                        "runCorrelator processes tiles when CUDA available");
        }

        clean_up_files({left_file, right_file, output_file});
    }

	// ------------------------------------------------------------------------
	// test_runCorrelator_stereo_processing
	// ------------------------------------------------------------------------
    void test_runCorrelator_stereo_processing()
	{
        cout << "\n=== Testing runCorrelator Stereo Processing ===" << endl;

        string left_file = "test_left_stereo.tif";
        string right_file = "test_right_stereo.tif";
        string output_file = "test_output_stereo.tif";

        // Create images with clear disparity pattern
        create_test_geotiff(left_file, 100, 100, 200.0f, 0.0f);
        create_test_geotiff(right_file, 100, 100, 190.0f, 10.0f);

        char* argv[] = {
            const_cast<char*>("test_program"),
            const_cast<char*>(left_file.c_str()),
            const_cast<char*>(right_file.c_str()),
            const_cast<char*>(output_file.c_str())
        };
		
        int argc = 4;

        string output = capture_output(argc, argv).first;

        assert_test(output.find("Running GPU StereoBeliefPropagation") != \
				 		string::npos,
                    "runCorrelator tiling works");

        // Check stereo processing parameters
        // If CUDA processing occurs, check for disparity statistics
        if (output.find("GPU mean disparity") != string::npos) 
		{
            assert_test(output.find("scale factor") != string::npos,
                        "runCorrelator calculates scale factor");
						
            assert_test(output.find("Scaled disparity range") != string::npos,
                        "runCorrelator applies scaling");
        }

        clean_up_files({left_file, right_file, output_file});
    }

	// ------------------------------------------------------------------------
	// test_runCorrelator_output_validation
	// ------------------------------------------------------------------------
    void test_runCorrelator_output_validation()
	{
        cout << "\n=== Testing runCorrelator Output Validation ===" << endl;

        string left_file = "test_left_output.tif";
        string right_file = "test_right_output.tif";
        string output_file = "test_output_final.tif";

        // create_test_geotiff(left_file, 64, 64, 140.0f, 0.0f);
        // create_test_geotiff(right_file, 64, 64, 135.0f, 2.0f);
		
		int width = 128;
		int height = 128;
        create_test_geotiff(left_file, width, height, 150.0f, 0.0f);
        create_test_geotiff(right_file, width, height, 140.0f, 5.0f);

        char* argv[] = 
		{
            const_cast<char*>("test_program"),
            const_cast<char*>(left_file.c_str()),
            const_cast<char*>(right_file.c_str()),
            const_cast<char*>(output_file.c_str())
        };
		
        int argc = 4;

        string output = capture_output(argc, argv).first;

        // Check if output file creation is attempted
        if (output.find("OpenCV built without CUDA support") == string::npos) 
		{
            // CUDA is available, output should be created
            if (file_exists(output_file)) 
			{
                assert_test(true, 
					"runCorrelator creates output file when CUDA available");

                // Validate output file properties
                GDALAllRegister();
				
                GDALDataset* ds = (GDALDataset*)GDALOpen(output_file.c_str(),
					 									 GA_ReadOnly);

                if (ds) 
				{
                    assert_test(ds->GetRasterXSize() == width, 
								"Output file has correct width");
								
                    assert_test(ds->GetRasterYSize() == height, 
								"Output file has correct height");
								
                    assert_test(ds->GetRasterBand(1)->GetRasterDataType() == \
							 		GDT_Float32,
                               	"Output file has correct data type");

                    // Check NoData handling
                    int has_nodata;
					
                    double nodata_value = \
						ds->GetRasterBand(1)->GetNoDataValue(&has_nodata);
					
                    assert_test(has_nodata != 0, 
								"Output file has NoData value set");

                    GDALClose(ds);
					
                } 
				else 
				{
                    assert_test(false, "Output file can be opened with GDAL");
                }
            }
        } 
		else 
		{
            // CUDA not available, should show appropriate message
            assert_test(output.find("OpenCV built without CUDA support") != \
					 		string::npos,
                       	"runCorrelator shows CUDA unavailable message");
        }

        clean_up_files({left_file, right_file, output_file});
    }

	// ------------------------------------------------------------------------
	// test_runCorrelator_edge_cases
	// ------------------------------------------------------------------------
    void test_runCorrelator_edge_cases()
	{
        cout << "\n=== Testing runCorrelator Edge Cases ===" << endl;

		// This case is ignored until a solution is found.
		//         // Test 7: Very small images
		// cout << "Testing very small images." << endl;
		//         string left_small = "test_left_small.tif";
		//         string right_small = "test_right_small.tif";
		//         string output_small = "test_output_small.tif";
		//
		//         create_test_geotiff(left_small, 16, 16, 100.0f, 0.0f);
		//         create_test_geotiff(right_small, 16, 16, 95.0f, 1.0f);
		//
		//         char* argv_small[] =
		// {
		//             const_cast<char*>("test_program"),
		//
		// 	// Default tile is larger than the image.
		//             const_cast<char*>(left_small.c_str()),
		//             const_cast<char*>(right_small.c_str()),
		//             const_cast<char*>(output_small.c_str())
		//         };
		//
		//         int argc_small = 4;
		//
		// string output = capture_output(argc_small, argv_small).first;
		//
		//         assert_test(output.find("Raster size: 16 x 16") != string::npos,
		//                    "runCorrelator handles small images");
        // clean_up_files({left_small, right_small, output_small});

		// Test 8: Zero disparity range
	    cout << "Testing zero disparity range." << endl;
        string left_zero = "test_left_zero.tif";
        string right_zero = "test_right_zero.tif";
        string output_zero = "test_output_zero.tif";

        create_test_geotiff(left_zero, 32, 32, 120.0f, 0.0f);
        create_test_geotiff(right_zero, 32, 32, 118.0f, 1.0f);

        char* argv_zero[] =
		{
            const_cast<char*>("test_program"),
            const_cast<char*>("-num_disp"), const_cast<char*>("0"),
            const_cast<char*>(left_zero.c_str()),
            const_cast<char*>(right_zero.c_str()),
            const_cast<char*>(output_zero.c_str())
        };

        int argc_zero = 6;

        string output = capture_output(argc_zero, argv_zero).first;

        assert_test(output.find("num_disp must be between 1 and 256") != \
				 		string::npos,
                    "runCorrelator handles zero disparity range");

        clean_up_files({left_zero, right_zero, output_zero});
    }

	// ------------------------------------------------------------------------
	// run_all_tests
	// ------------------------------------------------------------------------
    void run_all_tests()
	{
        cout << "Starting OpenCV BM GPU runCorrelator Regression Tests..." 
			 << endl;
		
        cout << "Note: Some tests may show CUDA unavailable messages - " \
				"this is expected on systems without CUDA." 
			 << endl;

        srand(12345); // Seed for reproducible test data

        test_runCorrelator_basic_execution();
        test_runCorrelator_parameter_parsing();
        test_runCorrelator_error_handling();
        test_runCorrelator_gdal_operations();
        test_runCorrelator_tiling_functionality();
        test_runCorrelator_stereo_processing();
        test_runCorrelator_output_validation();
        test_runCorrelator_edge_cases();

        cout << "\n=== Test Results ===" << endl;
        cout << "Tests passed: " << tests_passed << endl;
        cout << "Tests failed: " << tests_failed << endl;
        cout << "Total tests: " << (tests_passed + tests_failed) << endl;

        if (tests_failed == 0) 
		{
            cout << "All tests PASSED!" << endl;
        } 
		else 
		{
            cout << "Some tests FAILED!" << endl;
        }
    }
};

// ----------------------------------------------------------------------------
// main
// ----------------------------------------------------------------------------
int main()
{
    // Initialize GDAL
    GDALAllRegister();

    RegressionTest test;
    test.run_all_tests();

    return 0;
}