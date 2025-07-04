#include <gdal_priv.h>
#include <cpl_conv.h>
#include <iostream>
#include <vector>

using namespace std;

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <input.tif>" << endl;
        return 1;
    }

    const char* filename = argv[1];
    GDALAllRegister();

    GDALDataset* ds = (GDALDataset*) GDALOpen(filename, GA_ReadOnly);
    if (!ds) {
        cerr << "ERROR: Failed to open " << filename << endl;
        return 1;
    }

    int width = ds->GetRasterXSize();
    int height = ds->GetRasterYSize();
    cout << "Raster size: " << width << " x " << height << endl;

    GDALRasterBand* band = ds->GetRasterBand(1);
    if (!band) {
        cerr << "ERROR: Failed to get raster band." << endl;
        GDALClose(ds);
        return 1;
    }

    // Allocate buffer
    vector<float> buffer(width * height, 0.0);

    // Read data
    CPLErr err = band->RasterIO(GF_Read, 0, 0, width, height, buffer.data(), width, height, GDT_Float32, 0, 0);
    if (err != CE_None) {
        cerr << "ERROR: RasterIO failed." << endl;
        GDALClose(ds);
        return 1;
    }

    // Print some sample pixels
    cout << "Sample pixels:" << endl;
    for (int y = 0; y < min(5, height); ++y) {
        for (int x = 0; x < min(8, width); ++x) {
            float val = buffer[y * width + x];
            cout << val << " ";
        }
        cout << endl;
    }

    // Compute min/max manually
    double min_val = std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::lowest();
    size_t nan_count = 0;
    for (size_t i = 0; i < buffer.size(); ++i) {
        if (std::isnan(buffer[i])) {
            nan_count++;
            continue;
        }
        min_val = std::min(min_val, (double)buffer[i]);
        max_val = std::max(max_val, (double)buffer[i]);
    }

    cout << "Min: " << min_val << ", Max: " << max_val << ", NaNs: " << nan_count << endl;

    GDALClose(ds);
    return 0;
}
