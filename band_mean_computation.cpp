#include <iostream>
#include <gdal.h>
#include <gdal_priv.h>
#include <cpl_conv.h>
#include <mpi.h>
#include <vector>
#include <filesystem>
#include <cstring>
#include <string>

std::vector<double> calculate_band_means(const std::string& filename) {
    GDALDatasetH dataset = GDALOpen(filename.c_str(), GA_ReadOnly);
    if (dataset == nullptr) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return std::vector<double>(6, 0.0); // 返回6个波段的均值为0
    }

    // 读取每个波段并计算均值
    std::vector<double> band_means(6, 0.0); // 用于存储每个波段的均值
    for (int band_idx = 1; band_idx <= 6; ++band_idx) {
        GDALRasterBandH band = GDALGetRasterBand(dataset, band_idx);
        int x_size = GDALGetRasterBandXSize(band);
        int y_size = GDALGetRasterBandYSize(band);

        std::vector<float> data(x_size * y_size);
        CPLErr err = GDALRasterIO(band, GF_Read, 0, 0, x_size, y_size, data.data(), x_size, y_size, GDT_Float32, 0, 0);
        if (err != CE_None) {
            std::cerr << "Error reading raster band " << band_idx << " from file: " << filename << std::endl;
            GDALClose(dataset);
            return std::vector<double>(6, 0.0);
        }

        double sum = 0;
        int count = 0;

        // 计算该波段的均值
        for (float value : data) {
            if (value != 0) { // 过滤无效值
                sum += value;
                count++;
            }
        }

        // 计算该波段的均值
        band_means[band_idx - 1] = (count > 0) ? (sum / count) : 0.0;
    }

    GDALClose(dataset);
    return band_means;
}


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    GDALAllRegister();

    // 使用 MPI_Wtime 计时
    double start_time = MPI_Wtime();  // 记录开始时间

    std::string input_directory = "dataset"; //运行时将要处理的图片放在dataset文件夹里
    std::vector<std::string> image_files;

    // Rank 0 进程负责收集所有文件路径
    if (rank == 0) {
        for (const auto &entry : std::filesystem::directory_iterator(input_directory)) {
            if (entry.path().extension() == ".tif") {
                image_files.push_back(entry.path().string());
            }
        }
    }

    // 获取文件的总数量
    int total_files = image_files.size();
 
    // 将文件数量广播给所有进程
    MPI_Bcast(&total_files, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 每个进程应该处理的文件数量
    int files_per_process = total_files / size;
    int remaining_files = total_files % size;

    // 计算每个进程要接收的文件数
    std::vector<int> send_counts(size, files_per_process);
    for (int i = 0; i < remaining_files; ++i) {
        send_counts[i]++;
    }

    // 计算每个进程接收数据的起始位置
    std::vector<int> displacements(size, 0);
    for (int i = 1; i < size; ++i) {
        displacements[i] = displacements[i-1] + send_counts[i-1];
    }

    // 设置缓冲区大小，这里我们假设文件路径长度为最大 256 字节
    int max_path_length = 256; 
    std::vector<char> recv_buffer(total_files * max_path_length);  // 用于发送的缓冲区
    std::vector<std::string> local_files(send_counts[rank]);

    // Rank 0 将文件路径写入缓冲区
    if (rank == 0) {
        int index = 0;
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < send_counts[i]; ++j) {
                std::string path = image_files[index++];
                // 确保路径不会超出缓冲区并且填充结束符 '\0'
                std::memcpy(&recv_buffer[(displacements[i] + j) * max_path_length], path.c_str(), path.size() + 1);
                // 如果路径小于最大长度，手动填充剩余的空间
                if (path.size() < max_path_length) {
                    std::memset(&recv_buffer[(displacements[i] + j) * max_path_length + path.size()], '\0', max_path_length - path.size());
                }
            }
        }
    }

    // 每个进程的本地缓冲区
    std::vector<char> local_recv_buffer(send_counts[rank] * max_path_length);

    // 使用 MPI_Send 和 MPI_Recv 为各进程分配要处理的文件
    if (rank == 0) {
        // 根进程将数据发送给其他进程
        int index = 0;
        for (int i = 1; i < size; ++i) {
            // 计算每个进程应该发送的数据大小
            int count = send_counts[i] * max_path_length;
            MPI_Send(&recv_buffer[displacements[i] * max_path_length], count, MPI_CHAR, i, 0, MPI_COMM_WORLD);
        }
        // 根进程自己处理其文件路径
        std::memcpy(local_recv_buffer.data(), &recv_buffer[0], send_counts[rank] * max_path_length);
    } else {
        // 其他进程接收数据
        MPI_Recv(local_recv_buffer.data(), send_counts[rank] * max_path_length, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // 提取接收到的文件路径
    for (int i = 0; i < send_counts[rank]; ++i) {
        local_files[i] = std::string(&local_recv_buffer[i * max_path_length]);
    }

    // 计算每个进程每个文件的6个波段的均值
    std::vector<double> local_band_means(6, 0.0);
    for (const auto& file : local_files) {
        std::vector<double> band_means = calculate_band_means(file);
        for (int i = 0; i < 6; ++i) {
            local_band_means[i] += band_means[i];
        }
    }

    // 使用 MPI_Reduce 计算全局均值
    std::vector<double> global_band_means(6, 0.0);
    MPI_Reduce(local_band_means.data(), global_band_means.data(), 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // 输出每个波段的全局均值
        for (int i = 0; i < 6; ++i) {
            global_band_means[i] /= total_files;
            std::cout << "Mean of band " << i + 1 << ": " << global_band_means[i] << std::endl;
        }

        // 使用 MPI_Wtime 记录结束时间并计算总运行时间
        double end_time = MPI_Wtime();
        double duration = end_time - start_time;
        std::cout << "Total running time: " << duration << " seconds" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
