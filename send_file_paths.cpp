#include <iostream>     // 用于输入输出操作
#include <gdal.h>       // GDAL 库的头文件
#include <gdal_priv.h>  // GDAL 库的头文件
#include <cpl_conv.h>   // GDAL 库的头文件
#include <mpi.h>        // 用于MPI（Message Passing Interface）并行计算的头文件
#include <vector>       // STL容器，存储动态数组
#include <filesystem>   // 用于文件系统操作
#include <cstring>      // 用于C风格的字符串操作
#include <string>       // 用于C++风格的字符串操作

int main(int argc, char **argv) {  // 接收命令行参数 argc 和 argv，但在这个程序中没有用到
    MPI_Init(&argc, &argv);                // 初始化 MPI 环境
    int rank, size;                    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // 获取当前进程的编号（rank）
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // 获取总的进程数（size）
    GDALAllRegister();                     // 注册所有可用的GDAL驱动，以便处理不同格式的栅格数据。使用GDAL之前需调用此函数。

    std::string input_directory = "dataset";  // 定义文件输入目录
    std::vector<std::string> image_files;     // 创建image_files向量用于存储符合条件的TIFF文件路径

    // Rank 0 进程负责收集所有文件路径
    if (rank == 0) {  // 如果当前是Rank 0进程，则使用std::filesystem::directory_iterator遍历输入目录dataset中的文件
        for (const auto &entry : std::filesystem::directory_iterator(input_directory)) {
            if (entry.path().extension() == ".tif") {  // 检查文件的扩展名是否为.tif，并将其路径存储到image_files向量中
                image_files.push_back(entry.path().string());
            }
        }
    }

    // 获取文件的总数量
    int total_files = image_files.size();  // total_files变量存储文件的总数
 
    // 将文件数量广播给所有进程，缺少这一步将导致所有进程都无法接收到文件路径
    MPI_Bcast(&total_files, 1, MPI_INT, 0, MPI_COMM_WORLD);  // 使用MPI_Bcast将文件数量广播给所有进程，包括0号进程

    // 每个进程应该处理的文件数量
    int files_per_process = total_files / size;  // 计算每个进程分配的文件数files_per_process 
    int remaining_files = total_files % size;    // 计算每个进程剩余文件数remaining_files，这是为了后续将文件动态分配给各进程，防止文件总数不是进程数的倍数时出现遗漏

    // 计算每个进程要接收的文件数
    std::vector<int> send_counts(size, files_per_process);  //send_counts向量存储每个进程应处理的文件数量
    for (int i = 0; i < remaining_files; ++i) {  // 首先将所有进程分配files_per_process个文件，剩余的文件均匀分配给前面几个进程
        send_counts[i]++;
    }

    // 计算每个进程接收数据的起始位置
    std::vector<int> displacements(size, 0);  // displacements向量存储每个进程在recv_buffer中的起始偏移量
    for (int i = 1; i < size; ++i) {   // 前一个进程的起始偏移量加上前一个进程要处理的文件数等于当前进程的起始偏移量
        displacements[i] = displacements[i-1] + send_counts[i-1];
    }

    // 设置缓冲区大小，这里我们将最大路径长度设为256字节，防止出现缓冲区空间不足的情况
    int max_path_length = 256; 
    std::vector<char> recv_buffer(total_files * max_path_length);  // recv_buffer是一个缓冲区，用来存储所有文件路径
    std::vector<std::string> local_files(send_counts[rank]);       // local_files向量存储每个进程将要处理的文件路径

    // Rank 0进程将文件路径写入缓冲区
    if (rank == 0) {  // 如果当前是rank == 0，将所有文件路径写入recv_buffer
        int index = 0;
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < send_counts[i]; ++j) {
                std::string path = image_files[index++];  // 每个文件路径按照最大路径长度max_path_length存储，若路径长度小于最大长度，则填充 \0
                // 确保路径不会超出缓冲区并且填充结束符 '\0'
                std::memcpy(&recv_buffer[(displacements[i] + j) * max_path_length], path.c_str(), path.size() + 1);
                // 如果路径小于最大长度，手动填充剩余的空间
                if (path.size() < max_path_length) {
                    std::memset(&recv_buffer[(displacements[i] + j) * max_path_length + path.size()], '\0', max_path_length - path.size());
                }
            }
        }
    }

    // 设置每个进程的本地缓冲区
    std::vector<char> local_recv_buffer(send_counts[rank] * max_path_length);

    // 使用MPI_Send和MPI_Recv发送数据
    if (rank == 0) {
        // 根进程将数据发送给其他进程
        int index = 0;
        for (int i = 1; i < size; ++i) {
            // 计算每个进程应该发送的数据大小
            int count = send_counts[i] * max_path_length;
            MPI_Send(&recv_buffer[displacements[i] * max_path_length], count, MPI_CHAR, i, 0, MPI_COMM_WORLD);
        }
        // 根进程自己处理其文件路径，缺少这一行代码将导致根进程无法接收到文件路径
        std::memcpy(local_recv_buffer.data(), &recv_buffer[0], send_counts[rank] * max_path_length);
    } else {
        // 其他进程接收数据存储到local_recv_buffer中
        MPI_Recv(local_recv_buffer.data(), send_counts[rank] * max_path_length, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // 提取接收到的文件路径
    for (int i = 0; i < send_counts[rank]; ++i) {  // 每个进程从local_recv_buffer中提取出文件路径并存储到local_files中
        local_files[i] = std::string(&local_recv_buffer[i * max_path_length]);
    }

    // 打印每个进程接收到的文件路径
    std::cout << "Rank " << rank << " received the following files:\n";
    for (const auto& file : local_files) {
        std::cout << "  " << file << std::endl;
    }

    MPI_Finalize();  // 结束MPI环境，释放所有资源
    return 0;
}
