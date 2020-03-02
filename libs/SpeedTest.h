#ifndef __CPU_NW_LIBS_SPEEDTEST_H
#define __CPU_NW_LIBS_SPEEDTEST_H

#include <iostream>
#include <chrono>

#define STOP_WATCH(var) auto var = std::chrono::high_resolution_clock::now();
#define PRINT_DIFF_WATCH(start,end) std::cout << "Elapsed: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microsec." << std::endl; 

#endif