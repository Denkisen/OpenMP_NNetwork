#ifndef __CPU_NW_LIBS_MATH_SIMPLEMATH_H
#define __CPU_NW_LIBS_MATH_SIMPLEMATH_H

#include <iostream>
#include <vector>

std::vector<double> NormalizeVector(std::vector<double> input);
std::vector<double> DivVector(std::vector<double> input, double val);
std::vector<double> MulVector(std::vector<double> input, double val);

#endif