#include "LSTM.h"
#include "../Functions/activation_functions.h"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <cmath>

LSTM::LSTM()
{
  std::srand(unsigned(std::time(0)));
  type = NetworkType::LSTM;
}

LSTM::~LSTM()
{

}

void LSTM::SetLayout(size_t size)
{
  std::lock_guard<std::mutex> lock(pass_forward_mutex);
  Clean();
  mem_cell = std::vector<double>(size, 0.0);
  last_output = std::vector<double>(size, 0.0);
  // w[3][4][size] - w[W,F,b][f,i,o,c][size]
  weights_layout.resize(3);
  weights = new double**[weights_layout.size()];
  for (size_t j = 0; j < weights_layout.size(); ++j)
  {
    weights_layout[j] = 4;
    weights[j] = new double*[weights_layout[j]];
    for (size_t i = 0; i < weights_layout[j]; ++i)
    {
      weights[j][i] = new double[size];
      for (size_t z = 0; z < size; z++)
        weights[j][i][z] = weight_init_func();
    }
  }
}

std::vector<double> LSTM::Pass(std::vector<double> input)
{
  ValueTable temp;
  std::vector<size_t> layout;
  std::vector result = Pass(input, temp, layout);

  return result;
}

std::vector<double> LSTM::Pass(std::vector<double> input, ValueTable &temporary_layers_values, std::vector<size_t> &layout)
{
  std::vector<double> result(input.size());
  std::lock_guard<std::mutex> lock(pass_forward_mutex);

  if (input.size() != mem_cell.size())
    throw std::runtime_error(std::string(__func__) + "Incorrect size.");

  layout.resize(weights_layout.size());
  temporary_layers_values.resize(4);

  for (size_t i = 0; i < mem_cell.size(); ++i)
  {
    for (size_t j = 0; j < temporary_layers_values.size(); ++j)
    {
      temporary_layers_values[j].resize(mem_cell.size());
      if (j < temporary_layers_values.size() - 1)
      {
        temporary_layers_values[j][i] = Sigmoid((input[i] * weights[0][j][i]) + (last_output[i] * weights[1][j][i]) + weights[2][j][i], 1.0);
      }
      else
      {
        temporary_layers_values[j][i] = std::tanh((input[i] * weights[0][j][i]) + (last_output[i] * weights[1][j][i]) + weights[2][j][i]);
      }
    }
    mem_cell[i] = (temporary_layers_values[0][i] * mem_cell[i]) + (temporary_layers_values[1][i] * temporary_layers_values[3][i]);
    result[i] = temporary_layers_values[2][i] * std::tanh(mem_cell[i]);
  }

  return result;
}

Weights LSTM::GetWeights(std::vector<size_t> &layout)
{
  layout = weights_layout;
  return weights;
}

bool LSTM::Load(std::string file_path)
{
  std::ifstream f(file_path, std::ifstream::in);

  if (!f.is_open()) return false;

  auto Split = [] (std::string s, std::string delim)
  {
    size_t pos = 0;
    std::string token = "";
    std::vector<std::string> result;
    while ((pos = s.find(delim)) != std::string::npos) 
    {
      token = s.substr(0, pos);
      result.push_back(token);
      s.erase(0, pos + delim.length());
    }
    result.push_back(s);
    return result;
  };

  std::vector<std::string> sections = {"Info:", "Layout:", "Data:"};
  std::string line;
  f >> line;

  if (line == version)
  {
    std::string section = "";
    size_t line_in_section = 0;
    size_t i = 0;
    size_t j = 0;

    do 
    {
      f >> line;

      if (line != "end")
      {      
        if (auto index = std::find(sections.begin(), sections.end(), line); index != sections.end())
        {
          section = *index;
          section.pop_back();
          line_in_section = 0;
          continue;
        }

        if (section == "Info")
        {
          std::cout << line << std::endl;
          line_in_section++;
        }
        if (section == "Layout")
        {
          if (size_t size = std::stoul(line); size > 0) SetLayout(size); else return false;

          line_in_section++;
        }
        if (section == "Data")
        {
          auto spt = Split(line, ",");
          #pragma omp parallel for
          for (size_t l = 0; l < mem_cell.size(); ++l)
            weights[i][j][l] = std::stod(spt[l]);
          j++;

          if (j == weights_layout[i])
          {
            j = 0;
            i++;
          }
          if (i == weights_layout.size())
            return false;
          line_in_section++;
        }
      }
      else 
      {
        line_in_section = 0;
        section = "";
      }
    } while (!f.eof());
  }

  f.close();
  return true;
}

void LSTM::Save(std::string file_path, std::string comments)
{
  std::lock_guard<std::mutex> lock(pass_forward_mutex);
  std::ofstream f(file_path, std::ios::trunc);
  if (!f.is_open())
    throw std::runtime_error(std::string(__func__) + "Can not create file.");

  f << version << std::endl;
  f << "Info:" << std::endl;
  f << comments << std::endl;
  f << "end" << std::endl;
  f << "Layout:" << std::endl;
  f << mem_cell.size() << std::endl;
  f << "end" << std::endl;
  f << "Data:" << std::endl;
  for (size_t i = 0; i < weights_layout.size(); ++i)
  {
    for (size_t j = 0; j < weights_layout[i]; ++j)
    {
      for (size_t l = 0; l < mem_cell.size(); ++l)
      {
        f << std::fixed << std::setprecision(5) << weights[i][j][l];
        if (l < mem_cell.size() - 1) f << ",";
      }
      f << std::endl;
    }
  }
  f << "end" << std::endl;
  f.close();
}