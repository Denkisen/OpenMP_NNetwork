#include "LSTM.h"

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

void LSTM::SetLayout(size_t input_size, size_t output_size)
{
  std::lock_guard<std::mutex> lock(pass_forward_mutex);
  if (input_size == 0 || output_size == 0)
    std::runtime_error("Incorect layout size.");

  Clean();
  mem_cell = std::vector<double>(output_size, 0.0);
  last_output = std::vector<double>(output_size, 0.0);
  this->input_size = input_size;
  weights_layout.resize(4);
  weights = new double**[weights_layout.size()];
  for (size_t i = 0; i < weights_layout.size(); ++i)
  {
    weights_layout[i] = output_size;
    weights[i] = new double*[weights_layout[i]];
    for (size_t j = 0; j < weights_layout[i]; ++j)
    {
      weights[i][j] = new double[input_size + output_size + 1];
      for (size_t z = 0; z < input_size + output_size + 1; ++z)
      {
        weights[i][j][z] = weight_init_func();
      }
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
  std::vector<double> result(last_output.size());
  std::lock_guard<std::mutex> lock(pass_forward_mutex);
  if (input.size() != input_size)
    std::runtime_error("Incorect input");

  temporary_layers_values.resize(weights_layout.size());
  std::vector<double> com_input = input;
  com_input.insert(com_input.end(), last_output.begin(), last_output.end());
  layout = weights_layout;

  for (size_t i = 0; i < weights_layout.size(); ++i)
  {
    temporary_layers_values[i].resize(weights_layout[i]);
    #pragma omp parallel for
    for (size_t j = 0; j < weights_layout[i]; ++j)
    {
      double sum = 0.0;
      for (size_t z = 0; z < com_input.size(); ++z)
      {
        sum += (weights[i][j][z] * com_input[z]);
      }
      sum += weights[i][j][com_input.size()];
      switch (i)
      {
        case FORGET_INDEX:
          temporary_layers_values[i][j] = activation_func(sum);
          mem_cell[j] *= temporary_layers_values[FORGET_INDEX][j];
          break;
        case INPUT_INDEX:
          temporary_layers_values[INPUT_INDEX][j] = activation_func(sum);
          break;
        case CELL_INDEX:
          temporary_layers_values[CELL_INDEX][j] = std::tanh(sum);
          mem_cell[j] += (temporary_layers_values[CELL_INDEX][j] * temporary_layers_values[INPUT_INDEX][j]);
          break;
        case OUTPUT_INDEX:
          temporary_layers_values[OUTPUT_INDEX][j] = activation_func(sum);
          last_output[j] = temporary_layers_values[OUTPUT_INDEX][j] * std::tanh(mem_cell[j]);
          result[j] = last_output[j];
          break;
        default: 
          break;
      }
    }
  }

  return result;
}

void LSTM::StateReset()
{
  for (size_t i = 0; i < last_output.size(); ++i)
  {
    last_output[i] = 0.0;
    mem_cell[i] = 0.0;
  }
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
          auto spt = Split(line, ",");
          if (spt.size() != 2) return false;

          input_size = std::stoul(spt[0]);
          size_t mem_size = std::stoul(spt[1]);

          if (input_size > 0 && mem_size > 0) 
            SetLayout(input_size, mem_size); 
          else 
            return false;

          line_in_section++;
        }
        if (section == "Data")
        {
          auto spt = Split(line, ",");
          #pragma omp parallel for
          for (size_t l = 0; l < mem_cell.size() + input_size + 1; ++l)
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
  f << input_size << "," << mem_cell.size() << std::endl;
  f << "end" << std::endl;
  f << "Data:" << std::endl;
  for (size_t i = 0; i < weights_layout.size(); ++i)
  {
    for (size_t j = 0; j < weights_layout[i]; ++j)
    {
      for (size_t l = 0; l < input_size + mem_cell.size() + 1; ++l)
      {
        f << std::fixed << std::setprecision(5) << weights[i][j][l];
        if (l < input_size + mem_cell.size()) f << ",";
      }
      f << std::endl;
    }
  }
  f << "end" << std::endl;
  f.close();
}