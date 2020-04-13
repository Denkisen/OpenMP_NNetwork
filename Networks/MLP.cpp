#include "MLP.h"
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <cmath>

MLP::MLP(/* args */)
{
  std::srand(unsigned(std::time(0)));
  type = NetworkType::MLP;
}

MLP::~MLP()
{

}

std::vector<double> MLP::Pass(std::vector<double> input, ValueTable &temporary_layers_values, std::vector<size_t> &layout)
{
  std::vector<double> result;
  if (weights_layout.size() < 3) throw;

  std::lock_guard<std::mutex> lock(pass_forward_mutex);

  size_t t_val_size = weights_layout.size();
  layout.resize(t_val_size);
  temporary_layers_values.resize(t_val_size);

  #pragma omp parallel for if (t_val_size > 100)
  for (size_t i = 0; i < t_val_size; ++i)
  {
    layout[i] = weights_layout[i];
    temporary_layers_values[i].resize(weights_layout[i]);
  }

  input.resize(weights_layout[0]);
  temporary_layers_values[0] = input;

  for (size_t i = 1; i < t_val_size; ++i)
  {
    #pragma omp parallel for
    for (size_t j = 0; j < layout[i] - 1; ++j)
    {  
      
      double sum = 0.0; 
      temporary_layers_values[i - 1][layout[i - 1] - 1] = bias ? 1 : 0;
      for (size_t l = 0; l < layout[i - 1]; ++l)
      {
        sum += temporary_layers_values[i - 1][l] * weights[i][j][l];
      }

      temporary_layers_values[i][j] = activation_func(sum);
    }
  }    
  result = temporary_layers_values[t_val_size - 1];
  result.resize(result.size() - 1);

  return result;
}

std::vector<double> MLP::Pass(std::vector<double> input)
{
  ValueTable temp;
  std::vector<size_t> layout;
  std::vector result = Pass(input, temp, layout);

  return result;
}

void MLP::SetLayersCount(size_t size)
{
  if (size < 3) throw;
  if (weights_layout.size() > 0) throw;

  weights_layout.resize(size);
  weights = new double**[weights_layout.size()];
}

void MLP::AddLayer(size_t size)
{
  if (size < 1) throw;
  if (weights_layout.size() == 0) throw;

  std::lock_guard<std::mutex> lock(pass_forward_mutex);
  size++;
  double ** layer = new double *[size];
  size_t curr_layer = 0;
  bool found = false;
  for (size_t i = 0; i < weights_layout.size(); ++i)
  {
    if (weights_layout[i] == 0)
    {
      curr_layer = i;
      found = true;
      break;
    }
  }
  if (!found) throw;
  
  if (curr_layer > 0)
  {
    for (size_t i = 0; i < size; ++i)
    {
      layer[i] = new double[weights_layout[curr_layer - 1]];
      for (size_t j = 0; j < weights_layout[curr_layer - 1]; ++j)
      {
        layer[i][j] = weight_init_func();
      } 
    }
  }
  else
  {
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i)
    {
      layer[i] = new double[1]; 
      layer[i][0] = 1.0;
    }
  }

  weights_layout[curr_layer] = size;
  weights[curr_layer] = layer;
}

Weights MLP::GetWeights(std::vector<size_t> &layout)
{
  layout = weights_layout;
  return weights;
}

bool MLP::Load(std::string file_path)
{
  std::ifstream f(file_path, std::ifstream::in);

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

  if (f.is_open())
  {
    Clean();

    std::vector<std::string> sections = {"Info:", "Layout:", "Data:"};
    std::string line;

    f >> line;
    if (line == version)
    {
      std::string section = "";
      size_t line_in_section = 0;
      size_t layers = 0;
      size_t i = 0;
      size_t j = 0;
      do 
      {
        f >> line;
        if (line == "") continue;

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
            if (line_in_section == 0)
            {
              layers = std::stoul(line);
            }
            else if (line_in_section == 1)
            {
              auto layout = Split(line, ",");
              if (layout.size() != layers) 
              {
                std::cout << "Error: layout.size() != layers :" << layout.size() << ":" << layers << std::endl;
                return false;
              }
              SetLayersCount(layers);
              std::vector<size_t> layers_size(layers);

              #pragma omp parallel for
              for (size_t i = 0; i < layers; ++i)
                layers_size[i] = std::stoul(layout[i]);

              for (size_t i = 0; i < layers; ++i)
                AddLayer(layers_size[i] - 1);
            }
            else if (line_in_section == 2) bias = (bool) std::stoi(line);
            else return false;

            line_in_section++;
          }
          if (section == "Data")
          {
            if (weights_layout.size() > 2)
            {
              if (i > 0)
              {
                auto values = Split(line, ",");
                #pragma omp parallel for
                for (size_t l = 0; l < weights_layout[i - 1]; ++l)
                  weights[i][j][l] = std::stod(values[l]);
              }
              else weights[i][j][0] = std::stod(line);

              if (j == weights_layout[i] - 1)
              {
                i++;
                j = 0;
              }
              else j++;

            }
            else return false;

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
  return false;
}

void MLP::Save(std::string file_path, std::string comments)
{
/* Format
Version:1.0
Info:
Info about network
end
Layout:
3 // layers
5 10 6 // neurons
1 // bias
end
Data:
*/
  std::lock_guard<std::mutex> lock(pass_forward_mutex);
  std::ofstream f(file_path, std::ios::trunc);
  if (f.is_open())
  {
    f << version << std::endl;
    f << "Info:" << std::endl;
    f << comments << std::endl;
    f << "end" << std::endl;
    f << "Layout:" << std::endl;
    f << weights_layout.size() << std::endl;
    for (size_t i = 0; i < weights_layout.size(); ++i)
    {
      f << weights_layout[i];
      if (i < weights_layout.size() - 1) f << ",";
    }
    f << std::endl << (int) bias;
    f << std::endl << "end" << std::endl;
    f << "Data:" << std::endl;
    for (size_t i = 0; i < weights_layout.size(); ++i)
    {
      for (size_t j = 0; j < weights_layout[i]; ++j)
      {
        if (i > 0)
        {
          for (size_t l = 0; l < weights_layout[i - 1]; ++l)
          {
            f << std::fixed << std::setprecision(5) << weights[i][j][l];
            if (l < weights_layout[i - 1] - 1) f << ",";
          }
        }
        else f << std::fixed << std::setprecision(5) << weights[i][j][0];

        f << std::endl;
      }
    }
    f << "end" << std::endl;
    f.close();
  }
}

bool MLP::Bias()
{
  return bias;
}

void MLP::Bias(bool state)
{
  bias = state;
}

void MLP::SetActivationFunc(ActivationFunc func)
{
  if (func == nullptr) throw;
  std::lock_guard<std::mutex> lock(pass_forward_mutex);
  activation_func = func;
}

void MLP::SetWeightInitFunc(WeightInitFunc func)
{
  if (func == nullptr) throw;
  weight_init_func = func;
}