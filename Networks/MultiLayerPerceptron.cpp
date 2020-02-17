#include "MultiLayerPerceptron.h"
#include <algorithm>
#include <cstring>
#include <iomanip>

MultiLayerPerceptron::MultiLayerPerceptron(/* args */)
{
  std::srand(unsigned(std::time(0)));
}

MultiLayerPerceptron::~MultiLayerPerceptron()
{

}

std::vector<double> MultiLayerPerceptron::Pass(std::vector<double> input, ValueTable &temporary_layers_values, std::vector<size_t> &layout)
{
  std::vector<double> result;
  if (weights_layout.size() < 3) throw;

  std::lock_guard<std::mutex> lock(pass_forward_mutex);

  size_t t_val_size = weights_layout.size();
  size_t t_weights_layout[t_val_size];
  ValueTable t_val = new double*[t_val_size];

  #pragma omp parallel for
  for (size_t i = 0; i < t_val_size; ++i)
  {
    t_weights_layout[i] = weights_layout[i];
    t_val[i] = new double[weights_layout[i]];
  }

  input.resize(weights_layout[0]);
  std::copy(input.begin(), input.end(), t_val[0]);

  for (size_t i = 1; i < t_val_size; ++i)
  {
    #pragma omp parallel for
    for (size_t j = 0; j < t_weights_layout[i]; ++j)
    {  
      double sum = 0.0; 
      t_val[i - 1][t_weights_layout[i - 1] - 1] = bias ? 1 : 0;
      for (size_t l = 0; l < t_weights_layout[i - 1]; ++l)
      {
        sum += t_val[i - 1][l] * weights[i][j][l];
      }

      t_val[i][j] = activation_func(sum);
    }
  }    

  temporary_layers_values = t_val;
  layout.resize(t_val_size);
  std::copy(t_weights_layout, &t_weights_layout[t_val_size], layout.begin());
  result.resize(t_weights_layout[t_val_size - 1]);
  std::copy(t_val[t_val_size - 1], &t_val[t_val_size - 1][result.size()],result.begin());

  return result;
}

std::vector<double> MultiLayerPerceptron::Pass(std::vector<double> input)
{
  ValueTable temp;
  std::vector<size_t> layout;
  return Pass(input, temp, layout);
}

void MultiLayerPerceptron::AddLayer(size_t size)
{
  if (size < 3) throw;
  std::lock_guard<std::mutex> lock(pass_forward_mutex);
  size++;
  ValueTable layer = new double*[size];
  
  if (weights_layout.size() > 0)
  {
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i)
    {
      layer[i] = new double[weights_layout[weights_layout.size() - 1]];
      for (size_t j = 0; j < weights_layout[weights_layout.size() - 1]; ++j)
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

  weights_layout.push_back(size);
  Weights tmp = new double**[weights_layout.size()];

  #pragma omp parallel for
  for (size_t i = 0; i < weights_layout.size() - 1; ++i)
  {
    tmp[i] = weights[i];
  }
  tmp[weights_layout.size() - 1] = layer;

  if (weights != nullptr)
    delete[] weights;
  weights = tmp;
}

Weights MultiLayerPerceptron::GetWeights(std::vector<size_t> &layout)
{
  layout = weights_layout;
  return weights;
}

void MultiLayerPerceptron::Load(std::string file_path)
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
    std::string line;

    f >> line;
    if (line == "Version:1.0")
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
          if (line == "Info:")  
          { 
            section = "Info";
            line_in_section = 0;
            continue;
          }
          if (line == "Layout:")
          {
            section = "Layout";
            line_in_section = 0;
            continue;
          } 
          if (line == "Data:")
          {
            section = "Data";
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
                throw;
              }
              std::vector<size_t> layers_size(layers);

              #pragma omp parallel for
              for (size_t i = 0; i < layers; ++i)
                layers_size[i] = std::stoul(layout[i]);

              for (size_t i = 0; i < layers; ++i)
                AddLayer(layers_size[i] - 1);
            }
            else
            {
              throw;
            }
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
                {
                  weights[i][j][l] = std::stod(values[l]);
                }
              }
              else
              {
                weights[i][j][0] = std::stod(line);
              }

              if (j == weights_layout[i] - 1)
              {
                i++;
                j = 0;
              }
              else
              {
                j++;
              }
            }
            else
            {
              throw;
            }
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
  }
}

void MultiLayerPerceptron::Save(std::string file_path)
{
/* Format
Version:1.0
Info:
Info about network
end
Layout:
3 // layers
5 10 6 // neurons
end
Data:
*/
  std::lock_guard<std::mutex> lock(pass_forward_mutex);
  std::ofstream f(file_path, std::ios::trunc);
  if (f.is_open())
  {
    f << "Version:1.0" << std::endl;
    f << "Info:" << std::endl;
    f << "Test" << std::endl;
    f << "end" << std::endl;
    f << "Layout:" << std::endl;
    f << weights_layout.size() << std::endl;
    for (size_t i = 0; i < weights_layout.size(); ++i)
    {
      f << weights_layout[i];
      if (i < weights_layout.size() - 1) f << ",";
    }
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
        else
        {
          f << std::fixed << std::setprecision(5) << weights[i][j][0];
        }
        f << std::endl;
      }
    }
    f << "end" << std::endl;
    f.close();
  }
}

bool MultiLayerPerceptron::Bias()
{
  return bias;
}

void MultiLayerPerceptron::Bias(bool state)
{
  bias = state;
}

void MultiLayerPerceptron::SetActivationFunc(ActivationFunc func)
{
  if (func == nullptr) throw;
  std::lock_guard<std::mutex> lock(pass_forward_mutex);
  activation_func = func;
}

void MultiLayerPerceptron::SetWeightInitFunc(WeightInitFunc func)
{
  if (func == nullptr) throw;
  weight_init_func = func;
}