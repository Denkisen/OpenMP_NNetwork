#include "Images.h"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../libs/Stb/stb_image.h"
#include "../libs/Stb/stb_image_write.h"
#include "../libs/Stb/stb_image_resize.h"
#include <cstring>
#include <cstdlib>
#include <ctime>

Image OpenImage(std::string file_path)
{
  Image result = {};
  result.canva = stbi_load(file_path.c_str(), &result.width, &result.height, &result.bpp, 0);
  return result;
}

void SaveImage(std::string file_path, Image data)
{
  if (file_path.find(".png") != std::string::npos)
  {
    stbi_write_png(file_path.c_str(), data.width, data.height, data.bpp, data.canva, data.width*data.bpp);
  }
  else if (file_path.find(".jpg") != std::string::npos)
  {
    stbi_write_jpg(file_path.c_str(), data.width, data.height, data.bpp, data.canva, 100);
  }
  else if (file_path.find(".bmp") != std::string::npos)
  {
    stbi_write_bmp(file_path.c_str(), data.width, data.height, data.bpp, data.canva);
  }
}

void FreeImage(Image data)
{
  stbi_image_free(data.canva);
}

Image ResizeImage(Image data, int w, int h)
{
  Image result = {};
  result.height = h;
  result.width = w;
  result.bpp = data.bpp;
  result.canva = new unsigned char[result.height * result.width * result.bpp];
  stbir_resize_uint8(data.canva, data.width, data.height, data.width * data.bpp, 
                    result.canva, result.width, result.height, result.width * result.bpp, result.bpp);
  return result;
}

Image CutOfImage(Image data, int x, int y, int w, int h)
{
  Image result = {};
  result.height = h;
  result.width = w;
  result.bpp = data.bpp;
  result.canva = new unsigned char[result.height * result.width * result.bpp];

  #pragma omp parallel for
  for (int i = 0; i < h; ++i)
  {
    std::memcpy(&result.canva[i * (w * data.bpp)], (data.canva + (data.width * data.bpp * (i + y)) + (x * data.bpp)), w * data.bpp);
  }
  return result;
}

Image GrayscaleImage(Image data)
{
  Image result = {};
  result.height = data.height;
  result.width = data.width;
  result.bpp = 1;
  result.canva = new unsigned char[result.height * result.width * result.bpp];

  #pragma omp parallel for
  for (size_t i = 0; i < (size_t) (result.height * result.width * result.bpp); ++i)
  {
    size_t shift = (size_t) (i * data.bpp);
    result.canva[i] = (unsigned char) ((double) data.canva[shift] * 0.2126 + (double) data.canva[shift + 1] * 0.7152 + (double) data.canva[shift + 2] * 0.0722);
  }
  return result;
}

Image CorruptImage(Image data, int corruption_percentage, TransformFunction &&transform)
{
  Image result = {};
  result.height = data.height;
  result.width = data.width;
  result.bpp = data.bpp;
  result.canva = new unsigned char[result.height * result.width * result.bpp];
  std::memcpy(result.canva, data.canva, result.height * result.width * result.bpp);
  transform.Transform(result, corruption_percentage * 0.01);

  return result;
}

void StripCorruptionFunction::Transform(Image data, double val)
{
  int w = std::rand() % (int) trunc(data.width * val);
  int h = std::rand() % (int) trunc(data.height * val);  
  int x = std::rand() % (int) trunc(data.width - w);
  int y = std::rand() % (int) trunc(data.height - h);

  for (int i = 0; i < h; ++i)
  {
    std::memset(&data.canva[(data.width * data.bpp * (i + y)) + (x * data.bpp)], 0, sizeof(unsigned char) * w);
  }
}

Image MakeEmptyImage(int x, int y, int bpp)
{
  Image result = {};
  result.height = y;
  result.width = x;
  result.bpp = bpp;
  result.canva = new unsigned char[x * y * bpp];
  return result;
}