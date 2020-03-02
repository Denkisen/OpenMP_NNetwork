#ifndef __CPU_NW_DATAPROVIDERS_IMAGES_H
#define __CPU_NW_DATAPROVIDERS_IMAGES_H

#include <iostream>

struct Image
{
  int bpp = 0;
  int height = 0;
  int width = 0;
  unsigned char *canva = nullptr;
};

class TransformFunction
{
public:
  virtual void Transform(Image data, double val = 1.0) {}
};

class StripCorruptionFunction : public TransformFunction
{
public:
  void Transform(Image data, double val);
};

Image OpenImage(std::string file_path);
void SaveImage(std::string file_path, Image data);
void FreeImage(Image data);
Image ResizeImage(Image data, int w, int h);
Image CutOfImage(Image data, int x, int y, int w, int h);
Image GrayscaleImage(Image data);
Image CorruptImage(Image data, int corruption_percentage, TransformFunction &&transform);

#endif