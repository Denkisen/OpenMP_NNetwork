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


Image OpenImage(std::string file_path);
void SaveImage(std::string file_path, Image data);
void FreeImage(Image data);
Image ResizeImage(Image data, int w, int h);
Image GrayscaleImage(Image data);

#endif