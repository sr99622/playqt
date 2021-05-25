#ifndef YUVCOLOR_H
#define YUVCOLOR_H

#include <stdint.h>
#include <Qt>

class YUVColor
{

public:
    YUVColor();
    YUVColor(enum Qt::GlobalColor color);
    YUVColor(uint8_t y, uint8_t u, uint8_t v);

    uint8_t y;
    uint8_t u;
    uint8_t v;

    void set(enum Qt::GlobalColor color);

};

#endif // YUVCOLOR_H
