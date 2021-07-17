#ifndef YUVCOLOR_H
#define YUVCOLOR_H

#include <stdint.h>
#include <QColor>

class YUVColor
{

public:
    YUVColor();
    YUVColor(const QColor& color);
    YUVColor(enum Qt::GlobalColor color);
    YUVColor(uint8_t y, uint8_t u, uint8_t v);
    bool isValid();

    uint8_t y;
    uint8_t u;
    uint8_t v;

    bool valid = false;

    void set(enum Qt::GlobalColor color);

};

#endif // YUVCOLOR_H
