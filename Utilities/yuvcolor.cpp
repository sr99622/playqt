#include "yuvcolor.h"

YUVColor::YUVColor()
{

}

bool YUVColor::isValid()
{
    return valid;
}

YUVColor::YUVColor(const QColor& color)
{
    QRgb rgb = color.rgb();

    int R = qRed(rgb);
    int B = qBlue(rgb);
    int G = qGreen(rgb);

    y = (0.257 * R) + (0.504 * G) + (0.098 * B) + 16;
    v = (0.439 * R) - (0.368 * G) - (0.071 * B) + 128;
    u = -(0.148 * R) - (0.291 * G) + (0.439 * B) + 128;

    valid = true;
}

YUVColor::YUVColor(uint8_t y, uint8_t u, uint8_t v)
{
    this->y = y;
    this->u = u;
    this->v = v;
    valid = true;
}

YUVColor::YUVColor(enum Qt::GlobalColor color)
{
    set(color);
}

void YUVColor::set(enum Qt::GlobalColor color)
{
    valid = true;
    switch (color) {
    case Qt::white:
        y = 235;
        u = 128;
        v = 128;
        break;
    case Qt::black:
        y = 16;
        u = 128;
        v = 128;
        break;
    case Qt::red:
        y = 81;
        u = 90;
        v = 240;
        break;
    case Qt::green:
        y = 145;
        u = 54;
        v = 34;
        break;
    case Qt::darkGreen:
        y = 58;
        u = 94;
        v = 86;
        break;
    case Qt::blue:
        y = 41;
        u = 240;
        v = 110;
        break;
    case Qt::darkBlue:
        y = 15;
        u = 197;
        v = 116;
        break;
    case Qt::cyan:
        y = 170;
        u = 166;
        v = 16;
        break;
    case Qt::darkCyan:
        y = 97;
        u = 151;
        v = 58;
        break;
    case Qt::magenta:
        y = 106;
        u = 202;
        v = 222;
        break;
    case Qt::darkMagenta:
        y = 57;
        u = 174;
        v = 186;
        break;
    case Qt::yellow:
        y = 210;
        u = 16;
        v = 146;
        break;
    case Qt::darkYellow:
        y = 123;
        u = 58;
        v = 139;
        break;
    case Qt::lightGray:
        y = 211;
        u = 128;
        v = 128;
        break;
    case Qt::darkGray:
        y = 64;
        u = 128;
        v = 128;
        break;
    case Qt::gray:
        y = 128;
        u = 128;
        v = 128;
        break;
    default:
        valid = false;
        break;
    }

}
