#include "kalman.h"

Kalman::Kalman()
{
    clear();
}

void Kalman::clear()
{
    initialized = false;
    dt = 0;
    xh00 = 0;
    xph00 = 0;
    xh10 =0;
    xph10 = 0;
    innovator = 0;
    alpha = 0;
    beta = 0;
}

void Kalman::initialize(float position, float velocity, float alpha, float beta)
{
    xh00 = position;
    xph00 = velocity;
    this->alpha = alpha;
    this->beta = beta;
    initialized = true;
}

void Kalman::measure(float measurement, float time_interval)
{
    xh10 = xh00 + time_interval * xph00;
    xph10 = xph00;
    innovator = measurement - xh10;
    xh00 = alpha * innovator + xh10;
    xph00 = beta * innovator / time_interval + xph10;
}
