#ifndef KALMAN_H
#define KALMAN_H


class Kalman
{

public:
    Kalman();

    bool initialized;
    float dt;
    float xh00;
    float xph00;
    float xh10;
    float xph10;

    float innovator;
    float alpha;
    float beta;

    void measure(float measurement, float time_interval);
    void initialize(float position, float velocity, float alpha, float beta);
    void clear();
    //void estimate();
};

#endif // KALMAN_H
