#ifndef SIMPLEFILTER_H
#define SIMPLEFILTER_H

#include <QMainWindow>
#include "Display.h"
#include "Frame.h"
#include "yolo_v2_class.hpp"

class SimpleFilter : public QObject
{
    Q_OBJECT
public:

    SimpleFilter(QMainWindow *parent);
    ~SimpleFilter();
    void initialize(AVFrame *f);
    void destroy();
    void test(AVFrame *f);

    QMainWindow *mainWindow;
    bool first_pass;
    AVFrame *rgb;
    image_t img;
    SwsContext *sws_ctx;

    void processCPU(Frame *vp);
    void processGPU(Frame *vp);
    void process(Frame *vp);
    void infer(Frame *vp);

};

#endif // SIMPLEFILTER_H
