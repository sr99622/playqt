#ifndef SIMPLEFILTER_H
#define SIMPLEFILTER_H

#include <QMainWindow>
#include "Display.h"
#include "Frame.h"
#include "yolo_v2_class.hpp"
#include <npp.h>
#include "cudaexception.h"


class SimpleFilter : public QObject
{
    Q_OBJECT
public:

    SimpleFilter(QMainWindow *parent);
    ~SimpleFilter();
    void initialize(AVFrame *f);
    void destroy();

    QMainWindow *mainWindow;
    bool first_pass;
    AVFrame *rgb;
    image_t img;
    SwsContext *sws_ctx;

    void processCPU(Frame *vp);
    void processGPU(Frame *vp);
    void process(Frame *vp);
    void infer(Frame *vp);
    void test();

    void cuda_example(Frame *vp);
    void nppi_example(Frame *vp);
    void box_filter(Frame *vp);

    NppStreamContext initializeNppStreamContext();

    CudaExceptionHandler eh;
};

#endif // SIMPLEFILTER_H
