#ifndef MODEL_H
#define MODEL_H

#include "Ffplay/Frame.h"

#include <QMainWindow>
#include <QRunnable>

#include "Utilities/avexception.h"
#include "yolo_v2_class.hpp"
#include "Utilities/waitbox.h"

#include <npp.h>
#include "Utilities/cudaexception.h"

class ModelLoader : public QObject, public QRunnable
{
    Q_OBJECT

public:
    ModelLoader(QObject *model);
    void run() override;

    QObject *model;
    string cfg_file;
    string weights_file;
    int gpu_id;

signals:
    void done(int);
};

class Model : QObject
{
    Q_OBJECT

public:
    Model(QMainWindow *parent);
    void initialize(QString cfg_file, QString weights_file, QString names_file, int gpu_id = 0);
    image_t get_image(Frame *vp);
    image_t hw_get_image(Frame *vp);
    void show_console_result(vector<bbox_t> const result_vec, vector<string> const obj_names, int frame_id = -1);
    vector<bbox_t> infer(Frame *vp, float detection_threshold = 0.2f);

    //uint8_t *ptr_image = nullptr;
    //float *ptr_normalized = nullptr;

    QMainWindow *mainWindow;
    Detector *detector = nullptr;
    ModelLoader *modelLoader;
    WaitBox *waitBox;

    bool show_wait_box = true;

    AVExceptionHandler av;
    CudaExceptionHandler eh;

signals:
    void msg(const QString&);
};

#endif // MODEL_H
