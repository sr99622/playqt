#include "model.h"
#include "mainwindow.h"

#include <fstream>
#include <iomanip>
#include <QThreadPool>

ModelLoader::ModelLoader(QObject *model)
{
    this->model = model;
}

void ModelLoader::run()
{
    ((Model*)model)->detector = new Detector(cfg_file, weights_file, gpu_id);
    emit done(0);
}

Model::Model(QMainWindow *parent) : QObject(parent)
{
    mainWindow = parent;
    waitBox = new WaitBox(mainWindow);
    modelLoader = new ModelLoader(this);
    connect(modelLoader, SIGNAL(done(int)), waitBox, SLOT(done(int)));
    //connect(this, SIGNAL(msg(const QString&)), mainWindow, SLOT(msg(const QString&)));
}

vector<bbox_t> Model::infer(Frame *vp, float detection_threshold)
{
    vector<bbox_t> result;

    try {
        image_t img = get_image(vp);
        result = detector->detect(img, detection_threshold);
        detector->free_image(img);
    }
    catch(exception &e) {
        cout << "Model::infer error: " << e.what() << endl;
        QString error_msg = QString("Model::infer error: %1").arg(e.what());
        emit msg(error_msg);
    }

    return result;
}

void Model::initialize(QString cfg_file, QString weights_file, QString names_file, int gpu_id)
{
    if (detector != nullptr)
        delete detector;


    ifstream file(names_file.toLatin1().data());
    if (!file.is_open())
        return;

    for (string line; getline(file, line);)
        obj_names.push_back(line);


    if (show_wait_box) {
        modelLoader->cfg_file = cfg_file.toStdString();
        modelLoader->weights_file = weights_file.toStdString();
        modelLoader->gpu_id = gpu_id;
        QThreadPool::globalInstance()->tryStart(modelLoader);
        waitBox->exec();
    }
    else {
        detector = new Detector(cfg_file.toStdString(), weights_file.toStdString(), gpu_id);
    }

}

image_t Model::get_image(Frame *vp)
{
    AVFrame *frame            = NULL;
    image_t img;

    try {
        frame = vp->frame;
        int width = frame->width;
        int height = frame->height;

        img.h = height;
        img.w = width;
        img.c = 3;
        img.data = (float*)malloc(sizeof(float) * width * height * 3);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < frame->linesize[0]; x++) {
                int i = y * frame->linesize[0] + x;
                img.data[i] = (float)frame->data[0][i] / 255.0f;
            }
        }

    }
    catch (AVException *e) {
        cout << "ERROR " << av.tag(e->cmd_tag).toLatin1().data() << " " << e->error_text << endl;
    }

    return img;
}

void Model::show_console_result(vector<bbox_t> const result_vec, vector<string> const obj_names, int frame_id)
{
    if (frame_id >= 0) cout << " Frame: " << frame_id << endl;
    for (auto &i : result_vec) {
        if (obj_names.size() > i.obj_id) cout << obj_names[i.obj_id] << " - ";

        cout << "obj_id = " << i.obj_id << ", x = " << i.x << ", y = " << i.y
             << ", w = " << i.w << ", h = " << i.h
             << ", prob = " << i.prob << endl;

    }
}

