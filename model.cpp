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
        image_t img = hw_get_image(vp);
        result = detector->detect(img, detection_threshold);
        detector->free_image(img);
    }
    catch(const exception &e) {
        cout << "Model::infer error: " << e.what() << endl;
        QString error_msg = QString("Model::infer error: %1").arg(e.what());
        emit msg(error_msg);
    }

    return result;
}

image_t Model::get_image(Frame *vp)
{
    image_t img;

    img.h = vp->frame->height;
    img.w = vp->frame->width;
    img.c = 3;
    img.data = (float*)malloc(sizeof(float) * img.w * img.h * 3);

    for (int y = 0; y < img.h; y++) {
        for (int x = 0; x < img.w; x++) {
            int i = y * img.w + x;
            img.data[i] = (float)vp->frame->data[0][i] / 255.0f;
        }
    }

    return img;
}

image_t Model::hw_get_image(Frame *vp)
{
    int width = vp->width;
    int height = vp->height;
    int frame_size = width * height;

    Npp8u *pSrc;
    Npp32f *pNorm;
    image_t img;
    img.h = vp->frame->height;
    img.w = vp->frame->width;
    img.c = 3;
    img.data = (float*)malloc(sizeof(float) * img.w * img.h * 3);

    try {
        eh.ck(cudaMalloc((void**)(& pSrc), sizeof(Npp8u) * frame_size), "Allocate device source buffer");
        eh.ck(cudaMemcpy(pSrc, vp->frame->data[0], sizeof(Npp8u) * frame_size, cudaMemcpyHostToDevice), "Copy frame picture data buffer to device");
        eh.ck(cudaMalloc((void **)(&pNorm), sizeof(Npp32f) * frame_size), "Allocate a float buffer version of source for device");
        eh.ck(nppsConvert_8u32f(pSrc, pNorm, frame_size), "Convert frame date to float");
        eh.ck(nppsDivC_32f_I(255.0f, pNorm, frame_size), "normalize frame data");
        eh.ck(cudaMemcpy(img.data, pNorm, sizeof(Npp32f) * frame_size, cudaMemcpyDeviceToHost), "copy normalized frame data to img");
        eh.ck(cudaFree(pNorm));;
        eh.ck(cudaFree(pSrc));
    }
    catch (const exception& e) {
        cout << e.what() << endl;
    }

    return img;
}

void Model::initialize(QString cfg_file, QString weights_file, QString names_file, int gpu_id)
{
    if (detector != nullptr)
        delete detector;

    MW->cfg_file = cfg_file;
    MW->weights_file = weights_file;
    MW->names_file = names_file;

    if (!QFile::exists(cfg_file)) {
        QString str;
        QTextStream(&str) << "Unable to load config file: " << cfg_file;
        QMessageBox::critical(MW, "Model Config Load Error", str);
        return;
    }

    if (!QFile::exists(weights_file)) {
        QString str;
        QTextStream(&str) << "Unable to load weights file: " << weights_file;
        QMessageBox::critical(MW, "Model Weights Load Error", str);
        return;
    }


    ifstream file(names_file.toLatin1().data());
    if (!file.is_open()) {
        QString str;
        QTextStream(&str) << "Unable to load model names file: " << names_file;
        QMessageBox::critical(MW, "Model Names Load Error", str);
        return;
    }

    for (string line; getline(file, line);)
        MW->obj_names.push_back(line);


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

