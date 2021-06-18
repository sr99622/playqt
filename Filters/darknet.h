#ifndef DARKNET_H
#define DARKNET_H

#include <QMainWindow>
#include <QCheckBox>
#include <QRunnable>

#include "Ffplay/Frame.h"
#include "Filters/filter.h"
#include "Utilities/waitbox.h"
#include "Utilities/filesetter.h"
#include "Utilities/numbertextbox.h"
#include "yolo_v2_class.hpp"

class DarknetLoader : public QObject, public QRunnable
{
    Q_OBJECT

public:
    DarknetLoader(QObject *model);
    void run() override;

    QObject *model;
    string cfg_file;
    string weights_file;
    int gpu_id;

signals:
    void done(int);
};

class DarknetModel : QObject
{
    Q_OBJECT

public:
    DarknetModel(QMainWindow *parent);
    void initialize(QString cfg_file, QString weights_file, QString names_file, int gpu_id = 0);
    image_t get_image(Frame *vp);
    image_t hw_get_image(Frame *vp);
    void show_console_result(vector<bbox_t> const result_vec, vector<string> const obj_names, int frame_id = -1);
    vector<bbox_t> infer(Frame *vp, float detection_threshold = 0.2f);
    const QString getName(int obj_id);

    QMainWindow *mainWindow;
    Detector *detector = nullptr;
    DarknetLoader *loader;
    WaitBox *waitBox;
    vector<string> obj_names;
    CudaExceptionHandler eh;

signals:
    void msg(const QString&);
};


class Darknet : public Filter
{
    Q_OBJECT

public:
    Darknet(QMainWindow *parent);
    void filter(Frame *vp) override;
    void initialize() override;
    void saveSettings(QSettings *settings) override;
    void restoreSettings(QSettings *settings) override;

    QSize getModelDimensions();
    DarknetModel *model = nullptr;

    QMainWindow *mainWindow;
    FileSetter *names;
    FileSetter *cfg;
    FileSetter *weights;
    QCheckBox *initOnStartup;
    QSize modelDimensions;
    NumberTextBox *modelWidth;
    NumberTextBox *modelHeight;
    QPushButton *setDims;

public slots:
    void setNames(const QString&);
    void setCfg(const QString&);
    void setWeights(const QString&);
    void setInitOnStartup(int);
    void cfgEdited(const QString &text);
    void setModelDimensions();
    void loadModel();
    void clearModel();
    void clearSettings();

};

#endif // DARKNET_H
