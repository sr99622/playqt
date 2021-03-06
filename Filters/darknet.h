#ifndef DARKNET_H
#define DARKNET_H

#include <QMainWindow>
#include <QCheckBox>
#include <QRunnable>
#include <QMutex>
#include <QSlider>

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


class DarknetModel : public QObject
{
    Q_OBJECT

public:
    DarknetModel(QMainWindow *parent);
    void initialize(QString cfg_file, QString weights_file, QString names_file, int gpu_id = 0);
    image_t get_image(Frame *vp);
    image_t hw_get_image(Frame *vp);
    void show_console_result(vector<bbox_t> const result_vec, vector<string> const obj_names, int frame_id = -1);
    vector<bbox_t> infer(Frame *vp, float detection_threshold = 0.2f);

    QMainWindow *mainWindow;
    Detector *detector = nullptr;
    DarknetLoader *loader;
    WaitBox *waitBox;
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
    void autoSave() override;

    QString getName(int obj_id) const;
    QSize getModelDimensions();
    DarknetModel *model = nullptr;
    bool loading = false;

    QMainWindow *mainWindow;
    FileSetter *names;
    FileSetter *cfg;
    FileSetter *weights;
    QSize modelDimensions;
    NumberTextBox *modelWidth;
    NumberTextBox *modelHeight;
    QPushButton *setDims;
    QSlider *sldrThreshold;
    QLabel *lblThreshold;
    float threshold = 0.2f;

    vector<bbox_t> result;
    vector<int> showableObjIDs;
    vector<string> obj_names;
    vector<YUVColor> obj_drawn;

    const QString cfgKey       = "DarknetModel/cfg";
    const QString weightsKey   = "DarknetModel/weights";
    const QString namesKey     = "DarknetModel/names";
    const QString thresholdKey = "DarknetModel/threshold";

signals:
    void send(vector<bbox_t>*);
    void namesSet();

public slots:
    void setNames(const QString&);
    void setCfg(const QString&);
    void setWeights(const QString&);
    void cfgEdited(const QString &text);
    void setModelDimensions();
    void loadModel();
    void clearModel();
    void clearSettings();
    void setThreshold(int);
    void draw(int, const YUVColor&);

};

#endif // DARKNET_H
