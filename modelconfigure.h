#ifndef MODELCONFIGURE_H
#define MODELCONFIGURE_H

#include <QMainWindow>
#include <QObject>
#include <QWidget>
#include <QSettings>
#include <QCheckBox>
#include <QGroupBox>

#include "Utilities/directorysetter.h"
#include "Utilities/filesetter.h"
#include "Utilities/paneldialog.h"
#include "Utilities/numbertextbox.h"

class ModelConfigure : public QWidget
{
    Q_OBJECT

public:
    ModelConfigure(QMainWindow *parent);
    QSize getModelDimensions();

    QMainWindow *mainWindow;
    //DirectorySetter *filePath;
    FileSetter *namesFile;
    FileSetter *cfgFile;
    FileSetter *weightsFile;
    QCheckBox *initializeModelOnStartup;
    QSize modelDimensions;
    NumberTextBox *modelWidth;
    NumberTextBox *modelHeight;
    QPushButton *setDims;


public slots:
    void test();
    //void setFileDirectory(QString);
    void setNamesFile(const QString&);
    void setCfgFile(const QString&);
    void setWeightsFile(const QString&);
    void setInitializeModelOnStartup(int);
    void loadModel();
    void clearModel();
    void setModelDimensions();
    void cfgEdited(const QString &);
    void clearSettings();
    void close();

};

class ModelConfigureDialog : public PanelDialog
{
    Q_OBJECT

public:
    ModelConfigureDialog(QMainWindow *parent);
    int getDefaultWidth() override;
    int getDefaultHeight() override;
    ModelConfigure *model_configure;

    int defaultWidth = 480;
    int defaultHeight = 240;

};

#endif // CONFIGURE_H
