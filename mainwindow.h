#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QGridLayout>
#include <QMenuBar>
#include <QMenu>
#include <QFile>
#include <QMessageBox>
#include <QSettings>
#include <iostream>
#include <string>
#include <fstream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "npp.h"
#include "nppi.h"
#include "npps.h"

#include "mainpanel.h"
#include "Display.h"
#include "CommandOptions.h"
#include "EventHandler.h"
#include "avexception.h"
#include "cudaexception.h"
#include "model.h"
#include "modelconfigure.h"

using namespace std;

#define MW dynamic_cast<MainWindow*>(mainWindow)

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    void closeEvent(QCloseEvent *event) override;
    void initializeSDL();
    void get_names(QString names_file);

    CommandOptions *co;
    MainPanel *mainPanel;
    Display display;
    AVPacket flush_pkt;
    EventHandler e;
    VideoState *is;
    Model *model;
    ModelConfigureDialog *modelConfigureDialog;
    AVExceptionHandler av;
    QSettings *settings;

    QString cfg_file;
    QString weights_file;
    QString names_file;
    vector<string> obj_names;
    bool initializeModelOnStartup = false;

public slots:
    void runLoop();
    void fileMenuAction(QAction*);
    void toolsMenuAction(QAction*);
    void test();
};
#endif // MAINWINDOW_H
