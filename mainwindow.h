#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QGridLayout>
#include <QMenuBar>
#include <QMenu>
#include <QFile>
#include <QMessageBox>
#include <QSettings>
#include <QApplication>
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
#include "Filters/filterpanel.h"
#include "optionpanel.h"

using namespace std;

#define MW dynamic_cast<MainWindow*>(mainWindow)

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    void closeEvent(QCloseEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;
    void moveEvent(QMoveEvent *event) override;
    void initializeSDL();
    void get_names(QString names_file);
    void printSizes();

    CommandOptions *co;
    MainPanel *mainPanel;
    Display display;
    AVPacket flush_pkt;
    EventHandler e;
    VideoState *is;
    AVExceptionHandler av;
    QSettings *settings;

    Model *model = nullptr;
    ModelConfigureDialog *modelConfigureDialog;
    QString cfg_file;
    QString weights_file;
    QString names_file;
    vector<string> obj_names;
    bool initializeModelOnStartup = false;

    FilterDialog *filterDialog;
    OptionDialog *optionDialog;

signals:
    //void initComplete();

public slots:
    void runLoop();
    void fileMenuAction(QAction*);
    void toolsMenuAction(QAction*);
    //void initCallback();
    void test();
};
#endif // MAINWINDOW_H
