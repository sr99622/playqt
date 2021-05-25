#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QGridLayout>
#include <iostream>
#include <string>
#include <fstream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>

#include "npp.h"
#include "nppi.h"
#include "npps.h"

#include "mainpanel.h"
#include "Display.h"
#include "CommandOptions.h"
#include "EventHandler.h"
#include "avexception.h"
#include "model.h"

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

    uint8_t *ptr_image;
    float *ptr_data;

    CommandOptions *co;
    MainPanel *mainPanel;
    Display display;
    AVPacket flush_pkt;
    EventHandler e;
    VideoState *is;
    Model *model;
    AVExceptionHandler av;

public slots:
    void runLoop();
    void test();
};
#endif // MAINWINDOW_H
