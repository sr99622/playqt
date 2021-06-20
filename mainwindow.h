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
#include <QSplitter>
#include <QThreadPool>
#include <QTabWidget>
#include <QStandardPaths>
#include <QStatusBar>
#include <QGroupBox>
#include <QTimer>
#include <QMutex>

#include <iostream>
#include <string>
#include <fstream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>

#include "npp.h"
#include "nppi.h"
#include "npps.h"

#include "Ffplay/Display.h"
#include "Ffplay/CommandOptions.h"
#include "Ffplay/EventHandler.h"
#include "Utilities/avexception.h"
#include "Utilities/cudaexception.h"
#include "Utilities/filepanel.h"
#include "Utilities/messagebox.h"
#include "Filters/filterpanel.h"
#include "Filters/filterchain.h"
#include "mainpanel.h"
#include "optionpanel.h"
#include "camerapanel.h"
#include "streampanel.h"
#include "viewer.h"

enum CustomEventCode {
    FILE_POSITION_UPDATE,
    SLIDER_POSITION_UPDATE,
    GUI_EVENT_UPDATE
};

using namespace std;

#define MW dynamic_cast<MainWindow*>(mainWindow)

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(CommandOptions *co, QWidget *parent = nullptr);
    ~MainWindow();
    void closeEvent(QCloseEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;
    void moveEvent(QMoveEvent *event) override;
    void initializeSDL();
    void getNames(QString names_file);

    QString filename;

    CommandOptions *co;
    MainPanel *mainPanel;
    Display display;
    AVPacket flush_pkt;
    EventHandler *e;
    VideoState *is = nullptr;
    AVExceptionHandler av;
    QSettings *settings;
    QSplitter *splitter;
    FilePanel *videoPanel;
    FilePanel *picturePanel;
    FilePanel *audioPanel;
    CameraPanel *cameraPanel;
    StreamPanel *streamPanel;
    QTabWidget *tabWidget;
    MessageBox *messageBox;
    ParameterDialog *parameterDialog;
    QStatusBar *status;

    FilterDialog *filterDialog;
    FilterChain *filterChain;
    OptionDialog *optionDialog;

    ViewerDialog *viewerDialog;

    Uint32 sdlCustomEventType;
    QTimer *timer;
    QScreen *screen;

public slots:
    void runLoop();
    void poll();
    void fileMenuAction(QAction*);
    void toolsMenuAction(QAction*);
    void helpMenuAction(QAction*);
    void showHelp(const QString&);
    void msg(const QString&);
    void test();

};
#endif // MAINWINDOW_H
