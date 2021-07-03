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
#include <QHeaderView>
#include <QThreadPool>
#include <QTabWidget>
#include <QStandardPaths>
#include <QStatusBar>
#include <QGroupBox>
#include <QTimer>
#include <QMutex>
#include <QRunnable>
#include <QShortcut>

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
    REWIND,
    FASTFORWARD,
    PAUSE
};

#define APP_DEFAULT_WIDTH 1200
#define APP_DEFAULT_HEIGHT 600

using namespace std;

#define MW dynamic_cast<MainWindow*>(mainWindow)
#define TS QTime::currentTime().toString("hh:mm:ss.zzz").toStdString()

class Quitter : public QObject, public QRunnable
{
    Q_OBJECT

public:
    Quitter(QMainWindow *parent);
    void run() override;

    QMainWindow *mainWindow;

signals:
    void done();

};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(CommandOptions *co, QWidget *parent = nullptr);
    ~MainWindow();
    void closeEvent(QCloseEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;
    void moveEvent(QMoveEvent *event) override;
    void paintEvent(QPaintEvent *event) override;
    void initializeSDL();

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
    Quitter *quitter;

    Uint32 sdlCustomEventType;
    QTimer *timer;
    QScreen *screen;

    const QString geometryKey = "MainWindow/geometry";
    const QString splitterKey = "MainWindow/splitter";
    const QString videoPanelHeaderKey = "MainWindow/VideoPanel/header";
    const QString videoPanelDirKey = "MainWindow/VideoPanel/dir";
    const QString picturePanelHeaderKey = "MainWindow/PicturePanel/header";
    const QString picturePanelDirKey = "MainWindow/PicturePanel/dir";
    const QString audioPanelHeaderKey = "MainWindow/AudioPanel/header";
    const QString audioPanelDirKey = "MainWindow/AudioPanel/dir";

public slots:
    void runLoop();
    void poll();
    void fileMenuAction(QAction*);
    void toolsMenuAction(QAction*);
    void helpMenuAction(QAction*);
    void showHelp(const QString&);
    void msg(const QString&);
    void test();
    void ping(const vector<bbox_t>*);
    void guiUpdate(int);
    void start();

};
#endif // MAINWINDOW_H
