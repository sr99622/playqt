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
#include <QToolTip>
#include <QFileDialog>
#include <QColorDialog>
#include <QScrollBar>

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
#include "Utilities/messagedialog.h"
#include "Filters/filterpanel.h"
#include "Filters/filterchain.h"
#include "Cameras/camerapanel.h"
#include "mainpanel.h"
#include "parameterpanel.h"
#include "optionpanel.h"
#include "countpanel.h"
#include "configpanel.h"

enum CustomEventCode {
    FILE_POSITION_UPDATE,
    SLIDER_POSITION_UPDATE,
    FLUSH
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

class Launcher : public QObject, public QRunnable
{
    Q_OBJECT

public:
    Launcher(QMainWindow *parent);
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
    void showEvent(QShowEvent *event) override;
    void initializeSDL();
    void openFile();
    void applyStyle(const ColorProfile& profile);

    ConfigPanel *config();
    ControlPanel *control();
    FilterPanel *filter();
    ParameterPanel *parameter();
    CountPanel *count();
    QLabel *display();
    DisplayContainer *dc();

    QString filename;

    CommandOptions *co;
    MainPanel *mainPanel;
    Display ffDisplay;
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
    QTabWidget *tabWidget;
    MessageDialog *messageDialog;
    ParameterDialog *parameterDialog;
    ConfigDialog *configDialog;

    FilterDialog *filterDialog;
    FilterChain *filterChain;
    OptionDialog *optionDialog;

    CountDialog *countDialog;
    Quitter *quitter = nullptr;
    Launcher *launcher;
    QString style;

    Uint32 sdlCustomEventType;
    QTimer *timer;
    QTimer *autoSaveTimer;
    QScreen *screen;

    bool changed = false;
    const QString geometryKey = "MainWindow/geometry";
    const QString splitterKey = "MainWindow/splitter";

    bool clearSettings = false;

public slots:
    void runLoop();
    void poll();
    void menuAction(QAction*);
    void showHelp(const QString&);
    void msg(const QString&);
    void test();
    void start();
    void splitterMoved(int, int);
    void autoSave();

};
#endif // MAINWINDOW_H
