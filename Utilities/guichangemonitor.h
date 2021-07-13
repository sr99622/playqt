#ifndef GUICHANGEMONITOR_H
#define GUICHANGEMONITOR_H

#include <QObject>
#include <QRunnable>
#include <QMutex>
#include <QThread>

class GuiChangeMonitor : public QObject, public QRunnable
{
    Q_OBJECT

public:
    GuiChangeMonitor();
    void run() override;
    void setCountdown(int arg);

    QMutex mutex;
    bool running = false;
    int countdown;
    const int timeout = 100;

signals:
    void done();

};

#endif // GUICHANGEMONITOR_H
